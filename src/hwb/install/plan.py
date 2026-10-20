"""Install helpers: build an explicit, reviewable plan of commands per distro, then
(only with --apply, as root, after confirmation) execute it step by step.

Plans are pure data so they can be printed, logged and unit-tested.
Package names drift between releases — keep them in the constants below.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core import proc
from ..core.context import RunContext
from ..core.errors import HwbError, UnsupportedPlatformError, UserAbortError
from ..core.platform import ARCH, DEBIAN, OSInfo, is_linux, is_root
from ..core.result import CheckResult, Status

log = logging.getLogger(__name__)

DEFAULT_DRIVER_BRANCH = "580"  # Ubuntu server-open branch; review when new LTS branches land
NVIDIA_REPO = "https://developer.download.nvidia.com/compute/cuda/repos"
CUDA_KEYRING = "cuda-keyring_1.1-1_all.deb"
STEP_TIMEOUT_S = 1800.0

APT = ("env", "DEBIAN_FRONTEND=noninteractive", "apt-get")
DEBIAN_TOOLS = (
    "fio",
    "stress-ng",
    "sysbench",
    "lm-sensors",
    "pciutils",
    "dmidecode",
    "nvtop",
    "curl",
)
ARCH_TOOLS = (
    "fio",
    "stress-ng",
    "sysbench",
    "lm_sensors",
    "pciutils",
    "dmidecode",
    "nvtop",
    "curl",
)


@dataclass(frozen=True)
class Step:
    title: str
    argv: tuple[str, ...]


def _nvidia_repo_slug(osi: OSInfo) -> str:
    arch = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(osi.machine)
    if arch is None:
        raise UnsupportedPlatformError(f"no NVIDIA repo for architecture {osi.machine}")
    if osi.id == "ubuntu":
        return f"ubuntu{osi.version_id.replace('.', '')}/{arch}"
    if osi.id == "debian":
        return f"debian{osi.version_id.split('.')[0]}/{arch}"
    raise UnsupportedPlatformError(f"no NVIDIA CUDA repo mapping for {osi.id} {osi.version_id}")


def _keyring_steps(osi: OSInfo, workdir: Path) -> list[Step]:
    deb = workdir / CUDA_KEYRING
    url = f"{NVIDIA_REPO}/{_nvidia_repo_slug(osi)}/{CUDA_KEYRING}"
    return [
        Step(
            "download NVIDIA CUDA repo keyring",
            ("curl", "-fsSL", "--proto", "=https", "-o", str(deb), url),
        ),
        Step("install NVIDIA CUDA repo keyring", ("dpkg", "-i", str(deb))),
        Step("refresh package index", (*APT, "update")),
    ]


def _arch_headers_pkg(kernel: str) -> str:
    for flavor in ("lts", "zen", "hardened"):
        if kernel.endswith(f"-{flavor}"):
            return f"linux-{flavor}-headers"
    return "linux-headers"


def driver_plan(osi: OSInfo, workdir: Path, branch: str = DEFAULT_DRIVER_BRANCH) -> list[Step]:
    """Open kernel modules everywhere: required for Blackwell, recommended by NVIDIA for Turing+."""
    if osi.id == "ubuntu":
        return [
            Step("refresh package index", (*APT, "update")),
            Step("install kernel headers", (*APT, "install", "-y", f"linux-headers-{osi.kernel}")),
            Step(
                f"install NVIDIA driver {branch} (server, open modules)",
                (*APT, "install", "-y", f"nvidia-driver-{branch}-server-open"),
            ),
        ]
    if osi.family == DEBIAN:
        return [
            Step("install kernel headers", (*APT, "install", "-y", f"linux-headers-{osi.kernel}")),
            *_keyring_steps(osi, workdir),
            Step(
                "install NVIDIA driver (open modules, latest branch)",
                (*APT, "install", "-y", "nvidia-open"),
            ),
        ]
    if osi.family == ARCH:
        return [
            Step(
                "upgrade system and install NVIDIA open DKMS driver",
                (
                    "pacman",
                    "-Syu",
                    "--needed",
                    "--noconfirm",
                    _arch_headers_pkg(osi.kernel),
                    "nvidia-open-dkms",
                    "nvidia-utils",
                ),
            ),
        ]
    raise UnsupportedPlatformError(f"unsupported distro: {osi.pretty_name}")


def cuda_plan(osi: OSInfo, workdir: Path, version: str | None = None) -> list[Step]:
    """CUDA toolkit (nvcc etc.). Optional: PyTorch/vLLM wheels ship their own CUDA runtime."""
    if osi.family == DEBIAN:
        pkg = f"cuda-toolkit-{version.replace('.', '-')}" if version else "cuda-toolkit"
        return [*_keyring_steps(osi, workdir), Step(f"install {pkg}", (*APT, "install", "-y", pkg))]
    if osi.family == ARCH:
        if version:
            log.warning("Arch ships a single cuda package; ignoring --cuda-version %s", version)
        return [Step("install cuda", ("pacman", "-S", "--needed", "--noconfirm", "cuda"))]
    raise UnsupportedPlatformError(f"unsupported distro: {osi.pretty_name}")


def tools_plan(osi: OSInfo) -> list[Step]:
    if osi.family == DEBIAN:
        return [
            Step("refresh package index", (*APT, "update")),
            Step("install benchmark tools", (*APT, "install", "-y", *DEBIAN_TOOLS)),
        ]
    if osi.family == ARCH:
        return [
            Step(
                "install benchmark tools", ("pacman", "-S", "--needed", "--noconfirm", *ARCH_TOOLS)
            )
        ]
    raise UnsupportedPlatformError(f"unsupported distro: {osi.pretty_name}")


def format_plan(steps: Sequence[Step]) -> str:
    return "\n".join(f"  {i}. {s.title}\n     $ {' '.join(s.argv)}" for i, s in enumerate(steps, 1))


def _confirm(assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise UserAbortError("refusing to modify the system non-interactively without --yes")
    if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        raise UserAbortError("aborted by user")


def execute(
    ctx: RunContext, target: str, steps: Sequence[Step], *, apply: bool, assume_yes: bool
) -> None:
    ctx.sections["install_plan"] = {"target": target, "steps": [s.__dict__ for s in steps]}
    print(f"Plan for '{target}':\n{format_plan(steps)}")
    if not apply:
        ctx.add(
            CheckResult(f"install.{target}", Status.SKIP, "dry run; re-run with --apply as root")
        )
        return
    if not is_linux() or not is_root():
        raise HwbError("--apply requires root on Linux (run with sudo)")
    _confirm(assume_yes)

    for i, step in enumerate(steps, 1):
        log.info("step %d/%d: %s", i, len(steps), step.title)
        res = proc.run(step.argv, timeout=STEP_TIMEOUT_S)
        name = f"install.{target}.{i}"
        if not res.ok:
            ctx.add(
                CheckResult(
                    name,
                    Status.FAIL,
                    f"{step.title} failed (rc={res.returncode}{', timed out' if res.timed_out else ''})",
                    data={"stderr_tail": res.stderr[-4000:]},
                    hint=f"full output in {ctx.run_dir / 'hwb.log.jsonl'}",
                )
            )
            return
        ctx.add(
            CheckResult(
                name, Status.PASS, step.title, data={"duration_s": round(res.duration_s, 1)}
            )
        )
    if target == "driver":
        log.warning("reboot required, then run: hwb detect")
