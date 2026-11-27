"""Run a preset's server in docker: the quick way to try builds that are not on PyPI.

The command is built here as data (``docker_command``) so it can be shown, tested on macOS and
stored in the report; ``serve.run`` spawns it like a local vLLM and removes the container in
``finally``. What the container may touch is explicit: the GPUs it was given, the Hugging Face
hub (read-only when it is another user's cache), a local model directory (read-only) and hwb's
own cache under $HWB_HOME. It runs as the invoking user unless the template says the image needs
root, so files it writes into the shared cache stay theirs. hwb never adds sudo: docker must
work for this user (docker group or root).
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from . import presets

NAME_PREFIX = "hwb-serve-"
HUB = "/hf-hub"  # the Hugging Face hub inside the container (HF_HUB_CACHE)
HOME = "/hwb-home"  # HOME for a non-root container: Triton/vLLM/FlashInfer kernel caches persist
# Environment handed to the container when hwb set it (values shown in the printed command).
PASS_ENV = ("HF_HUB_OFFLINE", "VLLM_SKIP_P2P_CHECK")
# Secrets: passed by name only, so docker reads the value from hwb's environment and it is never
# printed or stored in the report.
SECRET_ENV = ("HF_TOKEN",)
# Set by the operator: NVIDIA_DISABLE_REQUIRE=1 skips the toolkit's "cuda>=X.Y" check of an image
# built with a newer CUDA minor than the driver reports (CUDA 13.2 image on R580 = 13.0 runs, by
# minor-version compatibility).
OPERATOR_ENV = ("NVIDIA_DISABLE_REQUIRE",)
AMD_DEVICES = ("/dev/kfd", "/dev/dri")


@dataclass(frozen=True)
class Mounts:
    hub: Path | None  # host Hugging Face hub directory
    hub_readonly: bool
    home: Path  # host directory for HOME (non-root containers)
    cache: Path  # host directory for the template's own cache


def device_gids(paths: tuple[str, ...] = AMD_DEVICES) -> list[int]:
    """Groups owning the AMD device nodes (render, video): a non-root container needs them, and
    by number, since the image's /etc/group may not know the host's names."""
    gids: set[int] = set()
    for p in paths:
        root = Path(p)
        nodes = [root] if not root.is_dir() else sorted(root.iterdir())
        for n in nodes:
            try:
                st = n.stat()
            except OSError:
                continue
            if stat.S_ISCHR(st.st_mode):
                gids.add(st.st_gid)
    return sorted(gids)


def invoking_user() -> tuple[int, int]:
    """uid:gid for --user: the sudo caller's when hwb runs under sudo."""
    env = os.environ
    if os.geteuid() == 0 and env.get("SUDO_UID") and env.get("SUDO_GID"):
        return int(env["SUDO_UID"]), int(env["SUDO_GID"])
    return os.getuid(), os.getgid()


def docker_command(
    preset: presets.Preset,
    spec: presets.Container,
    *,
    name: str,
    port: int,
    vendor: str,
    mounts: Mounts,
    gpus: list[str] | None = None,  # None: all of them
    model_arg: str | None = None,  # a local directory instead of the HF id
    env: dict[str, str] | None = None,
    user: tuple[int, int] | None = None,
    gids: list[int] | None = None,
    environ: dict[str, str] | None = None,  # hwb's own environment, for secrets
) -> list[str]:
    env = env or {}
    environ = dict(os.environ) if environ is None else environ
    argv = ["docker", "run", "--rm", "--name", name, "--ipc=host"]
    argv += ["-p", f"127.0.0.1:{port}:{spec.port}"]
    if vendor == "amd":
        for d in AMD_DEVICES:
            argv += ["--device", d]
        if not spec.root:
            for g in gids or []:
                argv += ["--group-add", str(g)]
        if gpus is not None:
            argv += ["-e", f"HIP_VISIBLE_DEVICES={','.join(gpus)}"]
    else:
        # docker wants the quotes for a list: --gpus '"device=0,1"'
        argv += ["--gpus", f'"device={",".join(gpus)}"' if gpus is not None else "all"]
        for k in OPERATOR_ENV:
            if environ.get(k):
                argv += ["-e", f"{k}={environ[k]}"]
    if not spec.root and user is not None:
        # torch's inductor asks getpass for a user name: without USER it fails on a uid the
        # image does not know.
        argv += ["--user", f"{user[0]}:{user[1]}", "-e", f"HOME={HOME}", "-e", "USER=hwb"]
        argv += ["-v", f"{mounts.home}:{HOME}"]
    if spec.cache:
        argv += ["-v", f"{mounts.cache}:{spec.cache}"]
    if "instanttensor" in preset.args:  # io_uring with registered buffers: locked memory
        argv += ["--ulimit", "memlock=-1:-1"]
    if not spec.own_command:
        if mounts.hub is not None:
            argv += ["-v", f"{mounts.hub}:{HUB}" + (":ro" if mounts.hub_readonly else "")]
            argv += ["-e", f"HF_HUB_CACHE={HUB}"]
        if model_arg and Path(model_arg).is_absolute():
            argv += ["-v", f"{model_arg}:{model_arg}:ro"]
        for k in PASS_ENV:
            if k in env:
                argv += ["-e", f"{k}={env[k]}"]
    for k, v in spec.env:
        argv += ["-e", f"{k}={v}"]
    for k in SECRET_ENV:
        if environ.get(k):
            argv += ["-e", k]
    argv.append(spec.image)
    if not spec.own_command:  # the image's entrypoint is `vllm serve`
        argv += preset.command(host="0.0.0.0", port=spec.port, model=model_arg)[2:]  # noqa: S104
    return argv


def remove_command(name: str) -> list[str]:
    return ["docker", "rm", "-f", name]
