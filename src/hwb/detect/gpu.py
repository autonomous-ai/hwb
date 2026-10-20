"""GPU driver / CUDA health checks.

Two phases: ``gather()`` reads the machine (sysfs, procfs, nvidia-smi, ...) into a
GpuFacts snapshot, tolerating every individual probe failing. The checks are pure
functions of that snapshot, so they are unit-testable off-box.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import nvsmi
from ..core import proc
from ..core.context import RunContext
from ..core.errors import HwbError
from ..core.log import fields
from ..core.platform import os_info
from ..core.result import CheckResult, Status

log = logging.getLogger(__name__)

NVIDIA_VENDOR = "0x10de"
# Blackwell (sm_100/sm_120, e.g. RTX 5090) requires the open kernel modules and >= R570.
MIN_BLACKWELL_DRIVER = (570,)
_DRIVER_RE = re.compile(r"Kernel Module(?: for \S+)?\s+(\d+\.\d+(?:\.\d+)?)")
_NVCC_RE = re.compile(r"release (\d+\.\d+)")
_EFI_DIR = Path("/sys/firmware/efi")
_SECURE_BOOT_VAR = _EFI_DIR / "efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c"
_XID_SAMPLE = 20


@dataclass(frozen=True)
class PciGpu:
    address: str
    device_id: str
    pci_class: str
    driver: str | None


@dataclass(frozen=True)
class KernelDriver:
    version: str
    open_module: bool


@dataclass
class GpuFacts:
    pci: list[PciGpu] = field(default_factory=list)
    modules: set[str] = field(default_factory=set)
    kernel_driver: KernelDriver | None = None
    smi_rows: list[nvsmi.Row] | None = None
    smi_error: str | None = None
    cuda_driver_api: str | None = None
    secure_boot: bool | None = None
    headers_present: bool = False
    libcuda: str | None = None
    nvcc_version: str | None = None
    xid_lines: list[str] | None = None


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v))


# ---------------------------------------------------------------- probes


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def scan_pci(root: Path = Path("/sys/bus/pci/devices")) -> list[PciGpu]:
    """NVIDIA display/3D controllers from sysfs (no pciutils needed)."""
    if not root.is_dir():
        return []
    gpus = []
    for dev in sorted(root.iterdir()):
        if _read(dev / "vendor") != NVIDIA_VENDOR:
            continue
        pci_class = _read(dev / "class") or ""
        if not pci_class.startswith("0x03"):  # skips the HDMI audio function (0x0403)
            continue
        driver_link = dev / "driver"
        driver = driver_link.resolve().name if driver_link.exists() else None
        gpus.append(PciGpu(dev.name, _read(dev / "device") or "?", pci_class, driver))
    return gpus


def loaded_modules(path: Path = Path("/proc/modules")) -> set[str]:
    text = _read(path)
    if not text:
        return set()
    return {line.split(maxsplit=1)[0] for line in text.splitlines() if line.strip()}


def parse_driver_version(text: str) -> KernelDriver | None:
    m = _DRIVER_RE.search(text)
    if not m:
        return None
    return KernelDriver(m.group(1), "Open Kernel Module" in text)


def secure_boot_enabled() -> bool | None:
    if not _EFI_DIR.exists():
        return False  # legacy BIOS boot: no Secure Boot
    try:
        data = _SECURE_BOOT_VAR.read_bytes()  # 4 attribute bytes + 1 value byte
    except OSError:
        return None
    return data[-1] == 1 if data else None


def find_libcuda() -> str | None:
    if not proc.have("ldconfig"):
        return None
    res = proc.run(["ldconfig", "-p"], timeout=30)
    for line in res.stdout.splitlines():
        if line.strip().startswith("libcuda.so.1 ") and "=>" in line:
            return line.split("=>", 1)[1].strip()
    return None


def nvcc_version() -> str | None:
    if not proc.have("nvcc"):
        return None
    m = _NVCC_RE.search(proc.run(["nvcc", "--version"], timeout=30).stdout)
    return m.group(1) if m else None


def xid_lines() -> list[str] | None:
    """NVRM Xid events from this boot's kernel log; None if the log is unreadable."""
    if not proc.have("journalctl"):
        return None
    res = proc.run(["journalctl", "-k", "-b", "--no-pager", "-o", "cat"], timeout=60)
    if not res.ok:
        return None
    return [ln for ln in res.stdout.splitlines() if "NVRM: Xid" in ln]


def gather() -> GpuFacts:
    facts = GpuFacts()
    kernel = os_info().kernel

    probes: list[tuple[str, Callable[[], None]]] = [
        ("pci", lambda: setattr(facts, "pci", scan_pci())),
        ("modules", lambda: setattr(facts, "modules", loaded_modules())),
        (
            "kernel_driver",
            lambda: setattr(
                facts,
                "kernel_driver",
                parse_driver_version(_read(Path("/proc/driver/nvidia/version")) or ""),
            ),
        ),
        ("secure_boot", lambda: setattr(facts, "secure_boot", secure_boot_enabled())),
        (
            "headers",
            lambda: setattr(
                facts, "headers_present", Path(f"/lib/modules/{kernel}/build").exists()
            ),
        ),
        ("libcuda", lambda: setattr(facts, "libcuda", find_libcuda())),
        ("nvcc", lambda: setattr(facts, "nvcc_version", nvcc_version())),
        ("xid", lambda: setattr(facts, "xid_lines", xid_lines())),
    ]
    for name, probe in probes:
        try:
            probe()
        except Exception:  # a broken probe must not hide the others
            log.debug("probe failed", exc_info=True, extra=fields(probe=name))

    try:
        facts.smi_rows = nvsmi.query_gpus()
        facts.cuda_driver_api = nvsmi.cuda_driver_version()
    except HwbError as exc:
        facts.smi_error = str(exc)
    return facts


# ---------------------------------------------------------------- checks


def check_pci(f: GpuFacts) -> CheckResult:
    if not f.pci:
        return CheckResult(
            "gpu.pci",
            Status.FAIL,
            "no NVIDIA GPU on the PCI bus",
            hint="check seating and PCIe power cables; enable Above 4G Decoding in BIOS",
        )
    return CheckResult(
        "gpu.pci",
        Status.PASS,
        f"{len(f.pci)} NVIDIA GPU(s) on PCI bus",
        data={"devices": [dataclasses.asdict(g) for g in f.pci]},
    )


def check_nouveau(f: GpuFacts) -> CheckResult:
    if "nouveau" in f.modules:
        return CheckResult(
            "gpu.nouveau",
            Status.FAIL,
            "nouveau driver is loaded",
            hint="install the NVIDIA driver (hwb install driver), which blacklists nouveau; reboot",
        )
    return CheckResult("gpu.nouveau", Status.PASS, "nouveau not loaded")


def check_kernel_module(f: GpuFacts) -> CheckResult:
    if "nvidia" not in f.modules or f.kernel_driver is None:
        return CheckResult(
            "gpu.kernel_module",
            Status.FAIL,
            "nvidia kernel module not loaded",
            hint="hwb install driver --apply, then reboot",
        )
    flavor = "open" if f.kernel_driver.open_module else "proprietary"
    return CheckResult(
        "gpu.kernel_module",
        Status.PASS,
        f"nvidia {f.kernel_driver.version} ({flavor} kernel module)",
        data=dataclasses.asdict(f.kernel_driver),
    )


def check_bound(f: GpuFacts) -> CheckResult:
    if not f.pci:
        return CheckResult("gpu.bound", Status.SKIP, "no GPUs on PCI bus")
    unbound = [g.address for g in f.pci if g.driver != "nvidia"]
    if unbound:
        return CheckResult(
            "gpu.bound",
            Status.FAIL,
            f"{len(unbound)} GPU(s) not bound to the nvidia driver: {', '.join(unbound)}",
            data={"drivers": {g.address: g.driver for g in f.pci}},
            hint="check dmesg for NVRM errors (unsupported GPU for this driver branch?)",
        )
    return CheckResult("gpu.bound", Status.PASS, "all GPUs bound to nvidia")


def check_secure_boot(f: GpuFacts) -> CheckResult:
    if f.secure_boot is None:
        return CheckResult("gpu.secure_boot", Status.SKIP, "Secure Boot state unknown")
    if not f.secure_boot:
        return CheckResult("gpu.secure_boot", Status.PASS, "Secure Boot disabled")
    if "nvidia" not in f.modules:
        return CheckResult(
            "gpu.secure_boot",
            Status.WARN,
            "Secure Boot enabled and nvidia module not loaded",
            hint="enroll the DKMS MOK key (mokutil --import) or disable Secure Boot",
        )
    return CheckResult("gpu.secure_boot", Status.PASS, "Secure Boot enabled; module signed")


def check_headers(f: GpuFacts) -> CheckResult:
    if f.headers_present:
        return CheckResult("gpu.kernel_headers", Status.PASS, "headers for running kernel present")
    return CheckResult(
        "gpu.kernel_headers",
        Status.WARN,
        "no headers for the running kernel",
        hint="DKMS cannot rebuild the driver on kernel upgrade; install linux-headers",
    )


def check_nvidia_smi(f: GpuFacts) -> CheckResult:
    if f.smi_rows is None:
        return CheckResult(
            "gpu.nvidia_smi",
            Status.FAIL,
            f"nvidia-smi failed: {f.smi_error}",
            hint="driver userspace missing or mismatched with the kernel module; reboot after install",
        )
    names = [r.get("name") or "?" for r in f.smi_rows]
    data = {"gpus": f.smi_rows, "cuda_driver_api": f.cuda_driver_api}
    if f.pci and len(f.smi_rows) != len(f.pci):
        return CheckResult(
            "gpu.nvidia_smi",
            Status.FAIL,
            f"nvidia-smi sees {len(f.smi_rows)} of {len(f.pci)} PCI GPUs",
            data=data,
            hint="a GPU failed to initialise; check dmesg for 'fallen off the bus' / Xid 79",
        )
    return CheckResult(
        "gpu.nvidia_smi",
        Status.PASS,
        f"{len(names)} GPU(s): {', '.join(names)} (CUDA {f.cuda_driver_api or '?'})",
        data=data,
    )


def check_blackwell(f: GpuFacts) -> CheckResult:
    rows = f.smi_rows or []
    bw = [
        r for r in rows if r.get("compute_cap") and version_tuple(r["compute_cap"] or "")[0] >= 10
    ]
    if not bw:
        return CheckResult("gpu.blackwell", Status.SKIP, "no Blackwell-class GPUs")
    kd = f.kernel_driver
    if kd is None or not kd.open_module or version_tuple(kd.version) < MIN_BLACKWELL_DRIVER:
        return CheckResult(
            "gpu.blackwell",
            Status.FAIL,
            "Blackwell GPU needs the open kernel module and driver >= 570",
            data={"driver": dataclasses.asdict(kd) if kd else None},
            hint="install an -open driver package (hwb install driver)",
        )
    return CheckResult(
        "gpu.blackwell", Status.PASS, "open kernel module, driver supports Blackwell"
    )


def check_pcie_link(f: GpuFacts) -> CheckResult:
    if not f.smi_rows:
        return CheckResult("gpu.pcie_link", Status.SKIP, "nvidia-smi unavailable")
    degraded = []
    for r in f.smi_rows:
        cur, mx = r.get("pcie.link.width.current"), r.get("pcie.link.width.max")
        if cur and mx and int(cur) < int(mx):
            degraded.append(f"GPU{r.get('index')} x{cur}/x{mx}")
    if degraded:
        return CheckResult(
            "gpu.pcie_link",
            Status.WARN,
            "PCIe link narrower than supported: " + ", ".join(degraded),
            hint="check slot wiring/bifurcation and risers; verify with hwb bench gpu (H2D bandwidth)",
        )
    return CheckResult("gpu.pcie_link", Status.PASS, "PCIe link width at max on all GPUs")


def check_persistence(f: GpuFacts) -> CheckResult:
    if not f.smi_rows:
        return CheckResult("gpu.persistence", Status.SKIP, "nvidia-smi unavailable")
    off = [r.get("index") or "?" for r in f.smi_rows if r.get("persistence_mode") == "Disabled"]
    if off:
        return CheckResult(
            "gpu.persistence",
            Status.WARN,
            f"persistence mode off on GPU {', '.join(off)}",
            hint="systemctl enable --now nvidia-persistenced",
        )
    return CheckResult("gpu.persistence", Status.PASS, "persistence mode on")


def check_libcuda(f: GpuFacts) -> CheckResult:
    if f.libcuda:
        return CheckResult("gpu.libcuda", Status.PASS, f"libcuda.so.1 at {f.libcuda}")
    return CheckResult(
        "gpu.libcuda",
        Status.FAIL,
        "libcuda.so.1 not in linker cache",
        hint="driver userspace libraries missing; reinstall the driver package",
    )


def check_cuda_toolkit(f: GpuFacts) -> CheckResult:
    if f.nvcc_version is None:
        return CheckResult(
            "gpu.cuda_toolkit",
            Status.SKIP,
            "nvcc not found (optional: PyTorch/vLLM wheels bundle the CUDA runtime)",
        )
    if f.cuda_driver_api and version_tuple(f.nvcc_version) > version_tuple(f.cuda_driver_api):
        return CheckResult(
            "gpu.cuda_toolkit",
            Status.WARN,
            f"toolkit {f.nvcc_version} is newer than driver supports ({f.cuda_driver_api})",
            hint="upgrade the driver or install a matching cuda-toolkit-X-Y",
        )
    return CheckResult("gpu.cuda_toolkit", Status.PASS, f"CUDA toolkit {f.nvcc_version}")


def check_xid(f: GpuFacts) -> CheckResult:
    if f.xid_lines is None:
        return CheckResult("gpu.xid", Status.SKIP, "kernel log unreadable (run as root)")
    if f.xid_lines:
        return CheckResult(
            "gpu.xid",
            Status.WARN,
            f"{len(f.xid_lines)} NVRM Xid event(s) this boot",
            data={"sample": f.xid_lines[:_XID_SAMPLE]},
            hint="look up the Xid codes; 79/48/63/64 usually mean hardware, power or thermal trouble",
        )
    return CheckResult("gpu.xid", Status.PASS, "no Xid events this boot")


CHECKS: tuple[Callable[[GpuFacts], CheckResult], ...] = (
    check_pci,
    check_nouveau,
    check_kernel_module,
    check_bound,
    check_secure_boot,
    check_headers,
    check_nvidia_smi,
    check_blackwell,
    check_pcie_link,
    check_persistence,
    check_libcuda,
    check_cuda_toolkit,
    check_xid,
)


def run_detect(ctx: RunContext) -> None:
    facts = gather()
    section = dataclasses.asdict(facts)
    section["modules"] = sorted(m for m in facts.modules if m.startswith(("nvidia", "nouveau")))
    ctx.sections["gpu_facts"] = section
    for chk in CHECKS:
        ctx.check(f"gpu.{chk.__name__.removeprefix('check_')}", functools.partial(chk, facts))
