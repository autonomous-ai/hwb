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
from typing import Any

from .. import nvsmi
from ..core import proc
from ..core.context import RunContext
from ..core.errors import HwbError
from ..core.log import fields
from ..core.platform import os_info
from ..core.result import CheckResult, Status
from ..metrics import pcie
from ..metrics.pcie import LinkLimit, link_limit, normalize_bus_id

log = logging.getLogger(__name__)

NVIDIA_VENDOR = "0x10de"
# Drivers that hand a GPU to a VM (PCI passthrough). Such GPUs are accepted: the host's
# nvidia driver doesn't own them, and their driver stack lives inside the guest.
PASSTHROUGH_DRIVERS = frozenset({"vfio-pci", "pci-stub"})
# Blackwell (sm_100/sm_120, e.g. RTX 5090, RTX PRO 6000 Blackwell) requires the open kernel modules and >= R570.
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
    pcie_links: list[LinkLimit] = field(default_factory=list)
    # PCIe debug: both ends of each GPU link, nvidia-smi's PCI view, kernel PCIe/AER lines
    pcie_debug: list[dict[str, Any]] = field(default_factory=list)
    smi_pci: dict[str, dict[str, int]] | None = None
    kernel_pcie: dict[str, Any] | None = None
    pcie_cooling: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passthrough(self) -> list[PciGpu]:
        return [g for g in self.pci if g.driver in PASSTHROUGH_DRIVERS]

    @property
    def host_gpus(self) -> list[PciGpu]:
        """GPUs the host driver should own: everything not passed through to a VM."""
        return [g for g in self.pci if g.driver not in PASSTHROUGH_DRIVERS]

    @property
    def all_passed_through(self) -> bool:
        return bool(self.pci) and not self.host_gpus


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


def smi_pci() -> dict[str, dict[str, int]] | None:
    if not proc.have("nvidia-smi"):
        return None
    res = proc.run(["nvidia-smi", "-q"], timeout=120)
    return pcie.parse_smi_pci(res.stdout) if res.ok else None


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
        (
            "pcie_debug",
            lambda: setattr(
                facts,
                "pcie_debug",
                pcie.debug_snapshot(Path("/"), [g.address for g in facts.host_gpus]),
            ),
        ),
        ("kernel_pcie", lambda: setattr(facts, "kernel_pcie", pcie.kernel_events())),
        ("pcie_cooling", lambda: setattr(facts, "pcie_cooling", pcie.cooling_devices(Path("/")))),
        ("smi_pci", lambda: setattr(facts, "smi_pci", smi_pci())),
        (
            "pcie_links",
            lambda: setattr(
                facts, "pcie_links", [link_limit(Path("/"), g.address) for g in facts.host_gpus]
            ),
        ),
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

Check = Callable[[GpuFacts], CheckResult]


def _passthrough_note(f: GpuFacts) -> str:
    drivers = ", ".join(sorted({g.driver or "?" for g in f.passthrough}))
    return f"{len(f.passthrough)} passed through to VMs ({drivers})"


def needs_host_driver(chk: Check) -> Check:
    """SKIP a host driver check when every GPU is passed through to VMs."""

    @functools.wraps(chk)
    def wrapper(f: GpuFacts) -> CheckResult:
        if f.all_passed_through:
            return CheckResult(
                f"gpu.{chk.__name__.removeprefix('check_')}",
                Status.SKIP,
                f"all {len(f.pci)} GPUs are passed through to VMs; the driver stack lives in the guests",
            )
        return chk(f)

    return wrapper


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
        f"{len(f.pci)} NVIDIA GPU(s) on PCI bus"
        + (f", {_passthrough_note(f)}" if f.passthrough else ""),
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


@needs_host_driver
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
    drivers = {g.address: g.driver for g in f.pci}
    unbound = [g.address for g in f.host_gpus if g.driver != "nvidia"]
    if unbound:
        return CheckResult(
            "gpu.bound",
            Status.FAIL,
            f"{len(unbound)} GPU(s) bound to neither nvidia nor a VM: {', '.join(unbound)}",
            data={"drivers": drivers},
            hint="check dmesg for NVRM errors (unsupported GPU for this driver branch?)",
        )
    if f.passthrough:
        return CheckResult(
            "gpu.bound",
            Status.PASS,
            f"{len(f.host_gpus)} bound to nvidia, {_passthrough_note(f)}",
            data={"drivers": drivers},
        )
    return CheckResult("gpu.bound", Status.PASS, "all GPUs bound to nvidia")


@needs_host_driver
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


@needs_host_driver
def check_headers(f: GpuFacts) -> CheckResult:
    if f.headers_present:
        return CheckResult("gpu.kernel_headers", Status.PASS, "headers for running kernel present")
    return CheckResult(
        "gpu.kernel_headers",
        Status.WARN,
        "no headers for the running kernel",
        hint="DKMS cannot rebuild the driver on kernel upgrade; install linux-headers",
    )


@needs_host_driver
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
    host = f.host_gpus
    if f.pci and len(f.smi_rows) != len(host):
        scope = f"host GPUs ({_passthrough_note(f)})" if f.passthrough else "PCI GPUs"
        return CheckResult(
            "gpu.nvidia_smi",
            Status.FAIL,
            f"nvidia-smi sees {len(f.smi_rows)} of {len(host)} {scope}",
            data=data,
            hint="a GPU failed to initialise; check dmesg for 'fallen off the bus' / Xid 79",
        )
    return CheckResult(
        "gpu.nvidia_smi",
        Status.PASS,
        f"{len(names)} GPU(s): {', '.join(names)} (CUDA {f.cuda_driver_api or '?'})",
        data=data,
    )


@needs_host_driver
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


def _int(v: object) -> int | None:
    return int(str(v)) if v is not None and str(v).strip().isdigit() else None


@needs_host_driver
def check_pcie_link(f: GpuFacts) -> CheckResult:
    """Width below the card's (wiring, bifurcation, riser), generation capped below what card and
    slot can do, and correctable errors since boot. A capped generation on a link with errors is
    a fallback (marginal link), on a clean one a setting (BIOS). The current generation drops at
    idle on purpose, so only the maximum counts here; a link that trains low under load is judged
    by the benchmarks and serve."""
    if not f.smi_rows:
        return CheckResult("gpu.pcie_link", Status.SKIP, "nvidia-smi unavailable")
    limits = {lim.addr: lim for lim in f.pcie_links}
    # firmware-first AER: the OS never sees corrected errors, so "no errors" proves nothing
    aer_blind = bool(f.kernel_pcie) and f.kernel_pcie.get("aer_native") is False  # type: ignore[union-attr]
    narrow, capped, errors, notes = [], [], [], []
    bios = degraded = trained = False
    for r in f.smi_rows:
        # the index follows the slot (bus order); the serial names the card, so swaps can be traced
        gpu = f"GPU{r.get('index')}" + (f" (SN {r['gpu_serial']})" if r.get("gpu_serial") else "")
        cur, mx = _int(r.get("pcie.link.width.current")), _int(r.get("pcie.link.width.max"))
        if cur and mx and cur < mx:
            narrow.append(f"{gpu} x{cur}/x{mx}")
        lim = limits.get(normalize_bus_id(str(r.get("pci.bus_id") or "")))
        smi_gen = _int(r.get("pcie.link.gen.max"))  # what nvidia-smi says the system allows now
        if lim is None or not lim.gpu_gen:
            continue
        if lim.errors:
            errors.append(f"{gpu} {lim.errors}")
        slot = lim.slot_gen or lim.gpu_gen
        low = lim.target_gen if lim.target_capped else smi_gen
        if lim.port_gen and lim.port_gen < lim.gpu_gen:
            notes.append(
                f"{gpu} Gen{lim.port_gen} card Gen{lim.gpu_gen}: the slot is Gen{lim.port_gen}"
            )
        elif lim.bios_capped and aer_blind:
            capped.append(
                f"{gpu} Gen{lim.target_gen} of Gen{slot}: a BIOS setting or a fallback after "
                "errors (the firmware handles AER, so hwb cannot see errors)"
            )
        elif lim.bios_capped:
            bios = True
            capped.append(f"{gpu} Gen{lim.target_gen} of Gen{slot}: set in the BIOS")
        elif low and low < slot and lim.errors:
            degraded = True
            capped.append(f"{gpu} Gen{low} of Gen{slot}: fell back after link errors")
        elif low and low < slot and lim.target_gen:
            # the port is set to the full speed (not the BIOS), yet the link came up slower
            trained = True
            capped.append(
                f"{gpu} Gen{low} of Gen{slot}: the slot is set to Gen{lim.target_gen} "
                "but the link trained lower"
                + (" (the firmware handles AER, so hwb cannot see errors)" if aer_blind else "")
            )
        elif low and low < slot:
            capped.append(f"{gpu} Gen{low} of Gen{slot}")
    note = f" ({'; '.join(notes)})" if notes else ""
    data = {"links": [dataclasses.asdict(lim) for lim in f.pcie_links]}
    if not (narrow or capped or errors):
        return CheckResult(
            "gpu.pcie_link", Status.PASS,
            "PCIe link width and generation at max on all GPUs" + note, data=data,
        )  # fmt: skip
    parts, hints = [], []
    if narrow:
        parts.append("PCIe link narrower than supported: " + ", ".join(narrow))
        hints.append("check slot wiring/bifurcation and risers")
    if capped:
        parts.append("PCIe generation limited: " + ", ".join(capped))
    if errors:
        parts.append("correctable PCIe errors since boot: " + ", ".join(errors))
    if degraded or errors:
        hints.append(
            "the link is marginal: errors make it fall back a generation, often once it warms up "
            "(hot air around the slot or CPU, a riser, seating). Check airflow and the CPU cooler, "
            "reseat card and riser; a cold power-off restores the generation only until it warms "
            "up again. Record when it drops: hwb monitor"
        )
    if bios:
        hints.append(
            "set the slot's PCIe link speed to Auto in the BIOS (it was likely lowered for a "
            "riser), then confirm the link holds it under load without errors (hwb serve)"
        )
    elif trained and not degraded:
        hints.append(
            "the link could not hold the full speed when it trained (at power-on or after a "
            "retrain): signal quality on the way to the card, not a setting. Usual suspects: a "
            "riser or extension cable (length, bends, heat from GPU exhaust), seating, the slot. "
            "Swap risers or slots between cards to see whether the fault follows the part; a "
            "riser that cannot run Gen5 reliably can be locked to Gen4 in the BIOS"
        )
    elif capped and not degraded:
        hints.append(
            "check the slot's PCIe link speed in the BIOS (Auto) and any riser; run hwb as "
            "root to read the slot's setting and error counters and name the cause"
        )
    return CheckResult(
        "gpu.pcie_link", Status.WARN, "; ".join(parts) + note, data=data, hint="; ".join(hints)
    )


def check_pcie_kernel(f: GpuFacts) -> CheckResult:
    """PCIe trouble the kernel logged this boot (AER errors, link retrains) and kernel cooling
    devices that slowed a PCIe link. Every device counts: a sick NVMe link matters too."""
    k = f.kernel_pcie
    # cur_state just mirrors the link's current speed; only a bound thermal zone can slow it
    cooling = [c for c in f.pcie_cooling if c.get("zones")]
    if k is None and not cooling:
        return CheckResult("gpu.pcie_kernel", Status.SKIP, "kernel log unreadable (run as root)")
    k = k or {}
    gpus = {d["gpu"]["addr"]: f"GPU link {d['gpu']['addr']}" for d in f.pcie_debug}
    gpus |= {d["port"]["addr"]: f"GPU port {d['port']['addr']}" for d in f.pcie_debug if d["port"]}
    data = {"kernel": k, "cooling": f.pcie_cooling}
    parts = []
    errors = k.get("corrected", 0) + k.get("uncorrected", 0)
    if errors:
        kinds = ", ".join(f"{t} {n}" for t, n in list(k["error_types"].items())[:4])
        where = ", ".join(gpus.get(a, a) for a in list(k["devices"])[:3])
        parts.append(
            f"kernel logged {errors} PCIe error messages this boot"
            + (f" ({kinds})" if kinds else "") + (f" on {where}" if where else "")
            + f", first {k['first_s']:.0f} s after boot, last {k['last_s']:.0f} s"
        )  # fmt: skip
    for c in cooling:
        port = c["type"].removeprefix("PCIe_Port_Link_Speed_")
        parts.append(
            f"thermal zone {', '.join(c['zones'])} can slow the PCIe link at "
            f"{gpus.get(port, port)} (state {c['cur_state']} of {c['max_state']})"
        )
    if not parts:
        if k.get("aer_native") is False:
            return CheckResult(
                "gpu.pcie_kernel", Status.PASS,
                "no PCIe errors in the kernel log, but the firmware handles AER (_OSC), so the OS "
                "does not see corrected errors: zero error counts here prove nothing", data=data,
            )  # fmt: skip
        return CheckResult(
            "gpu.pcie_kernel", Status.PASS, "no PCIe errors in the kernel log this boot", data=data
        )
    return CheckResult(
        "gpu.pcie_kernel",
        Status.FAIL if k.get("uncorrected") else Status.WARN,
        "; ".join(parts),
        data=data,
        hint="errors on a GPU link make it fall back a generation: check airflow around the slot "
        "and CPU, reseat card and riser, try another riser; the times since boot show when it "
        "started (compare with hwb monitor)" if errors else
        "a thermal zone lowers this PCIe link's speed when it gets hot: check that zone's "
        "temperature and trip points in /sys/class/thermal",
    )  # fmt: skip


@needs_host_driver
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


@needs_host_driver
def check_libcuda(f: GpuFacts) -> CheckResult:
    if f.libcuda:
        return CheckResult("gpu.libcuda", Status.PASS, f"libcuda.so.1 at {f.libcuda}")
    return CheckResult(
        "gpu.libcuda",
        Status.FAIL,
        "libcuda.so.1 not in linker cache",
        hint="driver userspace libraries missing; reinstall the driver package",
    )


@needs_host_driver
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


CHECKS: tuple[Check, ...] = (
    check_pci,
    check_nouveau,
    check_kernel_module,
    check_bound,
    check_secure_boot,
    check_headers,
    check_nvidia_smi,
    check_blackwell,
    check_pcie_link,
    check_pcie_kernel,
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
