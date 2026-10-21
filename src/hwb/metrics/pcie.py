"""PCIe link health of each GPU over time: negotiated speed/width and error counters.

A link that retrains to a lower generation under load, or racks up correctable errors, is
the classic sign of a marginal channel (riser, retimer, slot, contacts), often one that only
shows once the chassis is warm. Two sources, both cheap to poll:

- sysfs (kernel view, needs no nvidia-smi): ``current_link_speed``/``current_link_width``
  and the AER counters ``aer_dev_correctable``/``_nonfatal``/``_fatal`` of the GPU and of the
  upstream port it hangs off;
- the replay counters in ``nvidia-smi -q`` (there is no PCIe-only ``-d`` section), snapshotted
  before and after.

Idle GPUs lower their link generation to save power; that is normal and judged elsewhere
(only while the GPU is busy).

What caps a link below the card's generation is read from the upstream port: its Link
Capabilities (``max_link_speed``: the slot itself is older) and, as root, its Target Link Speed
(Link Control 2 in config space). A lowered target is the BIOS's doing (often set for a riser)
only on a clean link: a marginal link that took errors falls back too, and then the target can
read low as well, so the correctable error count since boot decides which it is.

For debugging a link (``debug_snapshot``, ``kernel_events``): both ends' link registers (as
root), the AER counters by error type, nvidia-smi's PCI view, the kernel's PCIe/AER/Xid lines
and the kernel's PCIe link-speed cooling devices. Never the Device Serial Number capability.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import nvsmi
from ..core import proc
from ..core.errors import HwbError

log = logging.getLogger(__name__)

# GT/s per lane -> PCIe generation
_GTS_TO_GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
_PCI_ADDR = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_AER_KEYS = {
    "aer_dev_correctable": "TOTAL_ERR_COR",
    "aer_dev_nonfatal": "TOTAL_ERR_NONFATAL",
    "aer_dev_fatal": "TOTAL_ERR_FATAL",
}
_GPU_BLOCK = re.compile(r"^GPU ([0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])\s*$", re.M)
_REPLAYS = re.compile(r"Replays Since Reset\s*:\s*(\d+)")
_ROLLOVERS = re.compile(r"Replay Number Rollovers\s*:\s*(\d+)")
_REPLAY_ROLLOVER = 1 << 8  # the replay counter wraps at 256 (NVML)
_CAP_LIST = 0x10  # Status register bit: the device has a capabilities list
_CAP_PCIE = 0x10  # capability id of the PCI Express capability
_LNKCAP, _LNKSTA, _LNKCTL2, _LNKSTA2 = 0x0C, 0x12, 0x30, 0x32  # inside the PCIe capability
_ADDR_IN_TEXT = re.compile(r"\b([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\b")
_KMSG = re.compile(r"^\[\s*([\d.]+)\]\s+\S+\s+kernel:\s?(.*)$")  # short-monotonic; drops the host
# the error bits the kernel's AER driver names in "[ 0] RxErr  (First)" lines
_AER_NAMES = (
    "RxErr|BadTLP|BadDLLP|Rollover|Timeout|NonFatalErr|CorrIntErr|HeaderOF|Undefined|DLP|SDES|TLP"
    "|FCP|CmpltTO|CmpltAbrt|UnxCmplt|RxOF|MalfTLP|ECRC|UnsupReq|ACSViol|UncorrIntErr|BlockedTLP"
    "|AtomicOpBlocked|TLPBlockedErr|PoisonTLPBlocked|DMWrReqBlocked|IDECheck|MisIDETLP|PCRC_CHECK"
)
_ERR_TYPE = re.compile(rf"\[\s*\d+\]\s+({_AER_NAMES})\b")
_CORRECTABLE = frozenset(
    {"RxErr", "BadTLP", "BadDLLP", "Rollover", "Timeout", "NonFatalErr", "CorrIntErr", "HeaderOF"}
)
_PCIE_EVENT = re.compile(
    rf"AER|PCIe Bus Error|error status/mask|{_ERR_TYPE.pattern}|bwctrl"
    r"|NVRM: (?:Xid|GPU at PCI|.*fallen off)"
    # link state lines count only when they name a PCI device (not "ata3: SATA link down")
    r"|\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b.*(?:[Ll]ink (?:[Ss]peed|down|up)|limited by"
    r"|\bDPC\b|retrain)"
)
_OSC_AER = re.compile(r"_OSC: (OS now controls|platform does not support) \[([^\]]*)\]")
_GPU_UUID = re.compile(r"\bGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
_KERNEL_KEEP = (20, 80)  # first and last lines kept in the report


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def normalize_bus_id(bus_id: str) -> str:
    """nvidia-smi "00000000:41:00.0" -> sysfs "0000:41:00.0"."""
    dom, _, rest = bus_id.strip().lower().partition(":")
    return f"{dom[-4:].zfill(4)}:{rest}"


def link_gen(speed: str | None) -> int | None:
    """ "16.0 GT/s PCIe" -> 4."""
    m = re.match(r"\s*([\d.]+)\s*GT/s", speed or "")
    return _GTS_TO_GEN.get(float(m.group(1))) if m else None


def link_width(text: str | None) -> int | None:
    return int(text) if text and text.strip().isdigit() and int(text) > 0 else None


def aer_total(root: Path, addr: str, kind: str) -> int | None:
    text = _read(root / "sys/bus/pci/devices" / addr / kind)
    if text is None:
        return None
    for line in text.splitlines():
        key, _, value = line.strip().rpartition(" ")
        if key.strip() == _AER_KEYS[kind] and value.isdigit():
            return int(value)
    return None


def upstream_port(root: Path, addr: str) -> str | None:
    """The bridge/root port the GPU hangs off (its parent in the sysfs device tree)."""
    try:
        parent = (root / "sys/bus/pci/devices" / addr).resolve().parent.name
    except OSError:
        return None
    return parent if _PCI_ADDR.match(parent) else None


def parse_replays(text: str) -> dict[str, int]:
    """Total replays per GPU (by sysfs address) from the full ``nvidia-smi -q`` report."""
    out: dict[str, int] = {}
    heads = list(_GPU_BLOCK.finditer(text))
    for i, m in enumerate(heads):
        block = text[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        replays, rollovers = _REPLAYS.search(block), _ROLLOVERS.search(block)
        if replays:
            out[normalize_bus_id(m.group(1))] = int(replays.group(1)) + (
                int(rollovers.group(1)) * _REPLAY_ROLLOVER if rollovers else 0
            )
    return out


def _pcie_cap(config: bytes) -> int | None:
    """Offset of the PCI Express capability in a config space dump (None: unprivileged read)."""
    if len(config) <= 0x40 or not config[0x06] & _CAP_LIST:
        return None
    ptr, seen = config[0x34] & 0xFC, set()
    while ptr >= 0x40 and ptr not in seen and ptr + _LNKSTA2 + 2 <= len(config):
        seen.add(ptr)
        if config[ptr] == _CAP_PCIE:
            return ptr
        ptr = config[ptr + 1] & 0xFC
    return None


def _reg(config: bytes, off: int, size: int = 2) -> int:
    return int.from_bytes(config[off : off + size], "little")


def target_gen(config: bytes) -> int | None:
    """Target Link Speed (Link Control 2) from a port's config space: the highest generation
    the link may train to. Unprivileged reads of ``config`` stop at 64 bytes, so this needs root."""
    cap = _pcie_cap(config)
    if cap is None:
        return None
    return _reg(config, cap + _LNKCTL2) & 0xF or None  # the encoding is the generation


def link_regs(config: bytes) -> dict[str, Any] | None:
    """Link Capabilities/Status/Control 2/Status 2 of one end of a link (speeds as generations)."""
    cap = _pcie_cap(config)
    if cap is None:
        return None
    lnkcap, sta = _reg(config, cap + _LNKCAP, 4), _reg(config, cap + _LNKSTA)
    sta2 = _reg(config, cap + _LNKSTA2)
    return {
        "cap_gen": lnkcap & 0xF or None,
        "cap_width": (lnkcap >> 4) & 0x3F or None,
        "gen": sta & 0xF or None,
        "width": (sta >> 4) & 0x3F or None,
        "target_gen": _reg(config, cap + _LNKCTL2) & 0xF or None,
        "training": bool(sta & 1 << 11),
        "dl_active": bool(sta & 1 << 13),
        # speed/width changed by a retrain: software asked, or the link was unreliable
        "bw_mgmt_status": bool(sta & 1 << 14),
        "autonomous_bw_status": bool(sta & 1 << 15),
        "eq_complete": bool(sta2 & 1 << 1),
        "eq_request": bool(sta2 & 1 << 5),
    }


def aer_counts(root: Path, addr: str) -> dict[str, dict[str, int]] | None:
    """AER counters since boot by error type ("RxErr", "BadTLP", ...), non-zero ones and totals."""
    out: dict[str, dict[str, int]] = {}
    for kind, total_key in _AER_KEYS.items():
        text = _read(root / "sys/bus/pci/devices" / addr / kind)
        if text is None:
            continue
        counts: dict[str, int] = {}
        for line in text.splitlines():
            key, _, value = line.strip().rpartition(" ")
            if value.isdigit() and (int(value) or key.strip() == total_key):
                counts["total" if key.strip() == total_key else key.strip()] = int(value)
        out[kind.removeprefix("aer_dev_")] = counts
    return out or None


def _end(root: Path, addr: str) -> dict[str, Any]:
    dev = root / "sys/bus/pci/devices" / addr
    config = b""
    with contextlib.suppress(OSError):
        config = (dev / "config").read_bytes()
    return {
        "addr": addr,
        "max_gen": link_gen(_read(dev / "max_link_speed")),
        "max_width": link_width(_read(dev / "max_link_width")),
        "gen": link_gen(_read(dev / "current_link_speed")),
        "width": link_width(_read(dev / "current_link_width")),
        "regs": link_regs(config),
        "aer": aer_counts(root, addr),
    }


def debug_snapshot(root: Path, addrs: list[str]) -> list[dict[str, Any]]:
    """Both ends of each GPU's link: sysfs state, link registers (root), AER by type."""
    out = []
    for addr in addrs:
        port = upstream_port(root, addr)
        out.append({"gpu": _end(root, addr), "port": _end(root, port) if port else None})
    return out


def cooling_devices(root: Path) -> list[dict[str, Any]]:
    """The kernel's PCIe link-speed cooling devices (bwctrl) and the thermal zones bound to them.
    cur_state only mirrors the link's current speed (an idle GPU sits at Gen1, the top state):
    a device can slow a link only when a thermal zone drives it."""
    thermal = root / "sys/class/thermal"
    zones: dict[str, list[str]] = {}
    for z in sorted(thermal.glob("thermal_zone*")):
        kind = _read(z / "type") or z.name
        for link in z.glob("cdev[0-9]*"):
            if link.name.isalnum():  # cdevN, not cdevN_trip_point / cdevN_weight
                zones.setdefault(link.resolve().name, []).append(kind)
    out = []
    for d in sorted(thermal.glob("cooling_device*")):
        kind = _read(d / "type") or ""
        if kind.startswith("PCIe_Port_Link_Speed"):
            cur, mx = _read(d / "cur_state"), _read(d / "max_state")
            out.append(
                {
                    "type": kind,
                    "cur_state": int(cur) if cur and cur.isdigit() else None,
                    "max_state": int(mx) if mx and mx.isdigit() else None,
                    "zones": zones.get(d.name, []),
                }
            )
    return out


def parse_kernel_events(text: str) -> dict[str, Any]:
    """PCIe/AER/Xid lines of a short-monotonic kernel log: counts by error type and device, and
    the first and last lines with their time since boot."""
    events: list[tuple[float, str]] = []
    aer_native: bool | None = None  # does the OS (not the firmware) handle AER, so it sees errors?
    for line in text.splitlines():
        m = _KMSG.match(line)
        osc = _OSC_AER.search(m.group(2)) if m else None
        if osc and "AER" in osc.group(2).split():
            native = osc.group(1) == "OS now controls"
            aer_native = native if aer_native is None else aer_native and native
            continue
        if m and _PCIE_EVENT.search(m.group(2)):
            # "NVRM: GPU at PCI:...: GPU-<uuid>" names the unit: never into a report
            events.append((float(m.group(1)), _GPU_UUID.sub("GPU-[uuid]", m.group(2).strip())))
    types = Counter(m.group(1) for _, msg in events if (m := _ERR_TYPE.search(msg)))
    devices = Counter(a.group(1) for _, msg in events if (a := _ADDR_IN_TEXT.search(msg)))
    head, tail = _KERNEL_KEEP
    keep = events if len(events) <= head + tail else events[:head] + events[-tail:]
    # some kernels log only the "[ 0] RxErr" bit lines, without the "severity=" header line
    bits_corr = sum(n for t, n in types.items() if t in _CORRECTABLE)
    bits_uncorr = sum(n for t, n in types.items() if t not in _CORRECTABLE and t != "Undefined")
    return {
        "aer_native": aer_native,
        "lines": len(events),
        "corrected": max(sum("severity=Correct" in msg for _, msg in events), bits_corr),
        "uncorrected": max(sum("severity=Uncorrect" in msg for _, msg in events), bits_uncorr),
        "error_types": dict(types.most_common()),
        "devices": dict(devices.most_common()),
        "first_s": events[0][0] if events else None,
        "last_s": events[-1][0] if events else None,
        "sample": [[round(t, 1), msg] for t, msg in keep],
    }


def kernel_events(since: float | None = None) -> dict[str, Any] | None:
    """parse_kernel_events over this boot's kernel log (or since a wall-clock time); None if
    the log is unreadable (journald without permission, no journalctl)."""
    if not proc.have("journalctl"):
        return None
    argv = ["journalctl", "-k", "--no-pager", "-o", "short-monotonic"]
    argv += [f"--since=@{since:.0f}"] if since is not None else ["-b"]
    res = proc.run(argv, timeout=60)
    return parse_kernel_events(res.stdout) if res.ok else None


_SMI_FIELDS = {
    ("PCIe Generation", "Max"): "gen_max",
    ("PCIe Generation", "Current"): "gen_current",
    ("PCIe Generation", "Device Current"): "gen_device_current",
    ("PCIe Generation", "Device Max"): "gen_device_max",
    ("PCIe Generation", "Host Max"): "gen_host_max",
    ("Link Width", "Max"): "width_max",
    ("Link Width", "Current"): "width_current",
    (None, "Replays Since Reset"): "replays",
    (None, "Replay Number Rollovers"): "replay_rollovers",
    (None, "Tx Throughput"): "tx_kb_s",
    (None, "Rx Throughput"): "rx_kb_s",
}


def parse_smi_pci(text: str) -> dict[str, dict[str, int]]:
    """The PCI link fields of ``nvidia-smi -q`` per GPU (by sysfs address). Only these numbers
    are kept: the full output also carries serial numbers and UUIDs."""
    out: dict[str, dict[str, int]] = {}
    heads = list(_GPU_BLOCK.finditer(text))
    for i, m in enumerate(heads):
        block = text[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        fields: dict[str, int] = {}
        sub: str | None = None
        for line in block.splitlines():
            key, sep, value = line.strip().partition(":")
            key = key.strip()
            if not sep:
                sub = key if key in ("PCIe Generation", "Link Width") else None
                continue
            name = _SMI_FIELDS.get((sub, key)) or _SMI_FIELDS.get((None, key))
            num = re.match(r"\s*(\d+)", value)
            if name and num and name not in fields:
                fields[name] = int(num.group(1))
            if key in ("Replays Since Reset",):
                sub = None
        out[normalize_bus_id(m.group(1))] = fields
    return out


@dataclass(frozen=True)
class LinkLimit:
    """How fast a GPU's link may run: the card, its slot's port, and the port's setting."""

    addr: str
    gpu_gen: int | None  # the card's own maximum
    port_gen: int | None  # the upstream port's maximum (the slot)
    target_gen: int | None  # what the port is set to train to (root only)
    errors: int | None = None  # correctable errors since boot, card + port (AER)

    @property
    def slot_gen(self) -> int | None:
        """The highest generation the slot hardware allows the card (min of both ends)."""
        gens = [g for g in (self.gpu_gen, self.port_gen) if g]
        return min(gens) if gens else None

    @property
    def target_capped(self) -> bool:
        """Slot and card both do more than the port is set to train to."""
        return bool(self.slot_gen and self.target_gen and self.target_gen < self.slot_gen)

    @property
    def bios_capped(self) -> bool:
        """A lowered target on a clean link: set that way (BIOS), not a fallback after errors."""
        return self.target_capped and not self.errors


def link_limit(root: Path, addr: str) -> LinkLimit:
    dev = root / "sys/bus/pci/devices"
    port = upstream_port(root, addr)
    config = b""
    if port:
        with contextlib.suppress(OSError):
            config = (dev / port / "config").read_bytes()
    counts = [
        n
        for a in (addr, port)
        if a and (n := aer_total(root, a, "aer_dev_correctable")) is not None
    ]
    return LinkLimit(
        addr,
        link_gen(_read(dev / addr / "max_link_speed")),
        link_gen(_read(dev / port / "max_link_speed")) if port else None,
        target_gen(config),
        sum(counts) if counts else None,
    )


def _for_link(kernel: dict[str, Any] | None, *addrs: str | None) -> list[list[Any]] | None:
    """The kernel lines that name this GPU or its port (None: log unreadable)."""
    if kernel is None:
        return None
    names = [a for a in addrs if a]
    return [line for line in kernel["sample"] if any(a in line[1] for a in names)]


def gpu_addresses() -> dict[str, str]:
    """nvidia-smi GPU index -> sysfs PCI address, for the GPUs the host driver owns."""
    rows = nvsmi.query_gpus(("index", "pci.bus_id"))
    return {
        str(r["index"]): normalize_bus_id(str(r["pci.bus_id"])) for r in rows if r.get("pci.bus_id")
    }


def replay_snapshot() -> dict[str, int]:
    if not proc.have("nvidia-smi"):
        return {}
    res = proc.run(["nvidia-smi", "-q"], timeout=120)
    return parse_replays(res.stdout) if res.ok else {}


def counter_delta(
    base: dict[str, int | None], now: dict[str, int | None], kinds: tuple[str, ...]
) -> int | None:
    """Sum of (now - base) over the "<where>:<kind>" counters of the given kinds."""
    vals = [
        n - b
        for k, b in base.items()
        if k.split(":", 1)[1] in kinds and b is not None and (n := now.get(k)) is not None
    ]
    return sum(vals) if vals else None


@dataclass
class _Gpu:
    index: str
    addr: str
    upstream: str | None
    max_gen: int | None
    max_width: int | None
    limit: LinkLimit
    base: dict[str, int | None] = field(default_factory=dict)  # error counters at start
    worst_width: int | None = None


class PcieWatch:
    """Baseline at start, per-tick link state, deltas at the end."""

    def __init__(self, root: Path, gpus: dict[str, str]) -> None:
        self.root = root
        dev = root / "sys/bus/pci/devices"
        self.gpus = []
        for index, addr in sorted(gpus.items()):
            limit = link_limit(root, addr)
            self.gpus.append(
                _Gpu(
                    index,
                    addr,
                    upstream_port(root, addr),
                    limit.gpu_gen,
                    link_width(_read(dev / addr / "max_link_width")),
                    limit,
                )
            )
        self._replays0: dict[str, int] = {}
        self._started: float | None = None

    def _counters(self, g: _Gpu) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for kind in _AER_KEYS:
            for where, addr in (("gpu", g.addr), ("port", g.upstream)):
                out[f"{where}:{kind}"] = aer_total(self.root, addr, kind) if addr else None
        return out

    def start(self) -> None:
        self._started = time.time()
        for g in self.gpus:
            g.base = self._counters(g)
        try:
            self._replays0 = replay_snapshot()
        except HwbError as exc:
            log.debug("no replay counters: %s", exc)

    def sample(self) -> dict[str, dict[str, int | None]]:
        dev = self.root / "sys/bus/pci/devices"
        out = {}
        for g in self.gpus:
            width = link_width(_read(dev / g.addr / "current_link_width"))
            if width is not None and (g.worst_width is None or width < g.worst_width):
                g.worst_width = width
            out[g.index] = {
                "gen": link_gen(_read(dev / g.addr / "current_link_speed")),
                "width": width,
                "errors": counter_delta(g.base, self._counters(g), ("aer_dev_correctable",)),
            }
        return out

    def finish(self) -> dict[str, dict[str, Any]]:
        try:
            replays1 = replay_snapshot()
        except HwbError:
            replays1 = {}
        out: dict[str, dict[str, Any]] = {}
        try:
            # what the kernel said about PCIe links while this ran (AER, retrains, Xid)
            kernel = kernel_events(since=self._started) if self._started else None
        except HwbError as exc:
            log.debug("no kernel log: %s", exc)
            kernel = None
        for g in self.gpus:
            now = self._counters(g)
            r0, r1 = self._replays0.get(g.addr), replays1.get(g.addr)
            out[g.index] = {
                "addr": g.addr,
                "upstream": g.upstream,
                "max_gen": g.max_gen,
                "port_max_gen": g.limit.port_gen,
                "port_target_gen": g.limit.target_gen,
                "errors_since_boot": g.limit.errors,
                "max_width": g.max_width,
                "min_width": g.worst_width,
                "correctable_errors": counter_delta(g.base, now, ("aer_dev_correctable",)),
                "uncorrectable_errors": counter_delta(
                    g.base, now, ("aer_dev_nonfatal", "aer_dev_fatal")
                ),
                "replays": r1 - r0 if r0 is not None and r1 is not None else None,
                "kernel_events": _for_link(kernel, g.addr, g.upstream),
            }
        return out
