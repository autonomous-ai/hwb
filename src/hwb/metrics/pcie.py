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
"""

from __future__ import annotations

import contextlib
import logging
import re
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
_LNKCTL2 = 0x30  # Link Control 2, inside the PCI Express capability


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


def target_gen(config: bytes) -> int | None:
    """Target Link Speed (Link Control 2) from a port's config space: the highest generation
    the link may train to. Unprivileged reads of ``config`` stop at 64 bytes, so this needs root."""
    if len(config) <= 0x40 or not config[0x06] & _CAP_LIST:
        return None
    ptr, seen = config[0x34] & 0xFC, set()
    while ptr >= 0x40 and ptr not in seen and ptr + _LNKCTL2 + 2 <= len(config):
        seen.add(ptr)
        if config[ptr] == _CAP_PCIE:
            speed = int.from_bytes(config[ptr + _LNKCTL2 : ptr + _LNKCTL2 + 2], "little") & 0xF
            return speed or None  # the encoding is the generation: 1 = 2.5 GT/s ... 5 = 32 GT/s
        ptr = config[ptr + 1] & 0xFC
    return None


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

    def _counters(self, g: _Gpu) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for kind in _AER_KEYS:
            for where, addr in (("gpu", g.addr), ("port", g.upstream)):
                out[f"{where}:{kind}"] = aer_total(self.root, addr, kind) if addr else None
        return out

    def start(self) -> None:
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
            }
        return out
