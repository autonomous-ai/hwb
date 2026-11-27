"""AMD GPUs (amdgpu driver) from sysfs: inventory, the PCIe link that is the slot, and samples.

Everything here is a file read, so it needs no ROCm tools on the host (hwb serves AMD GPUs from
containers). Per GPU the driver publishes ``gpu_busy_percent``, ``mem_info_vram_*`` and a hwmon
directory with ``temp1`` (edge), ``temp2`` (junction), ``temp3`` (mem), ``power1_average`` (or
``power1_input``, in microwatts) and ``freq1`` (the shader clock, in Hz).

A discrete Radeon sits behind its own PCIe switch (upstream port, downstream port, GPU, all AMD
devices). The GPU function's link is the switch's internal one and always reads its maximum;
the slot's link is the switch's upstream port, so that is the address watched and judged.

GPUs are numbered in PCI address order, as ROCm numbers them (KFD topology order) on the rigs hwb
targets. Integrated GPUs (small VRAM carve-outs) are left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AMD_VENDOR = "0x1002"
MIN_VRAM_GIB = 4.0  # below this it is an APU's carve-out, not a card
_CARD = re.compile(r"^card\d+$")
_PCI_ADDR = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_GIB = 1024**3


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _num(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class AmdGpu:
    index: str
    addr: str  # the GPU function
    link: str  # the device whose upstream link is the slot: the card's switch, else the GPU
    hwmon: str | None  # hwmon directory under the device, relative to the root
    vram_gib: float | None
    device_id: str | None


def _slot_device(root: Path, addr: str) -> str:
    """The topmost AMD bridge right above the GPU (the card's switch upstream port), else addr."""
    dev = root / "sys/bus/pci/devices"
    try:
        chain = [p for p in (dev / addr).resolve().parts if _PCI_ADDR.match(p)]
    except OSError:
        return addr
    link = addr
    for parent in reversed(chain[:-1]):  # walk up from the GPU
        if _read(dev / parent / "vendor") != AMD_VENDOR:
            break
        if not (_read(dev / parent / "class") or "").startswith("0x0604"):
            break
        link = parent
    return link


def find(root: Path = Path("/")) -> list[AmdGpu]:
    drm = root / "sys/class/drm"
    if not drm.is_dir():
        return []
    found: dict[str, AmdGpu] = {}
    for card in drm.iterdir():
        if not _CARD.match(card.name):
            continue
        dev = card / "device"
        if _read(dev / "vendor") != AMD_VENDOR:
            continue
        try:
            addr = dev.resolve().name
        except OSError:
            continue
        if not _PCI_ADDR.match(addr) or addr in found:
            continue
        vram = _num(_read(dev / "mem_info_vram_total"))
        vram_gib = vram / _GIB if vram is not None else None
        if vram_gib is not None and vram_gib < MIN_VRAM_GIB:
            continue
        hw = sorted((dev / "hwmon").glob("hwmon*")) if (dev / "hwmon").is_dir() else []
        found[addr] = AmdGpu(
            index="",
            addr=addr,
            link=_slot_device(root, addr),
            hwmon=str(hw[0].relative_to(root)) if hw else None,
            vram_gib=round(vram_gib, 1) if vram_gib is not None else None,
            device_id=_read(dev / "device"),
        )
    return [
        AmdGpu(str(i), g.addr, g.link, g.hwmon, g.vram_gib, g.device_id)
        for i, g in enumerate(found[a] for a in sorted(found))
    ]


def _temps(hw: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for n in (1, 2, 3):
        label = (_read(hw / f"temp{n}_label") or "").lower()
        key = {"edge": "temp", "junction": "temp_junction", "mem": "temp_mem"}.get(label)
        if key is None and n == 1:
            key = "temp"
        if key:
            v = _num(_read(hw / f"temp{n}_input"))
            out[key] = v / 1000 if v is not None else None
    return out


def sample(root: Path, g: AmdGpu) -> dict[str, Any]:
    dev = root / "sys/bus/pci/devices" / g.addr
    row: dict[str, Any] = {"util": _num(_read(dev / "gpu_busy_percent"))}
    used = _num(_read(dev / "mem_info_vram_used"))
    row["vram_used_gib"] = round(used / _GIB, 2) if used is not None else None
    if g.hwmon:
        hw = root / g.hwmon
        row |= _temps(hw)
        power = _num(_read(hw / "power1_average"))
        if power is None:
            power = _num(_read(hw / "power1_input"))
        row["power"] = power / 1e6 if power is not None else None
        sclk = _num(_read(hw / "freq1_input"))
        row["sm_mhz"] = sclk / 1e6 if sclk is not None else None
    return row
