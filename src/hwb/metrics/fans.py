"""Fan speeds: every hwmon ``fan*_input`` (board, CPU and chassis fans, RPM) and every GPU
fan NVML reports (duty % and tach RPM per fan).

NVML is read in-process through ctypes (``libnvidia-ml.so.1`` ships with the driver), because
nvidia-smi only reports one duty % per card: no RPM, no per-fan values. A fan that is driven
but reads 0 RPM is a dead fan or a broken tach, and only the per-fan view shows it.

Both readers return nothing on machines without the files or the library (VMs, macOS,
passive server cards); they never raise.
"""

from __future__ import annotations

import ctypes
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MAX_HWMON_FANS = 16
NVML_LIB = "libnvidia-ml.so.1"
_NVML_SUCCESS = 0
_NVML_NEEDED = (
    "nvmlInit_v2",
    "nvmlDeviceGetCount_v2",
    "nvmlDeviceGetHandleByIndex_v2",
    "nvmlDeviceGetNumFans",
    "nvmlDeviceGetFanSpeed_v2",
)


@dataclass(frozen=True)
class HwmonFan:
    label: str  # fan*_label, else "<chip> fan<N>"
    input: Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def find_hwmon_fans(root: Path) -> list[HwmonFan]:
    """Every fan tachometer under /sys/class/hwmon. Chips that appear twice (two sockets,
    two identical controllers) get a ``#n`` suffix so labels stay unique."""
    chips = [
        (hw, _read(hw / "name") or hw.name)
        for hw in sorted((root / "sys/class/hwmon").glob("hwmon*"))
    ]
    seen: dict[str, int] = {}
    found: list[HwmonFan] = []
    for hw, name in chips:
        n = seen.get(name, 0)
        seen[name] = n + 1
        chip = name if n == 0 else f"{name} #{n}"
        inputs = sorted(hw.glob("fan*_input"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
        for inp in inputs:
            own = _read(hw / inp.name.replace("_input", "_label"))
            label = f"{chip} {own}" if own else f"{chip} {inp.name.replace('_input', '')}"
            found.append(HwmonFan(label, inp))
    return found[:MAX_HWMON_FANS]


def read_rpm(fan: HwmonFan) -> float | None:
    text = _read(fan.input)
    try:
        return float(text) if text else None
    except ValueError:
        return None


class _FanSpeedInfo(ctypes.Structure):  # nvmlFanSpeedInfo_t (v1)
    _fields_ = [("version", ctypes.c_uint), ("fan", ctypes.c_uint), ("speed", ctypes.c_uint)]


_FAN_SPEED_INFO_V1 = ctypes.sizeof(_FanSpeedInfo) | (1 << 24)


class NvmlFans:
    """``fans = NvmlFans.open()`` (None without NVML); ``fans.sample()``; ``fans.close()``.

    ``sample()`` returns {gpu index: [(duty %, rpm), ...]} with one entry per fan; either value
    is None when the driver does not report it (older drivers have no RPM call)."""

    def __init__(self, lib: ctypes.CDLL, handles: dict[str, tuple[ctypes.c_void_p, int]]) -> None:
        self._lib = lib
        self._handles = handles  # gpu index (as nvidia-smi prints it) -> (handle, fan count)

    @classmethod
    def open(cls) -> NvmlFans | None:
        try:
            lib = ctypes.CDLL(NVML_LIB)
        except OSError:
            return None
        if not all(hasattr(lib, f) for f in _NVML_NEEDED):  # a driver too old for per-fan calls
            return None
        if lib.nvmlInit_v2() != _NVML_SUCCESS:
            return None
        handles: dict[str, tuple[ctypes.c_void_p, int]] = {}
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) == _NVML_SUCCESS:
            for i in range(count.value):
                h = ctypes.c_void_p()
                fans = ctypes.c_uint()
                if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != _NVML_SUCCESS:
                    continue
                if lib.nvmlDeviceGetNumFans(h, ctypes.byref(fans)) == _NVML_SUCCESS and fans.value:
                    handles[str(i)] = (h, fans.value)
        if not handles:  # passive cards only: nothing to watch
            lib.nvmlShutdown()
            return None
        return cls(lib, handles)

    def sample(self) -> dict[str, list[tuple[float | None, float | None]]]:
        lib = self._lib
        out: dict[str, list[tuple[float | None, float | None]]] = {}
        rpm_api = hasattr(lib, "nvmlDeviceGetFanSpeedRPM")
        for gpu, (h, fans) in self._handles.items():
            row: list[tuple[float | None, float | None]] = []
            for f in range(fans):
                pct = ctypes.c_uint()
                duty = (
                    float(pct.value)
                    if lib.nvmlDeviceGetFanSpeed_v2(h, ctypes.c_uint(f), ctypes.byref(pct))
                    == _NVML_SUCCESS
                    else None
                )
                rpm = None
                if rpm_api:
                    info = _FanSpeedInfo(version=_FAN_SPEED_INFO_V1, fan=f)
                    if lib.nvmlDeviceGetFanSpeedRPM(h, ctypes.byref(info)) == _NVML_SUCCESS:
                        rpm = float(info.speed)
                row.append((duty, rpm))
            out[gpu] = row
        return out

    def close(self) -> None:
        self._lib.nvmlShutdown()
