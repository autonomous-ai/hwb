"""Background time series of host and GPU state while a benchmark runs.

Host metrics are read from procfs/sysfs on a thread every ``interval`` seconds: CPU
utilization, clocks, package temperatures, package power (RAPL), memory, load and the
kernel's thermal-throttle counters. GPUs are recorded by one ``nvidia-smi -lms`` child.
Full resolution goes to CSV files in the run directory; ``timeseries()`` returns a
compact, downsampled copy for the report and the viewer's charts.

Every probe tolerates missing files, so on a VM, a laptop or macOS the series are just
empty (None), never an error. All paths go through ``root`` for tests.
"""

from __future__ import annotations

import bisect
import contextlib
import csv
import io
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core import proc
from ..core.errors import HwbError
from ..core.log import fields

log = logging.getLogger(__name__)

ROOT = Path("/")
MAX_POINTS = 300  # per series in report.json; the CSV keeps every sample
DEFAULT_INTERVAL_S = 1.0
# hwmon drivers that report CPU package/die temperatures, and the throttle point to assume
# when the sensor exposes no temp*_crit: Intel TjMax is typically 100 °C; AMD Tctl
# (k10temp/zenpower publish no limit) throttles at 95 °C on Zen desktop/workstation parts.
CPU_TEMP_CHIPS = {"coretemp": 100.0, "k10temp": 95.0, "zenpower": 95.0}
DEFAULT_CPU_CRIT_C = 100.0
MAX_TEMP_SENSORS = 8

GPU_FIELDS = (
    "index",
    "timestamp",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "memory.used",
    "clocks_throttle_reasons.active",
)
# clocks_throttle_reasons bits that mean the GPU slowed itself down.
GPU_POWER_CAP = 0x4
GPU_SLOWDOWN = 0x8 | 0x20 | 0x40 | 0x80  # HW slowdown, SW thermal, HW thermal, HW power brake
_GPU_TS = "%Y/%m/%d %H:%M:%S.%f"

# group -> (unit, how to merge samples when downsampling)
GROUPS: dict[str, tuple[str, Callable[[list[float]], float]]] = {
    "temperature": ("°C", max),
    "utilization": ("%", lambda v: sum(v) / len(v)),
    "clock": ("MHz", lambda v: sum(v) / len(v)),
    "power": ("W", lambda v: sum(v) / len(v)),
    "load": ("", lambda v: sum(v) / len(v)),
}


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _num(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


# ---------------------------------------------------------------- probes (pure over root)


def cpu_times(root: Path) -> tuple[int, int] | None:
    """(busy, total) jiffies from the aggregate line of /proc/stat."""
    text = _read(root / "proc/stat")
    if not text or not text.startswith("cpu "):
        return None
    vals = [int(v) for v in text.splitlines()[0].split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    total = sum(vals[:8])  # guest time is already inside user/nice
    return total - idle, total


def utilization(prev: tuple[int, int] | None, cur: tuple[int, int] | None) -> float | None:
    if prev is None or cur is None or cur[1] <= prev[1]:
        return None
    return max(0.0, min(100.0, 100.0 * (cur[0] - prev[0]) / (cur[1] - prev[1])))


def cpu_mhz(root: Path) -> float | None:
    """Mean current clock across logical CPUs."""
    vals = [
        v / 1000
        for p in (root / "sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
        if (v := _num(_read(p))) is not None
    ]
    if not vals:
        text = _read(root / "proc/cpuinfo") or ""
        vals = [float(m) for m in re.findall(r"^cpu MHz\s*:\s*([\d.]+)", text, re.M)]
    return sum(vals) / len(vals) if vals else None


@dataclass(frozen=True)
class TempSensor:
    label: str  # e.g. "Package id 0", "Tctl"
    input: Path
    crit: float | None


def find_cpu_temp_sensors(root: Path) -> list[TempSensor]:
    """Package/die sensors: coretemp "Package id N" (cores only if no package sensor),
    k10temp/zenpower "Tdie" (else "Tctl"), one per chip, i.e. per socket. ``crit`` falls
    back to the chip's usual throttle point when the driver publishes none."""
    chips = [
        (hw, name)
        for hw in sorted((root / "sys/class/hwmon").glob("hwmon*"))
        if (name := _read(hw / "name") or "") in CPU_TEMP_CHIPS
    ]
    found: list[TempSensor] = []
    for n, (hw, name) in enumerate(chips):
        sensors: dict[str, TempSensor] = {}
        for inp in sorted(hw.glob("temp*_input")):
            label = _read(hw / inp.name.replace("_input", "_label")) or inp.name
            crit = _num(_read(hw / inp.name.replace("_input", "_crit")))
            sensors[label] = TempSensor(label, inp, crit / 1000 if crit else CPU_TEMP_CHIPS[name])
        pkg = [s for lbl, s in sensors.items() if lbl.startswith("Package id")]
        die = [sensors[k] for k in ("Tdie", "Tctl") if k in sensors][:1]
        chosen = pkg or die or list(sensors.values())
        # "Package id N" already names the socket; AMD labels repeat per chip.
        suffix = "" if pkg or len(chips) == 1 else f" #{n}"
        found += [TempSensor(f"{s.label}{suffix}", s.input, s.crit) for s in chosen]
    return found[:MAX_TEMP_SENSORS]


def read_temp(sensor: TempSensor) -> float | None:
    v = _num(_read(sensor.input))
    return v / 1000 if v is not None else None


def rapl_domains(root: Path) -> dict[str, Path]:
    """Top-level RAPL package domains (energy counters), e.g. {"package-0": .../intel-rapl:0}."""
    out = {}
    for d in sorted((root / "sys/class/powercap").glob("intel-rapl:[0-9]")):
        name = _read(d / "name")
        if name and name.startswith("package") and (d / "energy_uj").exists():
            out[name] = d
    return out


def rapl_watts(
    prev: dict[str, float], cur: dict[str, float], ranges: dict[str, float], dt: float
) -> float | None:
    """Total package power from two energy readings, handling counter wrap-around."""
    if dt <= 0 or not prev or prev.keys() != cur.keys():
        return None
    total = 0.0
    for k, e in cur.items():
        delta = e - prev[k]
        if delta < 0:
            delta += ranges.get(k, 0.0)
        if delta < 0:
            return None
        total += delta
    return total / 1e6 / dt


def throttle_events(root: Path) -> int | None:
    """Sum of the kernel's CPU thermal-throttle counters (Intel); None where absent."""
    counts = [
        int(v)
        for p in (root / "sys/devices/system/cpu").glob(
            "cpu[0-9]*/thermal_throttle/*_throttle_count"
        )
        if (v := _read(p)) and v.isdigit()
    ]
    return sum(counts) if counts else None


def mem_used_pct(root: Path) -> float | None:
    vals = {}
    for line in (_read(root / "proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        if rest.split() and rest.split()[0].isdigit():
            vals[key] = int(rest.split()[0])
    total, avail = vals.get("MemTotal"), vals.get("MemAvailable")
    return 100.0 * (total - avail) / total if total and avail is not None else None


def load1(root: Path) -> float | None:
    text = _read(root / "proc/loadavg")
    return _num(text.split()[0]) if text else None


def parse_gpu_csv(text: str, wall0: float) -> list[dict[str, Any]]:
    """Rows of ``nvidia-smi --query-gpu=GPU_FIELDS -lms`` output; malformed lines skipped."""
    rows = []
    for raw in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(raw) != len(GPU_FIELDS):
            continue
        r = dict(zip(GPU_FIELDS, (c.strip() for c in raw), strict=True))
        try:
            ts = datetime.strptime(r["timestamp"], _GPU_TS).timestamp()
        except ValueError:
            continue
        reasons = r["clocks_throttle_reasons.active"]
        try:
            bits = int(reasons, 16) if reasons.startswith("0x") else None
        except ValueError:
            bits = None
        rows.append(
            {
                "t": ts - wall0,
                "index": r["index"],
                "util": _num(r["utilization.gpu"]),
                "temp": _num(r["temperature.gpu"]),
                "power": _num(r["power.draw"]),
                "sm_mhz": _num(r["clocks.sm"]),
                "mem_mib": _num(r["memory.used"]),
                "reasons": bits,
            }
        )
    return rows


# ---------------------------------------------------------------- the sampler


@dataclass
class Timeseries:
    interval_s: float
    t: list[float] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    series: dict[str, dict[str, Any]] = field(default_factory=dict)  # key -> {label, group, values}
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, label: str, group: str) -> list[float | None]:
        if key not in self.series:
            self.series[key] = {"label": label, "group": group, "values": [None] * len(self.t)}
        values: list[float | None] = self.series[key]["values"]
        return values

    def values(self, key: str) -> list[float | None]:
        return list(self.series.get(key, {}).get("values", []))

    def phase_slice(self, name: str) -> slice:
        for p in self.phases:
            if p["name"] == name:
                idx = [i for i, t in enumerate(self.t) if p["start"] <= t <= p["end"]]
                return slice(idx[0], idx[-1] + 1) if idx else slice(0, 0)
        return slice(0, 0)

    def to_report(self, max_points: int = MAX_POINTS) -> dict[str, Any]:
        """Compact form for report.json: at most ``max_points`` per series, 1 decimal."""
        n = len(self.t)
        step = max(1, -(-n // max_points))  # ceil
        buckets = [range(i, min(i + step, n)) for i in range(0, n, step)]
        out_series = []
        for key, s in self.series.items():
            unit, merge = GROUPS[s["group"]]
            vals = []
            for b in buckets:
                got = [s["values"][i] for i in b if s["values"][i] is not None]
                vals.append(round(merge(got), 1) if got else None)
            if any(v is not None for v in vals):
                out_series.append(
                    {
                        "key": key,
                        "label": s["label"],
                        "group": s["group"],
                        "unit": unit,
                        "values": vals,
                    }
                )
        return {
            "interval_s": round(self.interval_s * step, 3),
            "t": [round(self.t[b[0]], 1) for b in buckets],
            "phases": [
                {**p, "start": round(p["start"], 1), "end": round(p["end"], 1)} for p in self.phases
            ],
            "series": out_series,
            **({"downsampled_from": n} if step > 1 else {}),
            **self.meta,
        }


class Sampler:
    """``with Sampler(out_dir) as s: s.phase("load"); ...``; then ``s.result``."""

    def __init__(
        self,
        out_dir: Path,
        *,
        name: str = "metrics",
        interval: float = DEFAULT_INTERVAL_S,
        gpus: bool = True,
        root: Path | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.name = name
        self.interval = interval
        self.gpus = gpus
        self.root = root if root is not None else ROOT
        self.result = Timeseries(interval)
        self._rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._exit = contextlib.ExitStack()
        self._t0 = 0.0
        self._wall0 = 0.0
        self._phase: dict[str, Any] | None = None

    # -- phases
    def now(self) -> float:
        return time.monotonic() - self._t0

    def phase(self, name: str) -> None:
        """End the current phase (if any) and start ``name``."""
        # The first phase starts at 0, so the sample taken on entry belongs to it.
        t = self.now() if self._phase else 0.0
        if self._phase:
            self._phase["end"] = t
        self._phase = {"name": name, "start": t, "end": t}
        self.result.phases.append(self._phase)

    # -- lifecycle
    def __enter__(self) -> Sampler:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._t0, self._wall0 = time.monotonic(), time.time()
        self._sensors = find_cpu_temp_sensors(self.root)
        self._rapl = rapl_domains(self.root)
        self._rapl_range = {
            k: _num(_read(d / "max_energy_range_uj")) or 0.0 for k, d in self._rapl.items()
        }
        crits = [s.crit for s in self._sensors if s.crit]
        self.result.meta["cpu_crit_c"] = min(crits) if crits else None
        self.result.meta["cpu_temp_sensors"] = [s.label for s in self._sensors]
        self._gpu_csv = self.out_dir / f"{self.name}-gpu.csv"
        if self.gpus and proc.have("nvidia-smi"):
            try:
                self._exit.enter_context(
                    proc.spawn(
                        [
                            "nvidia-smi",
                            f"--query-gpu={','.join(GPU_FIELDS)}",
                            "--format=csv,noheader,nounits",
                            f"-lms={max(int(self.interval * 1000), 100)}",
                        ],
                        stdout=self._gpu_csv,
                    )
                )
            except (HwbError, OSError) as exc:
                log.warning("GPU sampling unavailable: %s", exc)
        self._thread = threading.Thread(target=self._loop, name="hwb-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._phase:
            self._phase["end"] = self.now()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 5 + 5)
        self._exit.close()  # stops nvidia-smi
        self._build()

    def _probe(self, state: dict[str, Any]) -> dict[str, Any]:
        root = self.root
        now = time.monotonic()
        cpu = cpu_times(root)
        energy = {
            k: v for k, d in self._rapl.items() if (v := _num(_read(d / "energy_uj"))) is not None
        }
        row = {
            "t": now - self._t0,
            "cpu_util": utilization(state.get("cpu"), cpu),
            "cpu_mhz": cpu_mhz(root),
            "cpu_power": rapl_watts(
                state.get("energy", {}), energy, self._rapl_range, now - state.get("time", now)
            ),
            "mem_used": mem_used_pct(root),
            "load1": load1(root),
            "throttle": throttle_events(root),
            "temps": {s.label: read_temp(s) for s in self._sensors},
        }
        state.update(cpu=cpu, energy=energy, time=now)
        return row

    def _loop(self) -> None:
        state: dict[str, Any] = {}
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._rows.append(self._probe(state))
            except Exception:  # isolation boundary: a bad sample must not kill the benchmark
                log.debug("sample failed", exc_info=True, extra=fields(sampler=self.name))
            next_tick += self.interval
            self._stop.wait(max(0.0, next_tick - time.monotonic()))

    def _build(self) -> None:
        ts = self.result
        ts.t = [r["t"] for r in self._rows]

        def put(key: str, label: str, group: str, values: Iterator[float | None]) -> None:
            col = ts.add(key, label, group)
            col[:] = list(values)

        put("cpu.util", "CPU utilization", "utilization", (r["cpu_util"] for r in self._rows))
        put("mem.used", "Memory used", "utilization", (r["mem_used"] for r in self._rows))
        put("cpu.mhz", "CPU clock (avg)", "clock", (r["cpu_mhz"] for r in self._rows))
        put("cpu.power", "CPU package power", "power", (r["cpu_power"] for r in self._rows))
        put("load1", "Load average (1 min)", "load", (r["load1"] for r in self._rows))
        for s in self._sensors:
            put(
                f"cpu.temp.{s.label}",
                f"CPU {s.label}",
                "temperature",
                (r["temps"].get(s.label) for r in self._rows),
            )
        throttles = [r["throttle"] for r in self._rows if r["throttle"] is not None]
        ts.meta["cpu_throttle_events"] = (
            throttles[-1] - throttles[0] if len(throttles) > 1 else None
        )

        gpu_rows = (
            parse_gpu_csv(_read(self._gpu_csv) or "", self._wall0) if self._gpu_csv.exists() else []
        )
        self._write_csv(ts)
        if gpu_rows:
            self._merge_gpus(ts, gpu_rows)

    def _merge_gpus(self, ts: Timeseries, rows: list[dict[str, Any]]) -> None:
        """Put each GPU sample on the nearest host tick, so every series shares one time axis."""
        if not ts.t:
            return
        slow: dict[str, int] = {}
        capped: dict[str, int] = {}
        for r in rows:
            j = bisect.bisect_left(ts.t, r["t"])
            i = min(
                (k for k in (j - 1, j) if 0 <= k < len(ts.t)), key=lambda k: abs(ts.t[k] - r["t"])
            )
            if abs(ts.t[i] - r["t"]) > self.interval:
                continue
            g = r["index"]
            for metric, label, group in (
                ("temp", "temperature", "temperature"),
                ("util", "utilization", "utilization"),
                ("power", "power", "power"),
                ("sm_mhz", "SM clock", "clock"),
            ):
                ts.add(f"gpu{g}.{metric}", f"GPU{g} {label}", group)[i] = r[metric]
            if r["reasons"] is not None:
                slow[g] = slow.get(g, 0) + bool(r["reasons"] & GPU_SLOWDOWN)
                capped[g] = capped.get(g, 0) + bool(r["reasons"] & GPU_POWER_CAP)
        ts.meta["gpu_slowdown_samples"] = slow
        ts.meta["gpu_power_capped_samples"] = capped

    def _write_csv(self, ts: Timeseries) -> None:
        cols = [
            "t_s",
            "phase",
            "cpu_util_pct",
            "cpu_mhz",
            "cpu_power_w",
            "mem_used_pct",
            "load1",
            "throttle_count",
        ]
        cols += [f"temp_c[{s.label}]" for s in self._sensors]
        path = self.out_dir / f"{self.name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in self._rows:
                phase = next((p["name"] for p in ts.phases if p["start"] <= r["t"] <= p["end"]), "")
                w.writerow(
                    [round(r["t"], 3), phase]
                    + [
                        r[k] if r[k] is None else round(r[k], 2)
                        for k in ("cpu_util", "cpu_mhz", "cpu_power", "mem_used", "load1")
                    ]
                    + [r["throttle"]]
                    + [r["temps"].get(s.label) for s in self._sensors]
                )
        ts.meta["csv"] = [path.name] + ([self._gpu_csv.name] if self._gpu_csv.exists() else [])
