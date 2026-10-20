"""bench cpu: idle baseline, single-thread, all-core, cooldown, all under the sampler."""

from __future__ import annotations

import os
import time
from typing import Any

from ..core.context import RunContext
from ..core.result import CheckResult, Status
from ..metrics.sampler import Sampler
from . import analysis, load
from .options import BenchOptions, scaled

IDLE_S = 5.0
SINGLE_S = 15.0
COOL_S = 10.0


def _fmt(v: float | None, spec: str = ".0f", unit: str = "") -> str:
    return "?" if v is None else f"{v:{spec}}{unit}"


def run(ctx: RunContext, opts: BenchOptions) -> None:
    workers = os.cpu_count() or 1
    out = ctx.run_dir
    with Sampler(out, name="bench-cpu", interval=opts.interval_s) as s:
        s.phase("idle")
        time.sleep(scaled(IDLE_S, opts.duration_s))
        s.phase("single-core")
        single = load.run_cpu(out, "single", 1, scaled(SINGLE_S, opts.duration_s))
        s.phase("all-core")
        multi = load.run_cpu(out, "multi", workers, float(opts.duration_s))
        s.phase("cooldown")
        time.sleep(scaled(COOL_S, opts.duration_s))
    ts = s.result
    stats = {
        p: analysis.phase_stats(ts, p) for p in ("idle", "single-core", "all-core", "cooldown")
    }
    allcore = stats["all-core"]
    scaling = (
        multi.ops_per_s / (single.ops_per_s * workers)
        if multi.ops_per_s and single.ops_per_s
        else None
    )
    throttle = ts.meta.get("cpu_throttle_events")
    data: dict[str, Any] = {
        "method": multi.method,
        "unit": multi.unit,
        "workers": workers,
        "multi_ops_per_s": multi.ops_per_s,
        "single_ops_per_s": single.ops_per_s,
        "scaling_pct": None if scaling is None else round(100 * scaling, 1),
        "load_util_pct": allcore.util,
        "load_mhz": allcore.mhz,
        "load_power_w": allcore.power,
        "cpu_temp_peak_c": analysis.peak(analysis.cpu_temp(ts)),
        "cpu_throttle_events": throttle,
    }
    ctx.sections.setdefault("bench", {})["cpu"] = {
        **data,
        "phases": {p: vars(st) for p, st in stats.items()},
        "timeseries": ts.to_report(),
    }

    error = multi.error or single.error
    if error:
        ctx.add(
            CheckResult(
                "bench.cpu",
                Status.ERROR,
                f"CPU load failed: {error}",
                data=data,
                hint=f"see {multi.method} output in {out}",
            )
        )
        return
    parts = [
        f"{_fmt(multi.ops_per_s, ',.0f')} {multi.unit} on {workers} threads "
        f"(single {_fmt(single.ops_per_s, ',.0f')}, scaling {_fmt(data['scaling_pct'], '.0f', '%')})"
    ]
    if allcore.mhz is not None:
        parts.append(f"{allcore.mhz:.0f} MHz avg")
    if data["cpu_temp_peak_c"] is not None:
        parts.append(f"peak {data['cpu_temp_peak_c']:.0f} °C")
    summary = ", ".join(parts)
    if multi.method != load.STRESS_NG:
        summary += " [python fallback]"
    status, hint = Status.PASS, None
    if allcore.util is not None and allcore.util < analysis.MIN_LOAD_UTIL_PCT:
        status = Status.WARN
        summary = f"CPU only {allcore.util:.0f}% busy under all-core load; " + summary
        hint = "something limits the CPUs: cgroup/cpuset quotas, offline cores, or a power-saving profile"
    elif throttle:
        status = Status.WARN
        summary = f"{throttle} thermal-throttle events; " + summary
        hint = "the CPU cut clocks to cool down; run hwb bench thermal and check cooling"
    elif multi.method != load.STRESS_NG:
        hint = "install stress-ng (hwb install tools) for scores comparable across rigs"
    ctx.add(CheckResult("bench.cpu", status, summary, data=data, hint=hint))
