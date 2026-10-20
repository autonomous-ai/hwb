"""bench thermal: sustained all-core load, then cooldown; judges temperatures, throttling,
clock sag and GPU slowdowns from the time series. GPUs are only monitored here: loading
them needs the torch payload of `bench gpu`."""

from __future__ import annotations

import os
import time
from typing import Any

from ..core.context import RunContext
from ..core.result import CheckResult, Status
from ..metrics.sampler import Sampler
from . import analysis, load
from .options import BenchOptions, scaled

IDLE_S = 10.0
COOL_S = 30.0
_STATUS = {"pass": Status.PASS, "warn": Status.WARN, "fail": Status.FAIL}


def run(ctx: RunContext, opts: BenchOptions) -> None:
    workers = os.cpu_count() or 1
    with Sampler(ctx.run_dir, name="bench-thermal", interval=opts.interval_s) as s:
        s.phase("idle")
        time.sleep(scaled(IDLE_S, opts.duration_s))
        s.phase("load")
        result = load.run_cpu(ctx.run_dir, "thermal", workers, float(opts.duration_s))
        s.phase("cooldown")
        time.sleep(scaled(COOL_S, opts.duration_s))
    ts = s.result
    findings, facts = analysis.thermal_findings(ts, "load")
    stats = {p: analysis.phase_stats(ts, p) for p in ("idle", "load", "cooldown")}
    cool = ts.phase_slice("cooldown")
    temp = analysis.cpu_temp(ts)
    data: dict[str, Any] = {
        **facts,
        "method": result.method,
        "load_seconds": opts.duration_s,
        "idle_temp_c": stats["idle"].temp_mean,
        "load_mhz": stats["load"].mhz,
        "load_power_w": stats["load"].power,
        "cooldown_end_temp_c": analysis.mean(analysis.edges(temp[cool])[1]) if temp[cool] else None,
        "findings": [vars(f) for f in findings],
    }
    ctx.sections.setdefault("bench", {})["thermal"] = {
        **data,
        "phases": {p: vars(st) for p, st in stats.items()},
        "timeseries": ts.to_report(),
    }

    if result.error:
        ctx.add(
            CheckResult(
                "bench.thermal", Status.ERROR, f"CPU load failed: {result.error}", data=data
            )
        )
        return
    if not analysis.cpu_temp_keys(ts):
        ctx.add(
            CheckResult(
                "bench.thermal",
                Status.SKIP,
                "no CPU temperature sensors (hwmon coretemp/k10temp); clocks and power still recorded",
                data=data,
                hint="load the sensor driver: modprobe coretemp (Intel) or k10temp (AMD)",
            )
        )
        return
    level = analysis.verdict(findings)
    worst = [f for f in findings if f.level == level] or findings
    headline = (
        f"{opts.duration_s}s all-core: peak {facts['cpu_temp_peak_c']:.0f} °C "
        f"({facts.get('cpu_temp_margin_c', 0):.0f} °C below limit)"
    )
    summary = "; ".join([f.text for f in worst if f.level != "note"] or [headline])
    notes = [f.text for f in findings if f.level == "note"]
    if notes:
        summary += " · " + "; ".join(notes)
    hint = next((f.hint for f in worst if f.hint), None)
    ctx.add(CheckResult("bench.thermal", _STATUS[level], summary, data=data, hint=hint))
