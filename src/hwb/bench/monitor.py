"""hwb monitor: record the same time series as the benchmarks while the machine runs its
own workload, without adding load. For problems that only show under real jobs after a
while: PCIe links dropping a generation, temperatures creeping to the limit, throttling."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.context import RunContext
from ..core.result import CheckResult, Status
from ..metrics.sampler import Sampler
from . import analysis
from .options import BenchOptions

log = logging.getLogger(__name__)
_STATUS = {"pass": Status.PASS, "warn": Status.WARN, "fail": Status.FAIL}


def run(ctx: RunContext, opts: BenchOptions) -> None:
    log.info(
        "monitoring for %ss (Ctrl-C to stop early and keep what was recorded)", opts.duration_s
    )
    interrupted = False
    with Sampler(ctx.run_dir, name="monitor", interval=opts.interval_s) as s:
        s.phase("monitor")
        try:
            time.sleep(opts.duration_s)
        except KeyboardInterrupt:
            interrupted = True
    ts = s.result
    findings, facts = analysis.thermal_findings(ts, "monitor", sustained_load=False)
    stats = analysis.phase_stats(ts, "monitor")
    seconds = round(ts.t[-1], 1) if ts.t else 0.0
    data: dict[str, Any] = {
        **facts,
        "seconds": seconds,
        "interrupted": interrupted,
        "cpu_util_pct": stats.util,
        "load_mhz": stats.mhz,
        "load_power_w": stats.power,
        "findings": [vars(f) for f in findings],
    }
    ctx.sections.setdefault("bench", {})["monitor"] = {
        **data,
        "phases": {"monitor": vars(stats)},
        "timeseries": ts.to_report(),
    }
    level = analysis.verdict(findings)
    peak = facts.get("cpu_temp_peak_c")
    gpus = len(facts.get("pcie") or {})
    headline = f"{seconds:.0f}s recorded" + (
        f", CPU peak {peak:.0f} °C" if peak is not None else ""
    )
    headline += f", {gpus} GPU link(s) watched" if gpus else ""
    worst = [f for f in findings if f.level == level]
    summary = "; ".join(f.text for f in worst) or headline
    hint = next((f.hint for f in worst if f.hint), None)
    ctx.add(CheckResult("bench.monitor", _STATUS[level], summary, data=data, hint=hint))
    if interrupted:
        raise KeyboardInterrupt  # still exit 130, after the results are recorded
