"""Benchmarks. Each kind reports through ctx.add and puts its numbers and time series in
``ctx.sections["bench"][kind]``; the viewer charts them.

- cpu:     single-thread and all-core throughput (stress-ng, else a Python fallback),
           with utilization, clocks, temperatures and power sampled throughout.
- thermal: sustained all-core load; judges peak temperature vs the sensor limit, throttle
           events, clock sag and GPU slowdowns.
- memory, disk, gpu: not implemented yet (planned: stress-ng/sysbench, fio, torch payload).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..core.context import RunContext
from ..core.log import fields
from ..core.result import CheckResult, Status
from . import cpu, monitor, thermal
from .options import BenchOptions

__all__ = ["KINDS", "BenchOptions", "monitor", "run"]

log = logging.getLogger(__name__)

KINDS = ("cpu", "memory", "disk", "thermal", "gpu")
_IMPLEMENTED: dict[str, Callable[[RunContext, BenchOptions], None]] = {
    "cpu": cpu.run,
    "thermal": thermal.run,
}


def run(ctx: RunContext, kind: str, opts: BenchOptions) -> None:
    for k in KINDS if kind == "all" else (kind,):
        bench = _IMPLEMENTED.get(k)
        if bench is None:
            ctx.add(CheckResult(f"bench.{k}", Status.SKIP, "not implemented yet"))
            continue
        try:
            bench(ctx, opts)
        except Exception as exc:  # isolation boundary: one broken benchmark must not end the run
            log.debug("benchmark crashed", exc_info=True, extra=fields(bench=k))
            ctx.add(CheckResult(f"bench.{k}", Status.ERROR, f"{type(exc).__name__}: {exc}"))
