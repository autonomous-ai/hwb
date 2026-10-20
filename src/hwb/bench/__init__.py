"""Benchmarks. Each kind exposes ``run(ctx, opts) -> None`` and reports via ctx.add/ctx.check.

Status (v0.1): interfaces only. Planned implementation, in order:

- cpu:     stress-ng --cpu (bogo-ops/s) + sysbench cpu; per-socket via numactl when NUMA > 1.
- memory:  sysbench memory / stress-ng --stream; per-NUMA-node bandwidth.
- disk:    fio seq read/write 1M QD32 + rand 4k QD64, O_DIRECT, on a file in --path (never raw devices
           unless --device is given explicitly and confirmed).
- thermal: sustained stress-ng --cpu + GPU burn for --duration while metrics.sampler records temps,
           clocks, power and throttle reasons; FAIL on thermal throttling or Xid during the run.
- gpu:     torch payload in an isolated uv venv: GEMM TFLOPS (fp32/tf32/bf16/fp16), HBM/GDDR bandwidth,
           H2D/D2H PCIe bandwidth per GPU, P2P + NCCL all-reduce across GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.context import RunContext
from ..core.result import CheckResult, Status

KINDS = ("cpu", "memory", "disk", "thermal", "gpu")


@dataclass(frozen=True)
class BenchOptions:
    duration_s: int = 60
    path: Path = Path("/var/tmp")  # noqa: S108 - disk-bench scratch dir, user-overridable


def run(ctx: RunContext, kind: str, opts: BenchOptions) -> None:
    kinds = KINDS if kind == "all" else (kind,)
    for k in kinds:
        ctx.add(CheckResult(f"bench.{k}", Status.SKIP, "not implemented yet"))
