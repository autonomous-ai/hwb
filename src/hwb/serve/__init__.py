"""Real-model smoke test with vLLM.

Status (v0.1): interface only. Planned flow:

1. Ensure uv (bootstrap if missing), create venv at $HWB_HOME/venvs/vllm (default
   ~/.cache/hwb), ``uv pip install vllm`` (pinned in VLLM_SPEC; Blackwell needs a cu128+ build).
2. Launch ``vllm serve <model> --tensor-parallel-size <tp>`` in its own process group,
   stdout/stderr teed to <run_dir>/vllm.log; poll GET /health until ready or timeout.
3. Smoke request (/v1/chat/completions) then ``vllm bench serve`` with a fixed prompt set;
   record throughput, TTFT, ITL p50/p99 into the report while metrics.sampler records GPU stats.
4. Always tear down: SIGTERM the process group, SIGKILL after a grace period (try/finally +
   signal handler), so no orphaned server keeps holding GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.context import RunContext
from ..core.result import CheckResult, Status

DEFAULT_MODEL = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class ServeOptions:
    model: str = DEFAULT_MODEL
    tensor_parallel: int | None = None  # default: number of GPUs
    port: int = 8000
    ready_timeout_s: int = 900


def run(ctx: RunContext, opts: ServeOptions) -> None:
    ctx.add(
        CheckResult("serve.vllm", Status.SKIP, "not implemented yet", data={"model": opts.model})
    )
