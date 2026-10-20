"""CPU load generators: stress-ng when installed, otherwise plain Python workers.

stress-ng (``hwb install tools``) gives comparable bogo-ops/s scores. The Python fallback
keeps the benchmark usable on a bare rig: one busy process per worker, each reporting how
many loop iterations it managed. Its scores only compare against other fallback runs.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from ..core import proc

STRESS_NG = "stress-ng"
STRESS_METHOD = "matrixprod"  # FP-heavy, heats the cores like real compute
_SLACK_S = 15.0  # allowed overrun past the requested duration before we stop it
# stress-ng --metrics-brief row (stress-ng >= 0.13 says "metrc:", older "info:"):
#   stress-ng: metrc: [123] cpu   1234567   60.00   3836.80   0.30   20576.12   321.75
_STRESS_ROW = re.compile(
    r"(?:metrc|info):\s+\[\d+\]\s+cpu\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)
_PY_WORKER = (
    "import sys, time\n"
    "end = time.perf_counter() + float(sys.argv[1]); n = 0; x = 1\n"
    "while time.perf_counter() < end:\n"
    "    for _ in range(20000):\n"
    "        x = (x * 1103515245 + 12345) & 0x7FFFFFFF\n"
    "    n += 20000\n"
    "print(n)\n"
)


@dataclass(frozen=True)
class LoadResult:
    method: str  # "stress-ng" or "python"
    workers: int
    seconds: float
    ops_per_s: float | None  # bogo ops/s (stress-ng) or loop iterations/s (python)
    unit: str
    error: str | None = None


def parse_stress_ng(text: str) -> tuple[int, float] | None:
    """(bogo ops, bogo ops/s real time) from stress-ng --metrics-brief output."""
    m = _STRESS_ROW.search(text)
    return (int(m.group(1)), float(m.group(5))) if m else None


def method() -> str:
    return STRESS_NG if proc.have(STRESS_NG) else "python"


def run_cpu(out_dir: Path, tag: str, workers: int, seconds: float) -> LoadResult:
    """Load ``workers`` CPUs for ``seconds``; blocks until done, children always stopped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if method() == STRESS_NG:
        return _stress_ng(out_dir, tag, workers, seconds)
    return _python(out_dir, tag, workers, seconds)


def _stress_ng(out_dir: Path, tag: str, workers: int, seconds: float) -> LoadResult:
    log_path = out_dir / f"stress-ng-{tag}.log"
    argv = [
        STRESS_NG,
        "--cpu", str(workers),
        "--cpu-method", STRESS_METHOD,
        "--timeout", f"{max(round(seconds), 1)}s",
        "--metrics-brief",
    ]  # fmt: skip
    with proc.spawn(argv, stdout=log_path, stderr=log_path) as child:
        rc = child.wait(seconds + _SLACK_S)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_stress_ng(text)
    if rc is None:
        error = "stress-ng did not finish in time and was stopped"
    elif rc != 0:
        error = f"stress-ng exited with {rc}"
    elif parsed is None:
        error = "no metrics in stress-ng output"
    else:
        error = None
    return LoadResult(
        STRESS_NG, workers, seconds, parsed[1] if parsed else None, "bogo ops/s", error
    )


def _python(out_dir: Path, tag: str, workers: int, seconds: float) -> LoadResult:
    outs = [out_dir / f"pyload-{tag}-{i}.out" for i in range(workers)]
    total = 0
    with proc.spawn_many(
        [[sys.executable, "-c", _PY_WORKER, f"{seconds:.3f}"] for _ in outs], stdout=outs
    ) as children:
        if any(c.wait(seconds + _SLACK_S) is None for c in children):
            return LoadResult("python", workers, seconds, None, "loops/s", "a load worker hung")
    for path in outs:
        text = path.read_text(encoding="utf-8").strip()
        if not text.isdigit():
            return LoadResult("python", workers, seconds, None, "loops/s", "a load worker failed")
        total += int(text)
        path.unlink()
    return LoadResult("python", workers, seconds, total / seconds if seconds else None, "loops/s")
