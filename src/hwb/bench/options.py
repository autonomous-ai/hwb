from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..metrics.sampler import DEFAULT_INTERVAL_S

REFERENCE_DURATION_S = 60  # the phase lengths in cpu.py/thermal.py assume a load this long


@dataclass(frozen=True)
class BenchOptions:
    duration_s: int = REFERENCE_DURATION_S
    path: Path = Path("/var/tmp")  # noqa: S108 - disk-bench scratch dir, user-overridable
    interval_s: float = DEFAULT_INTERVAL_S


def scaled(seconds: float, duration_s: int) -> float:
    """Idle/cooldown phases shrink with very short runs (smoke tests), never grow."""
    return min(seconds, max(1.0, seconds * duration_s / REFERENCE_DURATION_S))
