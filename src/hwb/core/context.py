"""Per-invocation state: output directory, collected results and report sections."""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .log import fields
from .result import CheckResult, Status

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    run_dir: Path
    command: str
    host: str
    started_at: datetime
    results: list[CheckResult] = field(default_factory=list)
    sections: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, base: Path, command: str) -> RunContext:
        started = datetime.now(timezone.utc)
        host = socket.gethostname().split(".")[0] or "unknown"
        run_dir = base / f"{started:%Y%m%dT%H%M%SZ}-{host}-{command}"
        run_dir.mkdir(parents=True, exist_ok=False, mode=0o750)
        return cls(run_dir=run_dir, command=command, host=host, started_at=started)

    def add(self, *results: CheckResult) -> None:
        for r in results:
            log.info(
                f"{r.status.value.upper():5} {r.name}: {r.summary}",
                extra=fields(check=r.name, status=r.status.value, data=r.data, hint=r.hint),
            )
            self.results.append(r)

    def check(self, name: str, fn: Callable[[], CheckResult]) -> None:
        """Run one check; a crash becomes an ERROR result instead of aborting the suite."""
        try:
            result = fn()
        except Exception as exc:  # isolation boundary between checks
            log.debug("check crashed", exc_info=True, extra=fields(check=name))
            result = CheckResult(name, Status.ERROR, f"{type(exc).__name__}: {exc}")
        self.add(result)

    def exit_code(self) -> int:
        statuses = {r.status for r in self.results}
        if Status.FAIL in statuses:
            return 1
        if Status.ERROR in statuses:
            return 2
        return 0
