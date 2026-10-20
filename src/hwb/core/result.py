"""The result model every check and benchmark reports through."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"  # noqa: S105 - not a password
    WARN = "warn"  # works, but something an operator should look at
    FAIL = "fail"  # the rig is not acceptable as-is
    SKIP = "skip"  # not applicable, or a prerequisite is missing
    ERROR = "error"  # hwb itself crashed while checking; says nothing about the hardware


@dataclass
class CheckResult:
    name: str
    status: Status
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    hint: str | None = None  # remediation the operator can act on

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "hint": self.hint,
            "data": self.data,
        }
