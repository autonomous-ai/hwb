"""The only place hwb spawns processes.

Rules: argv lists only (never shell=True), every call has a timeout, every call is
audit-logged (argv, exit code, duration, output tail) to the run's JSONL log.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import ToolMissingError
from .log import fields

log = logging.getLogger("hwb.audit")

# sbin dirs are often missing from a non-root PATH; CUDA installs nvcc outside PATH.
_EXTRA_PATH = ("/usr/local/sbin", "/usr/sbin", "/sbin", "/usr/local/cuda/bin")
# Force a stable locale so tool output is parseable.
_BASE_ENV = {"LC_ALL": "C", "LANG": "C"}
_AUDIT_TAIL = 2000


@dataclass(frozen=True)
class ProcResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def which(name: str) -> str | None:
    found = shutil.which(name)
    if found is None:
        found = shutil.which(name, path=os.pathsep.join(_EXTRA_PATH))
    return found


def have(name: str) -> bool:
    return which(name) is not None


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run(
    argv: Sequence[str],
    *,
    timeout: float = 60.0,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> ProcResult:
    """Run ``argv`` and capture output. Raises ToolMissingError if argv[0] is not found.

    A non-zero exit is NOT an exception: callers inspect ``.ok`` because probing
    commands fail routinely on half-configured machines.
    """
    if not argv:
        raise ValueError("empty argv")
    exe = which(argv[0])
    if exe is None:
        raise ToolMissingError(argv[0])

    full_env = {**os.environ, **_BASE_ENV, **(env or {})}
    log.debug("exec", extra=fields(argv=list(argv), timeout=timeout))
    start = time.monotonic()
    try:
        cp = subprocess.run(
            [exe, *argv[1:]],
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=full_env,
            check=False,
        )
        result = ProcResult(
            tuple(argv), cp.returncode, cp.stdout, cp.stderr, time.monotonic() - start
        )
    except subprocess.TimeoutExpired as exc:
        result = ProcResult(
            tuple(argv), -1, _text(exc.stdout), _text(exc.stderr), time.monotonic() - start, True
        )

    log.debug(
        "exit",
        extra=fields(
            argv=list(argv),
            rc=result.returncode,
            timed_out=result.timed_out,
            duration_s=round(result.duration_s, 3),
            stdout_tail=result.stdout[-_AUDIT_TAIL:],
            stderr_tail=result.stderr[-_AUDIT_TAIL:],
        ),
    )
    return result
