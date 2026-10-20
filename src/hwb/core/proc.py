"""The only place hwb spawns processes.

Rules: argv lists only (never shell=True), every call has a timeout, every call is
audit-logged (argv, exit code, duration, output tail) to the run's JSONL log.

``run`` is for short commands. ``spawn`` is for long-running children (load generators,
``nvidia-smi -lms``): each gets its own process group and is torn down when the ``with``
block exits, however it exits, so nothing keeps loading the machine after hwb stops.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

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


STOP_GRACE_S = 5.0


@dataclass
class Child:
    """A running child started by ``spawn``; stdout/stderr go to files, read them after."""

    argv: tuple[str, ...]
    popen: subprocess.Popen[bytes]
    started: float = field(default_factory=time.monotonic)

    @property
    def returncode(self) -> int | None:
        return self.popen.poll()

    def wait(self, timeout: float) -> int | None:
        """Exit code, or None if still running after ``timeout`` seconds."""
        try:
            return self.popen.wait(timeout=max(timeout, 0.0))
        except subprocess.TimeoutExpired:
            return None

    def stop(self, grace: float = STOP_GRACE_S) -> int:
        """SIGTERM the whole process group, SIGKILL whatever is left after ``grace``."""
        for sig, wait_s in ((signal.SIGTERM, grace), (signal.SIGKILL, grace)):
            if self.popen.poll() is not None:
                break
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.popen.pid, sig)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.popen.wait(timeout=wait_s)
        # Workers forked by the child (stress-ng) can outlive it; the group goes too.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.popen.pid, signal.SIGKILL)
        return self.popen.wait()


def _tail(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - _AUDIT_TAIL, 0))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


@contextlib.contextmanager
def spawn(
    argv: Sequence[str],
    *,
    stdout: Path | None = None,
    stderr: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[Child]:
    """Start ``argv`` in its own process group; it is stopped when the block exits.

    Output goes to the given files (or is discarded). Raises ToolMissingError if argv[0]
    is not found.
    """
    if not argv:
        raise ValueError("empty argv")
    exe = which(argv[0])
    if exe is None:
        raise ToolMissingError(argv[0])
    full_env = {**os.environ, **_BASE_ENV, **(env or {})}
    with contextlib.ExitStack() as files:
        out = files.enter_context(stdout.open("wb")) if stdout else subprocess.DEVNULL
        err = files.enter_context(stderr.open("wb")) if stderr else subprocess.DEVNULL
        log.debug("spawn", extra=fields(argv=list(argv)))
        popen = subprocess.Popen(
            [exe, *argv[1:]],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=full_env,
            start_new_session=True,  # own process group, so stop() reaches its workers
        )
        child = Child(tuple(argv), popen)
        try:
            yield child
        finally:
            rc = child.stop()
            log.debug(
                "exit",
                extra=fields(
                    argv=list(argv),
                    rc=rc,
                    duration_s=round(time.monotonic() - child.started, 3),
                    stdout_tail=_tail(stdout),
                    stderr_tail=_tail(stderr),
                ),
            )


@contextlib.contextmanager
def spawn_many(argvs: Sequence[Sequence[str]], *, stdout: Sequence[Path]) -> Iterator[list[Child]]:
    """``spawn`` for a set of identical workers; all are stopped when the block exits."""
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(spawn(a, stdout=o)) for a, o in zip(argvs, stdout, strict=True)]
