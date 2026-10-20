"""Logging setup: JSON lines to the run directory, human-readable lines to stderr.

Attach structured data with ``log.info("msg", extra=fields(key=value))``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FILE = "hwb.log.jsonl"


def fields(**kw: Any) -> dict[str, Any]:
    return {"fields": kw}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        line = f"[{record.levelname.lower():>7}] {record.getMessage()}"
        extra = getattr(record, "fields", None)
        if self.verbose and isinstance(extra, dict) and extra:
            line += "  " + " ".join(f"{k}={v!r}" for k, v in extra.items())
        if record.exc_info and self.verbose:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(run_dir: Path | None, *, verbose: bool) -> None:
    root = logging.getLogger("hwb")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(ConsoleFormatter(verbose))
    root.addHandler(console)

    if run_dir is not None:
        file_handler = logging.FileHandler(run_dir / LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    logging.captureWarnings(True)
