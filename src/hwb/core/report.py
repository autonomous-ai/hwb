"""Write report.json / report.md into the run directory, refresh the combined report.html
next to the run directories, and print the console summary."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .. import __version__
from .context import RunContext
from .report_html import INDEX_HTML, write_index
from .result import CheckResult, Status

log = logging.getLogger(__name__)

REPORT_SCHEMA = 2  # 2: sections.bench with time series

_COLORS = {
    Status.PASS: "\033[32m",
    Status.WARN: "\033[33m",
    Status.FAIL: "\033[31m",
    Status.SKIP: "\033[90m",
    Status.ERROR: "\033[35m",
}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # pid: concurrent runs share --out
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_payload(ctx: RunContext) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "tool": {"name": "hwb", "version": __version__},
        "command": ctx.command,
        "host": ctx.host,
        "started_at": ctx.started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": ctx.exit_code(),
        "results": [r.to_dict() for r in ctx.results],
        "sections": ctx.sections,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# hwb report — {payload['host']}",
        "",
        f"- command: `{payload['command']}`",
        f"- started: {payload['started_at']}",
        f"- hwb: {payload['tool']['version']}",
        "",
        "| status | check | summary | hint |",
        "|---|---|---|---|",
    ]
    for r in payload["results"]:
        summary = str(r["summary"]).replace("|", "\\|")
        hint = str(r["hint"] or "").replace("|", "\\|")
        lines.append(f"| {r['status'].upper()} | {r['name']} | {summary} | {hint} |")
    return "\n".join(lines) + "\n"


def write_report(ctx: RunContext) -> Path:
    payload = build_payload(ctx)
    path = ctx.run_dir / "report.json"
    _atomic_write(path, json.dumps(payload, indent=2, default=str))
    _atomic_write(ctx.run_dir / "report.md", render_markdown(payload))
    try:
        _atomic_write(ctx.run_dir.parent / INDEX_HTML, write_index(ctx.run_dir.parent))
    except (OSError, ValueError) as exc:
        # The viewer is a convenience; report.json and report.md above are the record.
        log.warning("%s not written: %s", INDEX_HTML, exc, exc_info=True)
    return path


def render_console(results: list[CheckResult], stream: TextIO = sys.stdout) -> None:
    color = stream.isatty() and "NO_COLOR" not in os.environ
    width = max((len(r.name) for r in results), default=0)
    for r in results:
        tag = f"{r.status.value.upper():5}"
        if color:
            tag = f"{_COLORS[r.status]}{tag}\033[0m"
        stream.write(f"{tag}  {r.name:<{width}}  {r.summary}\n")
        if r.hint and r.status in (Status.WARN, Status.FAIL):
            stream.write(f"       {'':<{width}}  ↳ {r.hint}\n")
