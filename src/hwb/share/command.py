"""`hwb submit`: preview, confirm, upload; print the link and a QR code.

Uploading publishes the report to anyone with the link, so it is never implicit: the
operator sees what will be sent and confirms (or passes --yes).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.errors import HwbError, UserAbortError
from ..core.report_html import collect_runs
from . import client
from .sanitize import describe_redactions, local_identity, sanitize_bundle

log = logging.getLogger(__name__)

PREVIEW = "submission-preview.json"


def _counts(report: dict[str, Any]) -> str:
    statuses = [str(r.get("status")) for r in report.get("results") or []]
    parts = [
        f"{statuses.count(s)} {s}"
        for s in ("fail", "error", "warn", "pass", "skip")
        if s in statuses
    ]
    return ", ".join(parts) or "no checks"


def _confirm(question: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise UserAbortError("refusing to upload non-interactively without --yes")
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        raise UserAbortError("cancelled")


def submit_runs(
    out: Path,
    *,
    gateway: str | None,
    token: str | None,
    label: str | None,
    only: Sequence[str] = (),
    assume_yes: bool = False,
    dry_run: bool = False,
) -> int:
    runs = collect_runs(out)
    if only:
        missing = set(only) - {r["id"] for r in runs}
        if missing:
            raise HwbError(f"no such run in {out}: {', '.join(sorted(missing))}")
        runs = [r for r in runs if r["id"] in only]
    if not runs:
        raise HwbError(f"no runs to submit in {out} (run hwb info/detect/all first)")

    names, machine_id, home = local_identity()
    bundle, redactions = sanitize_bundle(
        runs, label=label, extra_hosts=names, machine_id=machine_id, home=home
    )

    print(f"Runs to submit from {out}:")
    for r in bundle["runs"]:
        rep = r["report"]
        print(f"  {rep.get('command', '?'):8} {rep.get('started_at', '')[:19]}  {_counts(rep)}")
    print(
        f"Rig appears as {bundle['runs'][0]['report']['host']}"
        + (f', label "{bundle["label"]}"' if "label" in bundle else "")
    )
    removed = describe_redactions(redactions)
    print(
        "Removed before upload: " + ("; ".join(removed) if removed else "nothing identifying found")
    )
    size = len(client.encode_bundle(bundle))
    if size > client.MAX_UPLOAD_BYTES:
        raise HwbError(
            f"report is {size} bytes, over the gateway's {client.MAX_UPLOAD_BYTES}-byte limit; "
            f"submit fewer runs with --run DIR"
        )

    if dry_run:
        path = out / PREVIEW
        path.write_text(json.dumps(bundle, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Dry run: the exact upload ({size} bytes) is in {path}. Nothing was sent.")
        return 0

    base = client.gateway_base(client.resolve_gateway(gateway))
    print(f"Destination: {base}  ({size} bytes). Anyone with the link will be able to view it.")
    _confirm("Submit this report?", assume_yes)
    sub = client.submit(base, token, bundle)
    saved = client.record(out, base, sub, [r["id"] for r in runs])
    log.info("submitted report %s", sub.id)

    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(f"\nReport: {sub.url}\n")
    print(client.render_qr(sub.qr, color=color))
    print()
    if sub.retention_days:
        print(f"Kept for {sub.retention_days} days.")
    print(f"Delete it with: hwb --out {out} submit --delete {sub.id}  (token saved in {saved})")
    return 0


def delete_submission(out: Path, report_id: str, *, gateway: str | None, assume_yes: bool) -> int:
    entry = client.find_submission(out, report_id)
    if entry is None or not entry.get("delete_token"):
        raise HwbError(f"no delete token for {report_id} in {out / client.SUBMISSIONS}")
    # The gateway it was submitted to wins over the current default.
    base = client.gateway_base(gateway or entry.get("gateway") or client.resolve_gateway())
    _confirm(f"Delete report {report_id} from {base}?", assume_yes)
    client.delete(base, report_id, str(entry["delete_token"]))
    print(f"Deleted {entry.get('url') or report_id}")
    return 0
