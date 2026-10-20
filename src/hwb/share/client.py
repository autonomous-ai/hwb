"""Talk to an hwb gateway: upload a sanitized bundle, delete it again, draw its QR code.

stdlib only (urllib), so it works from the bare rig. HTTPS is required except for localhost,
which is what ``wrangler dev`` serves.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import __version__
from ..core.errors import HwbError

# The gateway releases submit to (gateway/ in this repo, deployed on Cloudflare). Override per
# run with --gateway or HWB_GATEWAY, e.g. for a self-hosted gateway or `wrangler dev`.
DEFAULT_GATEWAY = "https://hwb-gateway.autonomous-ai.workers.dev"
ENV_GATEWAY = "HWB_GATEWAY"  # a URL, or "off" to disable sharing
DISABLED = "off"
# The public gateway's upload cap (MAX_REPORT_BYTES in gateway/wrangler.toml). Checked locally
# so an oversized report is refused before anything is sent.
MAX_UPLOAD_BYTES = 512_000
ENV_TOKEN = "HWB_GATEWAY_TOKEN"  # noqa: S105 - name of the env var, not a secret
SUBMISSIONS = "submissions.jsonl"
TIMEOUT_S = 60.0
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_LOCAL = {"localhost", "127.0.0.1", "::1"}
_MAX_QR = 177  # modules in a version 40 QR code
_QUIET = 4  # light border the QR spec asks for


class GatewayError(HwbError):
    pass


@dataclass(frozen=True)
class Submission:
    id: str
    url: str
    qr: list[str]
    delete_token: str
    created_at: str
    retention_days: int | None


def resolve_gateway(flag: str | None = None) -> str:
    """--gateway, else $HWB_GATEWAY, else the built-in DEFAULT_GATEWAY."""
    url = flag or os.environ.get(ENV_GATEWAY) or DEFAULT_GATEWAY
    if url.strip().lower() == DISABLED:
        raise GatewayError(f"sharing is disabled ({ENV_GATEWAY}={DISABLED})")
    return url


def sharing_enabled() -> bool:
    return (os.environ.get(ENV_GATEWAY) or "").strip().lower() != DISABLED


def gateway_base(url: str | None) -> str:
    if not url:
        raise GatewayError(f"no gateway configured: pass --gateway URL or set {ENV_GATEWAY}")
    parts = urlsplit(url.strip())
    local = parts.hostname in _LOCAL
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise GatewayError(
            f"gateway must use https:// (got {url!r}); http is allowed only for localhost"
        )
    if not parts.hostname:
        raise GatewayError(f"not a URL: {url!r}")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def _web_url(url: object) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOCAL)


def _request(
    method: str, url: str, token: str | None, body: bytes | None = None
) -> tuple[int, Any]:
    headers = {"User-Agent": f"hwb/{__version__}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310 - scheme checked in gateway_base
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error") or exc.reason
        except (ValueError, AttributeError):
            message = exc.reason
        if exc.code == HTTPStatus.UNAUTHORIZED:
            message = f"{message}; set {ENV_TOKEN} to the upload token for this gateway"
        retry = exc.headers.get("Retry-After") if exc.headers else None
        if exc.code in (HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE) and retry:
            message = f"{message} (retry in {retry} s)"
        raise GatewayError(f"gateway refused the request ({exc.code}): {message}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise GatewayError(f"cannot reach gateway {url}: {reason}") from None
    except ValueError:
        raise GatewayError(f"gateway sent a response that is not JSON ({url})") from None


def encode_bundle(bundle: dict[str, Any]) -> bytes:
    return json.dumps(bundle, default=str, separators=(",", ":")).encode()


def submit(base: str, token: str | None, bundle: dict[str, Any]) -> Submission:
    status, data = _request("POST", f"{base}/api/reports", token, encode_bundle(bundle))
    if status != 201 or not isinstance(data, dict):
        raise GatewayError(f"unexpected gateway response ({status})")
    qr = data.get("qr")
    if not (
        isinstance(data.get("id"), str)
        and _UUID.match(data["id"])
        and _web_url(data.get("url"))
        and isinstance(data.get("delete_token"), str)
        and isinstance(qr, list)
        and 0 < len(qr) <= _MAX_QR
        and all(isinstance(r, str) and len(r) == len(qr) and set(r) <= {"0", "1"} for r in qr)
    ):
        raise GatewayError("gateway response is missing or has malformed fields")
    days = data.get("retention_days")
    return Submission(
        id=data["id"],
        url=data["url"],
        qr=qr,
        delete_token=data["delete_token"],
        created_at=str(data.get("created_at") or ""),
        retention_days=days if isinstance(days, int) else None,
    )


def delete(base: str, report_id: str, token: str) -> None:
    if not _UUID.match(report_id):
        raise GatewayError(f"not a report id: {report_id!r}")
    _request("DELETE", f"{base}/api/reports/{report_id}", token)


def render_qr(rows: list[str], *, color: bool) -> str:
    """Two QR rows per text line using half blocks, with the quiet zone the spec asks for.

    With color, black/white are set explicitly so it scans on any terminal theme. Without,
    light modules are drawn as blocks, which reads correctly on the usual dark background.
    """
    size = len(rows)
    pad = "0" * (size + 2 * _QUIET)
    grid = [pad] * _QUIET + [f"{'0' * _QUIET}{r}{'0' * _QUIET}" for r in rows] + [pad] * _QUIET
    if len(grid) % 2:
        grid.append(pad)
    lines = []
    for top, bottom in zip(grid[0::2], grid[1::2], strict=True):
        if color:
            cells = (
                f"\033[{30 if t == '1' else 97};{40 if b == '1' else 107}m▀"
                for t, b in zip(top, bottom, strict=True)
            )
            lines.append("".join(cells) + "\033[0m")
        else:
            glyph = {("0", "0"): "█", ("0", "1"): "▀", ("1", "0"): "▄", ("1", "1"): " "}
            lines.append("".join(glyph[t, b] for t, b in zip(top, bottom, strict=True)))
    return "\n".join(lines)


def record(out: Path, base: str, sub: Submission, run_ids: list[str]) -> Path:
    """Append the submission (with its delete token) to <out>/submissions.jsonl, mode 0600."""
    path = out / SUBMISSIONS
    entry = {
        "id": sub.id,
        "url": sub.url,
        "gateway": base,
        "created_at": sub.created_at or datetime.now(timezone.utc).isoformat(),
        "runs": run_ids,
        "delete_token": sub.delete_token,
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return path


def find_submission(out: Path, report_id: str) -> dict[str, Any] | None:
    try:
        lines = (out / SUBMISSIONS).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("id") == report_id:
            return entry
    return None
