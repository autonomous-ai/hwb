"""Minimal streaming client for vLLM's OpenAI-compatible API (stdlib only) with timings.

Per request: time to first token (any of content, reasoning or a tool call), time per
output token (from the server's usage count, so speculative decoding that emits several
tokens per chunk is still measured correctly), the longest gap between chunks, the text,
the reasoning and reassembled tool calls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .. import __version__


@dataclass
class Reply:
    ok: bool = False
    error: str | None = None
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ttft_s: float | None = None
    duration_s: float = 0.0
    max_gap_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first."""
        if self.ttft_s is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        return (self.duration_s - self.ttft_s) / (self.completion_tokens - 1)


def _headers(api_key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "User-Agent": f"hwb/{__version__}"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def get_json(url: str, *, api_key: str | None = None, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers=_headers(api_key))  # noqa: S310 - http(s) endpoint given by the operator
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read() or b"null")


def is_ready(base: str, *, api_key: str | None = None) -> bool:
    try:
        req = urllib.request.Request(f"{base}/health", headers=_headers(api_key))  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def served_models(base: str, *, api_key: str | None = None) -> list[str]:
    data = get_json(f"{base}/v1/models", api_key=api_key)
    return [str(m.get("id")) for m in (data or {}).get("data", []) if isinstance(m, dict)]


def chat(
    base: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    api_key: str | None = None,
    timeout: float = 600.0,
    **extra: Any,
) -> Reply:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    req = urllib.request.Request(  # noqa: S310 - http(s) endpoint given by the operator
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={**_headers(api_key), "Accept": "text/event-stream"},
        method="POST",
    )
    r = Reply()
    calls: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()
    last: float | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    r.prompt_tokens = chunk["usage"].get("prompt_tokens")
                    r.completion_tokens = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    tools = delta.get("tool_calls") or []
                    for tc in tools:
                        slot = calls.setdefault(
                            int(tc.get("index", 0)), {"name": "", "arguments": ""}
                        )
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if piece or think or tools:
                        now = time.perf_counter()
                        if r.ttft_s is None:
                            r.ttft_s = now - t0
                        elif last is not None:
                            r.max_gap_s = max(r.max_gap_s or 0.0, now - last)
                        last = now
                    r.text += piece
                    r.reasoning += think
        r.ok = True
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("message") or exc.reason
        except (ValueError, AttributeError):
            detail = exc.reason
        r.error = f"HTTP {exc.code}: {str(detail)[:300]}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        r.error = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
    r.duration_s = time.perf_counter() - t0
    r.tool_calls = [calls[i] for i in sorted(calls)]
    return r


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100); None for no values."""
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)
