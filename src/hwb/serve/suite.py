"""What `hwb serve` asks the model: answer checks first, then a load sweep.

The answer checks catch a server that is up but wrong (bad quantization, wrong chat
template, broken tool-call or reasoning parser). They run with reasoning switched off and
temperature 0, so answers are short and deterministic. The load sweep sends distinct
prompts (no prefix-cache shortcuts) and forces a fixed output length with vLLM's
``ignore_eos``, so tokens/s are comparable between runs and rigs.

The default sweep's prompts are one sentence, so it measures decode. Profiles (``--profiles``)
add the input/output lengths published benchmarks use (vLLM ``bench serve``, NVIDIA, InferenceMAX):
long prompts load prefill, and with tensor parallelism the PCIe links between the GPUs.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import HwbError
from .client import Reply, chat, get_json, percentile

NEEDLE = "4817"
# Unknown model: the switches the common chat templates use (Qwen/GLM: enable_thinking,
# DeepSeek: thinking). Templates ignore keys they don't use.
NO_THINKING_ANY: dict[str, Any] = {"enable_thinking": False, "thinking": False}
_FILLER = (
    "The archive room held ledgers from the harbor office: shipping manifests, tide tables, "
    "repair invoices for the north pier and letters about the price of rope. "
)


def _haystack(words: int = 6000) -> str:
    chunk = _FILLER.split()
    text = [chunk[i % len(chunk)] for i in range(words)]
    text.insert(words * 2 // 3, f"Note for whoever reads this: the vault code is {NEEDLE}.")
    return " ".join(text)


def _json_answer(text: str) -> Any:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def _check_json(r: Reply) -> tuple[bool, str]:
    try:
        data = _json_answer(r.text)
    except ValueError:
        return False, "answer is not valid JSON"
    ok = isinstance(data, dict) and str(data.get("city", "")).lower() == "austin"
    return ok, "" if ok else f"expected city Austin, got {data!r}"[:200]


def _check_tool(r: Reply) -> tuple[bool, str]:
    if not r.tool_calls:
        return False, "no tool call (check --tool-call-parser / --enable-auto-tool-choice)"
    call = r.tool_calls[0]
    try:
        args = json.loads(call["arguments"] or "{}")
    except ValueError:
        return False, f"tool arguments are not JSON: {call['arguments'][:120]!r}"
    ok = call["name"] == "get_weather" and "chicago" in str(args.get("city", "")).lower()
    return ok, "" if ok else f"unexpected tool call {call['name']}({args})"[:200]


def _contains(needle: str) -> Callable[[Reply], tuple[bool, str]]:
    def check(r: Reply) -> tuple[bool, str]:
        ok = needle.lower() in r.text.lower()
        return ok, "" if ok else f"expected {needle!r} in the answer, got {r.text.strip()[:120]!r}"

    return check


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a US city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@dataclass(frozen=True)
class Check:
    name: str
    prompt: str
    verify: Callable[[Reply], tuple[bool, str]]
    max_tokens: int = 256
    tools: list[dict[str, Any]] = field(default_factory=list)


CHECKS: tuple[Check, ...] = (
    Check("arithmetic", "What is 17 * 23? Reply with only the number.", _contains("391"), 32),
    Check(
        "json",
        'Give the capital of Texas as a JSON object with the keys "city" and "state". '
        "Reply with the JSON only.",
        _check_json,
        96,
    ),
    Check(
        "code",
        "Write a Python function is_prime(n) that returns True if n is prime. Reply with only the code.",
        _contains("def is_prime"),
        256,
    ),
    Check(
        "long_context",
        _haystack() + "\n\nWhat is the vault code mentioned above? Reply with only the code.",
        _contains(NEEDLE),
        32,
    ),
    Check(
        "tool_call",
        "What's the weather in Chicago right now?",
        _check_tool,
        256,
        tools=[WEATHER_TOOL],
    ),
)


def run_checks(
    base: str,
    model: str,
    *,
    no_thinking: dict[str, Any],
    tools: bool,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in CHECKS:
        if c.tools and not tools:
            out.append(
                {"name": c.name, "status": "skip", "detail": "no tool-call parser configured"}
            )
            continue
        extra: dict[str, Any] = {"chat_template_kwargs": no_thinking} if no_thinking else {}
        if c.tools:
            extra |= {"tools": c.tools, "tool_choice": "auto"}
        r = chat(
            base,
            model,
            [{"role": "user", "content": c.prompt}],
            max_tokens=c.max_tokens,
            api_key=api_key,
            **extra,
        )
        ok, detail = c.verify(r) if r.ok else (False, r.error or "request failed")
        if not ok and r.ok and not r.text.strip() and not r.tool_calls and r.reasoning:
            detail = (
                f"no answer: the {c.max_tokens}-token budget went to reasoning; the server did not "
                "accept the switch to turn reasoning off (pass --preset for this model)"
            )
        out.append(
            {
                "name": c.name,
                "status": "pass" if ok else "fail",
                "detail": detail,
                "answer": (r.text.strip() or json.dumps(r.tool_calls))[:300],
                "ttft_s": r.ttft_s,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
        )
    return out


TOPICS = (
    "the history of the printing press", "how bridges carry load", "the water cycle",
    "why the sky is blue", "how vaccines train the immune system", "the life of honeybees",
    "the design of a jet engine", "how compilers optimize loops", "plate tectonics",
    "the economics of container shipping", "how a refrigerator moves heat", "coral reefs",
)  # fmt: skip


@dataclass(frozen=True)
class Profile:
    name: str
    input_tokens: int
    output_tokens: int
    about: str


# Input/output lengths as published benchmarks pair them (ISL/OSL).
PROFILES: tuple[Profile, ...] = (
    Profile("chat", 1024, 1024, "chat: 1k in, 1k out"),
    Profile("rag", 8192, 1024, "long prompt (RAG, summaries, code review): 8k in, 1k out"),
    Profile("reasoning", 1024, 8192, "long answer (reasoning, generation): 1k in, 8k out"),
    Profile("agent", 32768, 1024, "coding-agent context: 32k in, 1k out"),
)
_LEN = re.compile(r"^(\d+)(k?)/(\d+)(k?)$")


def _tokens(n: str, k: str) -> int:
    return int(n) * (1024 if k else 1)


def parse_profiles(spec: str | None) -> tuple[Profile, ...]:
    """``chat,rag``, ``all``, or input/output lengths such as ``4k/512``."""
    out: list[Profile] = []
    for item in (spec or "").replace(" ", "").lower().split(","):
        if not item:
            continue
        if item == "all":
            out += [p for p in PROFILES if p not in out]
        elif known := next((p for p in PROFILES if p.name == item), None):
            out.append(known)
        elif m := _LEN.match(item):
            isl, osl = _tokens(m.group(1), m.group(2)), _tokens(m.group(3), m.group(4))
            if not isl or not osl:
                raise HwbError(f"profile {item!r}: lengths must be above 0")
            out.append(Profile(item, isl, osl, f"{isl} in, {osl} out"))
        else:
            names = ", ".join(p.name for p in PROFILES)
            raise HwbError(f"unknown profile {item!r}: use {names}, all, or IN/OUT such as 4k/512")
    return tuple(out)


_VOCAB = (
    "harbor ledger manifest tide pier rope invoice cargo crane berth vessel captain pilot "
    "anchor channel buoy freight customs warehouse dock crew engine hull keel rudder sail mast "
    "deck cabin galley chart compass beacon storm current wave shore reef lighthouse signal "
    "voyage route port schedule tonnage container pallet forklift gantry quay mooring tug barge "
    "ferry tanker draft ballast bunker fuel valve pump boiler turbine shaft propeller winch cable"
)
_WORDS = tuple(_VOCAB.split())


def filler(seed: int, words: int) -> str:
    """Prose-like text of about ``words`` words, different for every seed."""
    rng = random.Random(seed)  # noqa: S311 - benchmark text, not secrets
    out, sentence = [], []
    for _ in range(words):
        sentence.append(rng.choice(_WORDS))
        if len(sentence) >= rng.randint(8, 16):
            out.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    if sentence:
        out.append(" ".join(sentence).capitalize() + ".")
    return " ".join(out)


def _prompt(i: int, words: int) -> str:
    # The request number leads, so no two prompts share a cacheable prefix.
    return f"[request {i}] Read these harbor notes, then write a long report on them.\n\n" + filler(
        i, words
    )


def tokens_per_word(
    base: str, model: str, *, no_thinking: dict[str, Any], api_key: str | None = None
) -> tuple[float, int]:
    """(tokens per filler word, fixed prompt overhead) from the server's own token counts, so a
    profile's prompt is as long as it says for this model's tokenizer. Falls back to 1.3 / 64."""
    extra: dict[str, Any] = {"chat_template_kwargs": no_thinking} if no_thinking else {}
    counts = []
    for words in (0, 2000):
        r = chat(
            base,
            model,
            [{"role": "user", "content": _prompt(-1, words)}],
            max_tokens=1,
            api_key=api_key,
            **extra,
        )
        counts.append(r.prompt_tokens if r.ok else None)
    if counts[0] is None or counts[1] is None or counts[1] <= counts[0]:
        return 1.3, 64
    return (counts[1] - counts[0]) / 2000, counts[0]


def max_model_len(base: str, model: str, *, api_key: str | None = None) -> int | None:
    try:
        data = get_json(f"{base}/v1/models", api_key=api_key)
    except (OSError, ValueError):
        return None
    for m in (data or {}).get("data", []):
        if isinstance(m, dict) and m.get("id") == model and isinstance(m.get("max_model_len"), int):
            return int(m["max_model_len"])
    return None


def _ms(v: float | None) -> float | None:
    return None if v is None else round(v * 1000, 1)


def run_level(
    base: str,
    model: str,
    *,
    concurrency: int,
    requests: int,
    max_tokens: int,
    no_thinking: dict[str, Any],
    api_key: str | None = None,
    offset: int = 0,
    prompt_words: int = 0,
) -> dict[str, Any]:
    """`requests` streaming requests, `concurrency` at a time; aggregate stats. With
    ``prompt_words`` each prompt carries that much distinct text (a profile's input length)."""

    def one(i: int) -> Reply:
        topic = TOPICS[(offset + i) % len(TOPICS)]
        prompt = (
            _prompt(offset + i, prompt_words)
            if prompt_words
            else f"[request {offset + i}] Write a long, detailed explanation of {topic}."
        )
        extra: dict[str, Any] = {"ignore_eos": True, "min_tokens": max_tokens}
        if no_thinking:
            extra["chat_template_kwargs"] = no_thinking
        return chat(
            base,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            api_key=api_key,
            **extra,
        )

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(one, range(requests)))
    wall = time.perf_counter() - t0
    ok = [r for r in replies if r.ok]
    tokens = sum(r.completion_tokens or 0 for r in ok)
    prompt_tokens = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    inputs = sum(prompt_tokens)
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [t for r in ok if (t := r.tpot_s) is not None]
    e2e = [r.duration_s for r in ok]
    tpot_p50 = percentile(tpot, 50)
    errors = [r.error for r in replies if not r.ok]
    return {
        "concurrency": concurrency,
        "requests": requests,
        "ok": len(ok),
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "wall_s": round(wall, 2),
        "req_per_s": round(len(ok) / wall, 3) if wall else None,
        "output_tok_per_s": round(tokens / wall, 1) if wall else None,
        "input_tok_per_s": round(inputs / wall, 1) if wall and prompt_tokens else None,
        "total_tok_per_s": round((inputs + tokens) / wall, 1) if wall and prompt_tokens else None,
        "prompt_tokens_mean": round(inputs / len(prompt_tokens)) if prompt_tokens else None,
        "per_request_tok_per_s": round(1 / tpot_p50, 1) if tpot_p50 else None,
        "ttft_ms_p50": _ms(percentile(ttft, 50)),
        "ttft_ms_p90": _ms(percentile(ttft, 90)),
        "ttft_ms_p99": _ms(percentile(ttft, 99)),
        "tpot_ms_p50": _ms(percentile(tpot, 50)),
        "tpot_ms_p90": _ms(percentile(tpot, 90)),
        "latency_s_p50": round(percentile(e2e, 50) or 0, 2) if e2e else None,
        "max_stall_ms": _ms(max((r.max_gap_s or 0) for r in ok)) if ok else None,
        "completion_tokens": tokens,
    }
