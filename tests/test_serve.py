"""hwb serve: presets, the streaming client, answer checks, load stats, launch and teardown."""

from __future__ import annotations

import json
import socket
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from hwb import cli
from hwb.serve import client, presets, suite

# A minimal vLLM: answers the check prompts, streams forced-length output, reports usage.
FAKE_SERVER = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    MODE = {"wrong": False, "no_tools": False}

    def answer(body):
        prompt = body["messages"][-1]["content"]
        if body.get("tools") and not MODE["no_tools"]:
            return None, [{"name": "get_weather", "arguments": '{"city": "Hanoi"}'}]
        if MODE["wrong"]:
            return "I am not sure.", []
        if "17 * 23" in prompt: return "391", []
        if "capital of France" in prompt: return '{"city": "Paris", "country": "France"}', []
        if "is_prime" in prompt: return "def is_prime(n):\\n    return n > 1 and all(n % d for d in range(2, n))", []
        if "vault code" in prompt: return "4817", []
        n = body.get("max_tokens", 16)
        return " ".join(["tok"] * n), []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _json(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            if self.path == "/health": self._json(200, {})
            elif self.path == "/v1/models": self._json(200, {"data": [{"id": "fake-model"}]})
            else: self._json(404, {"message": "no"})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text, calls = answer(body)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close"); self.end_headers()
            def send(obj): self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\\n\\n"); self.wfile.flush()
            send({"choices": [{"delta": {"reasoning_content": ""}}]})
            pieces = text.split(" ") if text else []
            for i, c in enumerate(calls):
                send({"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"name": c["name"], "arguments": ""}}]}}]})
                send({"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"arguments": c["arguments"]}}]}}]})
            for i, p in enumerate(pieces):
                time.sleep(0.001)
                send({"choices": [{"delta": {"content": p if i == 0 else " " + p}}]})
            send({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": max(len(pieces), len(calls))}})
            self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush()
            self.close_connection = True

    def serve(port, **mode):
        MODE.update(mode)
        ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

    if __name__ == "__main__":
        args = sys.argv[1:]
        port = int(args[args.index("--port") + 1]) if "--port" in args else 8000
        serve(port)
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def fake_module(tmp_path: Path) -> Path:
    path = tmp_path / "fake_vllm_server.py"
    path.write_text(FAKE_SERVER)
    return path


@pytest.fixture
def endpoint(fake_module: Path, request: pytest.FixtureRequest) -> Iterator[str]:
    sys.path.insert(0, str(fake_module.parent))
    try:
        import fake_vllm_server as fake  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(fake_module.parent))
    mode: dict[str, bool] = getattr(request, "param", {})
    fake.MODE.update({"wrong": False, "no_tools": False, **mode})
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), fake.H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        sys.modules.pop("fake_vllm_server", None)


# ---------------------------------------------------------------- presets


KNOWN_PARSERS = {
    "deepseek_v4",
    "glm45",
    "glm47",
    "qwen3",
    "qwen3_coder",
    "qwen3_xml",
    "qwen3_json",
}  # in vLLM


def test_presets_are_consistent() -> None:
    names = [p.name for p in presets.PRESETS]
    assert len(names) == len(set(names))
    for hw in presets.HARDWARE.values():
        defaults = [p for p in presets.for_hardware(hw) if p.default]
        assert len(defaults) == 1 and defaults[0].status == presets.VERIFIED
    for p in presets.PRESETS:
        hw = presets.HARDWARE[p.hardware]
        total_gb = hw.gpus * hw.min_mem_gib * 1.0737  # GiB -> GB
        assert p.weights_gb <= 0.9 * total_gb, f"{p.name} leaves no room for KV cache"
        args = list(p.args)
        for flag in ("--tool-call-parser", "--reasoning-parser"):
            if flag in args:
                assert args[args.index(flag) + 1] in KNOWN_PARSERS, (p.name, flag)
        assert ("--tool-call-parser" in args) == p.tools, p.name
        cmd = p.command(port=9000)
        assert cmd[:3] == ["vllm", "serve", p.model] and "--served-model-name" in cmd
        assert cmd[cmd.index("--port") + 1] == "9000"


def test_verified_deepseek_preset_is_the_production_command() -> None:
    p = presets.by_name("deepseek-v4-flash-vision")
    assert p is not None and p.status == presets.VERIFIED
    cmd = p.command()
    assert cmd[cmd.index("--revision") + 1] == "6821d6ad3681a4b137b066b76094fa82ebd0a380"
    assert "--attention-backend" in cmd and "B12X" in cmd and p.vllm_extras == ("b12x",)


def test_hardware_matching() -> None:
    pro = presets.Gpu("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 95.6, "12.0")
    a6000 = presets.Gpu("NVIDIA RTX A6000", 47.5, "8.6")
    hw = presets.matching_hardware([pro, pro])
    assert hw is not None and hw.id == "2x-rtx-pro-6000-blackwell"
    assert presets.matching_hardware([pro]) is None  # one card is not enough
    assert presets.matching_hardware([a6000, a6000]) is None  # the "A6000" naming trap


# ---------------------------------------------------------------- client and suite


def test_percentile() -> None:
    assert client.percentile([], 50) is None
    assert client.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert client.percentile([5.0], 99) == 5.0


def test_streaming_client_measures_and_reassembles(endpoint: str) -> None:
    r = client.chat(
        endpoint, "fake-model", [{"role": "user", "content": "x"}], max_tokens=20, ignore_eos=True
    )
    assert r.ok and r.completion_tokens == 20 and r.text.count("tok") == 20
    assert r.ttft_s is not None and r.ttft_s < r.duration_s and r.tpot_s is not None
    t = client.chat(
        endpoint,
        "fake-model",
        [{"role": "user", "content": "w"}],
        max_tokens=20,
        tools=[suite.WEATHER_TOOL],
    )
    assert t.tool_calls == [{"name": "get_weather", "arguments": '{"city": "Hanoi"}'}]
    bad = client.chat("http://127.0.0.1:9", "m", [], max_tokens=1, timeout=2)
    assert not bad.ok and bad.error


def test_checks_pass_on_a_correct_server(endpoint: str) -> None:
    results = suite.run_checks(
        endpoint, "fake-model", no_thinking={"enable_thinking": False}, tools=True
    )
    assert {c["name"]: c["status"] for c in results} == {
        "arithmetic": "pass",
        "json": "pass",
        "code": "pass",
        "long_context": "pass",
        "tool_call": "pass",
    }


@pytest.mark.parametrize("endpoint", [{"wrong": True, "no_tools": True}], indirect=True)
def test_checks_catch_wrong_answers_and_a_missing_tool_parser(endpoint: str) -> None:
    results = {
        c["name"]: c for c in suite.run_checks(endpoint, "fake-model", no_thinking={}, tools=True)
    }
    assert all(c["status"] == "fail" for c in results.values())
    assert "tool-call-parser" in results["tool_call"]["detail"]
    skipped = suite.run_checks(endpoint, "fake-model", no_thinking={}, tools=False)
    assert [c["status"] for c in skipped if c["name"] == "tool_call"] == ["skip"]


def test_load_level_stats(endpoint: str) -> None:
    lv = suite.run_level(
        endpoint, "fake-model", concurrency=4, requests=8, max_tokens=32, no_thinking={}
    )
    assert (lv["ok"], lv["errors"], lv["completion_tokens"]) == (8, 0, 8 * 32)
    assert (
        lv["output_tok_per_s"] > 0
        and lv["ttft_ms_p50"] is not None
        and lv["tpot_ms_p50"] is not None
    )


# ---------------------------------------------------------------- end to end


def _report(out: Path) -> dict[str, Any]:
    (run_dir,) = (p for p in out.iterdir() if p.is_dir())
    report: dict[str, Any] = json.loads((run_dir / "report.json").read_text())
    return report


def test_endpoint_mode_tests_a_running_server(endpoint: str, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    argv = [
        "--out",
        str(out),
        "serve",
        "--endpoint",
        endpoint,
        "--concurrency",
        "1,2",
        "--requests",
        "2",
        "--max-tokens",
        "16",
    ]
    assert cli.main(argv) == 0
    report = _report(out)
    st = {r["name"]: r["status"] for r in report["results"]}
    assert st == {"serve.checks": "pass", "serve.load": "pass"}
    sec = report["sections"]["bench"]["serve"]
    assert sec["mode"] == "endpoint" and sec["model"] == "fake-model"
    assert [lv["concurrency"] for lv in sec["load"]] == [1, 2] and sec["load"][1]["requests"] == 4
    assert [p["name"] for p in sec["timeseries"]["phases"]] == [
        "checks",
        "c=1",
        "c=2",
    ]  # local: sampled


def test_launch_mode_starts_tests_and_stops_vllm(fake_module: Path, tmp_path: Path) -> None:
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text(f'#!/bin/sh\nexec {sys.executable} {fake_module} "$@"\n')
    fake_vllm.chmod(0o755)
    out = tmp_path / "runs"
    port = _free_port()
    argv = [
        "--out", str(out), "serve", "--preset", "qwen3.6-35b-a3b-fp8", "--force", "--vllm", str(fake_vllm),
        "--port", str(port), "--concurrency", "2", "--requests", "1", "--max-tokens", "8", "--ready-timeout", "60",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    report = _report(out)
    st = {r["name"]: r["status"] for r in report["results"]}
    assert st == {"serve.vllm": "pass", "serve.checks": "pass", "serve.load": "pass"}
    sec = report["sections"]["bench"]["serve"]
    assert sec["command"][:3] == [str(fake_vllm), "serve", "Qwen/Qwen3.6-35B-A3B-FP8"]
    # Torn down: nothing listens on the port any more.
    time.sleep(0.2)
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0


def test_dry_run_and_no_preset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a"
    assert (
        cli.main(["--out", str(out), "serve", "--preset", "glm-4.7-flash", "--force", "--dry-run"])
        == 0
    )
    assert "vllm serve zai-org/GLM-4.7-Flash" in capsys.readouterr().out
    assert {r["name"]: r["status"] for r in _report(out)["results"]} == {"serve.vllm": "skip"}
    out = tmp_path / "b"
    assert cli.main(["--out", str(out), "serve"]) == 0  # no matching hardware on the test machine
    (r,) = _report(out)["results"]
    assert r["status"] == "skip" and "--endpoint" in r["hint"]


def test_running_models_are_recognised_by_their_served_name() -> None:
    assert presets.match_model("DeepSeek-V4-Flash-Vision-Exp").name == "deepseek-v4-flash-vision"  # type: ignore[union-attr]
    assert (
        presets.match_model("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp").name
        == "deepseek-v4-flash-vision"
    )  # type: ignore[union-attr]
    assert presets.match_model("glm-4.7-flash").name == "glm-4.7-flash"  # type: ignore[union-attr]
    assert presets.match_model("some-finetune") is None


def test_reasoning_that_eats_the_answer_budget_is_explained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking_only = client.Reply(
        ok=True, reasoning="Let me think step by step about 17 times 23", completion_tokens=32
    )
    monkeypatch.setattr(suite, "chat", lambda *a, **k: thinking_only)
    (first, *_) = suite.run_checks("http://x", "m", no_thinking=suite.NO_THINKING_ANY, tools=False)
    assert first["status"] == "fail" and "budget went to reasoning" in first["detail"]
