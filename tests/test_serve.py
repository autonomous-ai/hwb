"""hwb serve: presets, the streaming client, answer checks, load stats, launch and teardown."""

from __future__ import annotations

import json
import os
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

from hwb import cli, serve
from hwb.core.errors import HwbError
from hwb.serve import client, container, presets, suite

# A minimal vLLM: answers the check prompts, streams forced-length output, reports usage.
FAKE_SERVER = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    MODE = {"wrong": False, "no_tools": False}

    def answer(body):
        prompt = body["messages"][-1]["content"]
        if body.get("tools") and not MODE["no_tools"]:
            return None, [{"name": "get_weather", "arguments": '{"city": "Chicago"}'}]
        if MODE["wrong"]:
            return "I am not sure.", []
        if "17 * 23" in prompt: return "391", []
        if "capital of Texas" in prompt: return '{"city": "Austin", "state": "Texas"}', []
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
            elif self.path == "/v1/models": self._json(200, {"data": [{"id": "fake-model", "max_model_len": 20000}]})
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
            prompt_tokens = 10 + len(body["messages"][-1]["content"].split())  # one token per word
            send({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max(len(pieces), len(calls))}})
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
    "inkling",
    "qwen3",
    "qwen3_coder",
    "qwen3_xml",
    "qwen3_json",
}  # in vLLM


def test_presets_are_consistent() -> None:
    names = [p.name for p in presets.PRESETS]
    assert len(names) == len(set(names))
    for hw in presets.HARDWARE.values():
        own = [p for p in presets.for_hardware(hw) if p.default and p.hardware == hw.id]
        # a profile may have no verified preset yet; a default must be verified
        assert len(own) <= 1 and all(p.status == presets.VERIFIED for p in own)
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
        if hw.vendor == "amd":  # no ROCm vLLM venv: AMD presets are container templates
            assert p.container is not None
        if p.container and p.container.own_command:
            assert not p.args and not p.tools


def test_deepseek_preset_is_the_production_command() -> None:
    p = presets.by_name("deepseek-v4-flash-vision")
    assert p is not None
    cmd = p.command()
    assert cmd[cmd.index("--revision") + 1] == "6821d6ad3681a4b137b066b76094fa82ebd0a380"
    assert "--attention-backend" in cmd and "B12X" in cmd and p.vllm_extras == ("b12x",)
    # vllm 0.32: b12x linear layers leave wo_a 2-D and DeepGEMM's o_proj einsum asserts
    assert "--linear-backend" not in cmd


def test_hardware_matching() -> None:
    pro = presets.Gpu("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 95.6, "12.0")
    a6000 = presets.Gpu("NVIDIA RTX A6000", 47.5, "8.6")
    hw = presets.matching_hardware([pro, pro])
    assert hw is not None and hw.id == "2x-rtx-pro-6000-blackwell"
    one = presets.matching_hardware([pro])
    assert one is not None and one.id == "1x-rtx-pro-6000-blackwell"
    assert presets.matching_hardware([a6000, a6000]) is None  # the "A6000" naming trap
    # a 2-card rig runs the 1-card presets too, its own first; one card does not run 2-card ones
    names = [p.name for p in presets.for_hardware(hw)]
    assert names[0] == "deepseek-v4-flash-vision" and "inkling-2-mini" in names
    assert [p.name for p in presets.for_hardware(one)] == ["inkling-2-mini"]
    r9700 = presets.Gpu("AMD gfx1201 (device 0x7551)", 31.9, "gfx1201")
    rx9070 = presets.Gpu("AMD gfx1201 (device 0x7550)", 15.9, "gfx1201")
    amd = presets.matching_hardware([r9700])
    assert amd is not None and amd.vendor == "amd"
    assert {p.name for p in presets.for_hardware(amd)} == {
        "inkling-2-mini-r9700",
        "inkling-2-mini-r9700-w3",
    }
    assert presets.matching_hardware([rx9070]) is None  # same chip, 16 GB
    rtx5090 = presets.Gpu("NVIDIA GeForce RTX 5090", 31.8, "12.0")
    eight = presets.matching_hardware([rtx5090] * 8)
    assert eight is not None and eight.id == "1x-rtx-5090"
    assert [p.name for p in presets.for_hardware(eight)] == ["inkling-2-mini-5090"]


def test_choose_preset_without_a_default() -> None:
    one = presets.HARDWARE["1x-rtx-pro-6000-blackwell"]
    p, why = serve._choose_preset(None, one, [])
    assert p is not None and p.name == "inkling-2-mini" and "untested" in why
    two = presets.HARDWARE["2x-rtx-pro-6000-blackwell"]
    p, _ = serve._choose_preset(None, two, [])
    assert p is not None and p.default and p.hardware == two.id


def test_amd_gpus_from_kfd_topology(tmp_path: Path) -> None:
    cpu = tmp_path / "0"
    cpu.mkdir()
    (cpu / "properties").write_text("cpu_cores_count 32\ngfx_target_version 0\n")
    gpu = tmp_path / "1"
    (gpu / "mem_banks" / "0").mkdir(parents=True)
    (gpu / "properties").write_text("gfx_target_version 120001\ndevice_id 30033\n")
    (gpu / "mem_banks" / "0" / "properties").write_text("heap_type 1\nsize_in_bytes 34342961152\n")
    gpus = serve.amd_gpus(tmp_path)
    assert len(gpus) == 1
    assert gpus[0].compute_cap == "gfx1201" and gpus[0].name.endswith("0x7551)")
    assert 31.9 < gpus[0].mem_gib < 32.1
    assert serve.amd_gpus(tmp_path / "missing") == []


def _mounts(tmp_path: Path, readonly: bool = False) -> container.Mounts:
    return container.Mounts(tmp_path / "hub", readonly, tmp_path / "home", tmp_path / "cache")


def test_docker_command_nvidia(tmp_path: Path) -> None:
    p = presets.by_name("inkling-2-mini")
    assert p is not None and p.container is not None
    argv = container.docker_command(
        p, p.container, name="hwb-serve-x", port=8001, vendor="nvidia",
        mounts=_mounts(tmp_path), gpus=["1"], env={"HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "1"},
        user=(1000, 1000), environ={"HF_TOKEN": "secret", "NVIDIA_DISABLE_REQUIRE": "1"},
    )  # fmt: skip
    assert argv[:5] == ["docker", "run", "--rm", "--name", "hwb-serve-x"]
    assert argv[argv.index("-p") + 1] == "127.0.0.1:8001:8000"
    assert argv[argv.index("--gpus") + 1] == '"device=1"'
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert f"{tmp_path / 'hub'}:{container.HUB}" in argv and "HF_HUB_OFFLINE=1" in argv
    assert "secret" not in " ".join(argv) and "HF_TOKEN" in argv  # by name only
    assert not any(a.startswith("CUDA_VISIBLE_DEVICES") for a in argv)  # --gpus does that
    assert "NVIDIA_DISABLE_REQUIRE=1" in argv  # the operator's override reaches the toolkit
    tail = argv[argv.index(p.container.image) + 1 :]
    assert tail[:3] == [p.model, "--revision", p.revision]
    assert tail[tail.index("--port") + 1] == "8000" and tail[tail.index("--host") + 1] == "0.0.0.0"  # noqa: S104
    assert "--speculative-config" in tail


def test_docker_command_amd_own_command(tmp_path: Path) -> None:
    p = presets.by_name("inkling-2-mini-r9700")
    assert p is not None and p.container is not None
    argv = container.docker_command(
        p, p.container, name="n", port=8000, vendor="amd", mounts=_mounts(tmp_path),
        user=(1000, 1000), gids=[44, 992], environ={},
    )  # fmt: skip
    assert argv[-1] == p.container.image  # the image's own command, no vllm args
    assert argv[argv.index("/dev/kfd") - 1 : argv.index("/dev/kfd") + 1] == ["--device", "/dev/kfd"]
    assert "/dev/dri" in argv and "--gpus" not in argv
    assert "--user" not in argv and "--group-add" not in argv  # its launcher runs as root
    assert argv[argv.index("-p") + 1] == "127.0.0.1:8000:8042"
    assert f"{tmp_path / 'cache'}:/models/cache" in argv and "GPU_MAX_HW_QUEUES=1" in argv
    assert not any(container.HUB in a for a in argv)


def test_docker_command_plain_preset_and_local_dir(tmp_path: Path) -> None:
    p = presets.by_name("deepseek-v4-flash-vision")
    assert p is not None
    spec = presets.Container(presets.DEFAULT_IMAGE)
    argv = container.docker_command(
        p, spec, name="n", port=8000, vendor="nvidia", mounts=_mounts(tmp_path, readonly=True),
        model_arg="/models/ds", user=(1000, 1000), environ={},
    )  # fmt: skip
    assert argv[argv.index("--gpus") + 1] == "all"
    assert f"{tmp_path / 'hub'}:{container.HUB}:ro" in argv
    assert "/models/ds:/models/ds:ro" in argv
    assert argv[argv.index("--ulimit") + 1] == "memlock=-1:-1"  # instanttensor
    tail = argv[argv.index(presets.DEFAULT_IMAGE) + 1 :]
    assert tail[0] == "/models/ds" and "--revision" not in tail


def test_serve_dry_run_prints_docker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HWB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rc = cli.main(
        ["--out", str(tmp_path / "runs"), "serve", "--preset", "inkling-2-mini", "--force",
         "--dry-run"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0
    assert "docker command:" in out and "docker run --rm" in out
    assert '\'{"method":"dspark"' in out  # quoted so it pastes
    assert not (tmp_path / "home" / "docker").exists()  # a dry run creates nothing


def test_container_cache_is_shared_by_image_tags() -> None:
    a, b = presets.by_name("inkling-2-mini-r9700"), presets.by_name("inkling-2-mini-r9700-w3")
    assert a and b and a.container and b.container
    ca, cb = serve.container_cache(a, a.container), serve.container_cache(b, b.container)
    assert ca == cb and ca.name == "paiton-vllm-plugin--thinkingmachines--Inkling-2-Mini"


def test_gpus_option_picks_the_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HWB_HOME", str(tmp_path / "home"))
    r9700 = presets.Gpu("AMD gfx1201 (device 0x7551)", 31.9, "gfx1201")
    monkeypatch.setattr(serve, "detected_gpus", lambda: [r9700, r9700])
    base = ["serve", "--preset", "inkling-2-mini-r9700", "--dry-run"]
    assert cli.main(["--out", str(tmp_path / "a"), *base, "--gpus", "1"]) == 0
    assert "HIP_VISIBLE_DEVICES=1" in capsys.readouterr().out
    cli.main(["--out", str(tmp_path / "b"), *base, "--gpus", "2"])
    assert "no GPU 2" in "".join(capsys.readouterr())
    # a card another workload holds is refused: pick a free one
    monkeypatch.setattr(serve.workloads, "busy_gpus", lambda running: {"0": 9800.0})
    cli.main(["--out", str(tmp_path / "c"), *base, "--gpus", "0"])
    out = "".join(capsys.readouterr())
    assert "GPU 0 already in use" in out
    assert cli.main(["--out", str(tmp_path / "d"), *base, "--gpus", "1"]) == 0
    assert "HIP_VISIBLE_DEVICES=1" in capsys.readouterr().out


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
    assert t.tool_calls == [{"name": "get_weather", "arguments": '{"city": "Chicago"}'}]
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


def test_profiles_send_prompts_of_the_asked_length(endpoint: str, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    argv = ["--out", str(out), "serve", "--endpoint", endpoint, "--concurrency", "1",
            "--requests", "2", "--max-tokens", "8", "--profiles", "2k/16,agent"]  # fmt: skip
    assert cli.main(argv) == 0
    report = _report(out)
    st = {r["name"]: r["status"] for r in report["results"]}
    assert st["serve.profile.2k/16"] == "pass" and st["serve.load"] == "pass"
    assert st["serve.profile.agent"] == "skip"  # 32k + 1k does not fit the server's 20000
    sec = report["sections"]["bench"]["serve"]
    custom, agent = sec["profiles"]
    (lv,) = custom["levels"]
    assert abs(lv["prompt_tokens_mean"] - 2048) < 64 and lv["completion_tokens"] == 2 * 16
    assert lv["total_tok_per_s"] > lv["output_tok_per_s"] > 0
    assert "allows 20000" in agent["skipped"] and not agent["levels"]
    assert [p["name"] for p in sec["timeseries"]["phases"]][-1] == "2k/16 c=1"


def test_profile_specs() -> None:
    assert [p.name for p in suite.parse_profiles("chat, rag")] == ["chat", "rag"]
    assert [p.name for p in suite.parse_profiles("all")] == [p.name for p in suite.PROFILES]
    (custom,) = suite.parse_profiles("4k/512")
    assert (custom.input_tokens, custom.output_tokens) == (4096, 512)
    assert suite.parse_profiles(None) == ()
    for bad in ("huge", "0/10", "4k-512"):
        with pytest.raises(HwbError):
            suite.parse_profiles(bad)
    # every request reads different text, so prefix caching cannot shortcut the prefill
    assert suite.filler(1, 200) != suite.filler(2, 200)
    assert 190 <= len(suite.filler(1, 200).split()) <= 200


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


@pytest.mark.parametrize("preset", ["inkling-2-mini", "inkling-2-mini-r9700"])
def test_docker_launch_tests_and_removes_the_container(
    preset: str, fake_module: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fake docker: `run` serves the fake vLLM on the published host port, `rm` is recorded.
    calls = tmp_path / "docker-calls"
    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, runpy
        args = sys.argv[1:]
        with open({str(calls)!r}, "a") as f:
            f.write(" ".join(args) + "\\n")
        if args[0] == "run":
            port = args[args.index("-p") + 1].split(":")[1]
            sys.argv = ["fake", "--port", port]
            runpy.run_path({str(fake_module)!r}, run_name="__main__")
        """))  # fmt: skip
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("HWB_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    out = tmp_path / "runs"
    port = _free_port()
    argv = [
        "--out", str(out), "serve", "--preset", preset, "--force", "--port", str(port),
        "--concurrency", "2", "--requests", "1", "--max-tokens", "8", "--ready-timeout", "60",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    report = _report(out)
    st = {r["name"]: r["status"] for r in report["results"]}
    assert st["serve.vllm"] == "pass" and st["serve.load"] == "pass", st
    sec = report["sections"]["bench"]["serve"]
    assert sec["command"][:2] == ["docker", "run"] and sec["container"]["image"]
    lines = calls.read_text().splitlines()
    assert lines[0].startswith("run --rm --name hwb-serve-")
    name = lines[0].split()[3]
    assert lines[-1] == f"rm -f {name}"  # removed even though --rm should have done it
    if preset.endswith("r9700"):  # the image named its model: the tests used that name
        assert sec["model"] == "fake-model"
    else:
        assert (tmp_path / "home" / "docker" / "home").is_dir()  # HOME for kernel caches
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
    assert first["max_tokens"] == 32 + suite.REASONING_BUDGET


def test_a_check_retries_with_room_to_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    budgets: list[int] = []

    def chat(*a: object, max_tokens: int, **k: object) -> client.Reply:
        budgets.append(max_tokens)
        if max_tokens <= 32:
            return client.Reply(ok=True, reasoning="17 times 23 is", completion_tokens=32)
        return client.Reply(ok=True, reasoning="17*23", text="391", completion_tokens=200)

    monkeypatch.setattr(suite, "chat", chat)
    (first, *_) = suite.run_checks("http://x", "m", no_thinking={}, tools=False)
    assert budgets[:2] == [32, 32 + suite.REASONING_BUDGET]
    assert first["status"] == "pass" and "after reasoning" in first["detail"]


def test_old_driver_for_the_image_gets_a_hint(tmp_path: Path) -> None:
    from hwb import serve

    log = tmp_path / "vllm.log"
    log.write_text(
        "docker: Error response from daemon: failed to create task for container: ...\n"
        "nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.2, please "
        "update your driver to a newer version, or use an earlier cuda container\n"
    )
    assert "NVIDIA_DISABLE_REQUIRE=1" in serve._tail(log)


def test_failure_causes_skip_the_wrapping_tracebacks() -> None:
    from hwb import serve

    log = textwrap.dedent(
        """\
        /tmp/x/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
        (Worker_TP1 pid=1) ERROR [multiproc_executor.py:953] subprocess.CalledProcessError: Command '['/usr/bin/gcc']' returned 1
        (Worker_TP0 pid=2) ERROR [multiproc_executor.py:953] subprocess.CalledProcessError: Command '['/usr/bin/gcc']' returned 1
        (EngineCore pid=3) ERROR [core.py:1510] Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
        RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
        """
    )
    assert serve.failure_causes(log) == [
        "fatal error: Python.h: No such file or directory",
        "CalledProcessError: Command '['/usr/bin/gcc']' returned 1",
    ]


def test_venv_env_activates_the_venv_and_finds_its_pip_toolkit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hwb import serve
    from hwb.core import proc

    monkeypatch.setattr(proc, "which", lambda name: None)
    vllm = tmp_path / "venv" / "bin" / "vllm"
    vllm.parent.mkdir(parents=True)
    vllm.touch()
    (tmp_path / "venv" / "pyvenv.cfg").touch()
    bin_path = {"PATH": f"{vllm.parent.resolve()}:/usr/bin"}
    home = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "nvcc").touch()
    env = {"PATH": "/usr/bin"}
    assert serve.venv_env(str(vllm), env) == bin_path | {"CUDA_HOME": str(home.resolve())}
    assert serve.venv_env(str(vllm), env | {"CUDA_HOME": "/opt/cuda"}) == bin_path
    (home / "lib").mkdir()
    (home / "lib" / "libcudart.so.13").touch()
    serve.complete_pip_toolkit(home)
    assert (home / "lib64" / "libcudart.so").resolve() == (
        home / "lib" / "libcudart.so.13"
    ).resolve()
    (home / "bin" / "nvcc").unlink()
    assert serve.venv_env(str(vllm), env) == bin_path


def test_an_older_system_nvcc_does_not_shadow_the_venv_toolkit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ubuntu's nvidia-cuda-toolkit (nvcc 12.0) made FlashInfer's JIT find no Blackwell arch."""
    from hwb import serve
    from hwb.core import proc

    vllm = tmp_path / "venv" / "bin" / "vllm"
    vllm.parent.mkdir(parents=True)
    vllm.touch()
    home = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "nvcc").touch()
    monkeypatch.setattr(proc, "which", lambda name: "/usr/bin/nvcc")
    monkeypatch.setattr(serve, "nvcc_major", lambda nvcc: 12)
    assert serve.venv_env(str(vllm), {})["CUDA_HOME"] == str(home.resolve())
    monkeypatch.setattr(serve, "nvcc_major", lambda nvcc: 13)
    assert "CUDA_HOME" not in serve.venv_env(str(vllm), {})


def test_instanttensor_is_dropped_where_it_cannot_load() -> None:
    from hwb import serve

    assert serve.loader_problem(None, {"0": 16.0, "1": 16.0}) is None
    assert serve.loader_problem(8 << 20, {}) is not None  # Ubuntu's default 8 MiB
    held = serve.loader_problem(None, {"0": 985.0, "1": 16.0})
    assert held is not None and "GPU0 985 MiB" in held


def test_amd_container_sees_only_its_own_card(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for card, addr, render in (("card1", "0000:36:00.0", "renderD128"), ("card2", "0000:72:00.0", "renderD129")):  # fmt: skip
        dev = root / "sys/devices/pci0000:00" / addr
        (dev / "drm" / card).mkdir(parents=True)
        (dev / "drm" / render).mkdir()
        (dev / "vendor").write_text("0x1002")
        (dev / "mem_info_vram_total").write_text(str(32 << 30))
        link = root / "sys/bus/pci/devices" / addr
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(dev)
        drm = root / "sys/class/drm" / card
        drm.mkdir(parents=True)
        (drm / "device").symlink_to(dev)
    assert container.amd_device_nodes(["1"], root) == ["/dev/dri/card2", "/dev/dri/renderD129"]
    assert container.amd_device_nodes(["5"], root) is None
    p = presets.by_name("inkling-2-mini-r9700")
    assert p is not None and p.container is not None
    argv = container.docker_command(
        p, p.container, name="n", port=8001, vendor="amd", mounts=_mounts(tmp_path), gpus=["1"],
        environ={}, amd_nodes=container.amd_device_nodes(["1"], root),
    )  # fmt: skip
    devices = [argv[i + 1] for i, a in enumerate(argv) if a == "--device"]
    assert devices == ["/dev/kfd", "/dev/dri/card2", "/dev/dri/renderD129"]
    assert not any(a.startswith("HIP_VISIBLE_DEVICES") for a in argv)  # it is GPU 0 inside
