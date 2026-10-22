"""Models already running on the GPUs: process walk, framework parsing, privacy, serve's reaction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hwb.metrics import workloads as wl

SMI_APPS = (
    "1240, /usr/bin/python3.12, 92160, 00000000:01:00.0\n"
    "1241, /usr/bin/python3.12, 92160, 00000000:21:00.0\n"
    "777, /opt/game/bin/render, 300, 00000000:21:00.0\n"
)


def _proc(root: Path, pid: int, argv: list[str], ppid: int) -> None:
    d = root / "proc" / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_text("\0".join(argv) + "\0")
    (d / "stat").write_text(f"{pid} (VLLM::Worker (TP0)) S {ppid} 1 1 0 -1\n")


def test_parse_apps_keeps_only_the_binary_name() -> None:
    rows = wl.parse_apps(SMI_APPS + "garbage\n")
    assert [r["process"] for r in rows] == ["python3.12", "python3.12", "render"]
    assert rows[0] == {
        "pid": 1240,
        "process": "python3.12",
        "used_mib": 92160,
        "bus_id": "00000000:01:00.0",
    }


@pytest.mark.parametrize(
    ("argv", "expect"),
    [
        (["/venv/bin/python3", "/venv/bin/vllm", "serve", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
          "--served-model-name", "dsv4", "--port", "8001", "--tensor-parallel-size", "2"],
         ("vllm", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "dsv4", 8001, 2)),
        (["python", "-m", "vllm.entrypoints.openai.api_server", "--model=Qwen/Qwen3.6-35B-A3B-FP8"],
         ("vllm", "Qwen/Qwen3.6-35B-A3B-FP8", None, 8000, None)),
        (["python3", "-m", "sglang.launch_server", "--model-path", "zai-org/GLM-4.5-Air-FP8", "--tp", "2"],
         ("sglang", "zai-org/GLM-4.5-Air-FP8", None, 30000, 2)),
        (["/usr/local/bin/llama-server", "-m", "/home/alice/models/glm-4.7-flash-Q4_K_M.gguf", "--port", "9000"],
         ("llama.cpp", "glm-4.7-flash-Q4_K_M.gguf", None, 9000, None)),  # path shrunk: no user name
    ],
)  # fmt: skip
def test_frameworks_and_flags(argv: list[str], expect: tuple[Any, ...]) -> None:
    fw = wl.identify(argv)
    assert fw is not None
    d = wl.describe(argv, fw)
    assert (d["framework"], d["model"], d["served_name"], d["port"], d["tensor_parallel"]) == expect


def test_unrelated_processes_are_not_servers() -> None:
    assert wl.identify(["/opt/game/bin/render", "--vllm-like"]) is None
    assert wl.identify(["python", "train.py", "--model", "x"]) is None


def test_gather_walks_workers_to_their_server_and_never_stores_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ["/venv/bin/python3", "/venv/bin/vllm", "serve", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
              "--tensor-parallel-size", "2", "--port", "8000", "--api-key", "sk-SECRET-123"]  # fmt: skip
    _proc(tmp_path, 1234, server, 1)
    for worker in (1240, 1241):
        _proc(
            tmp_path,
            worker,
            ["python3", "-c", "from multiprocessing.spawn import spawn_main"],
            1234,
        )
    _proc(tmp_path, 777, ["/opt/game/bin/render"], 1)
    monkeypatch.setattr(wl, "compute_apps", lambda: wl.parse_apps(SMI_APPS))
    monkeypatch.setattr(
        wl.nvsmi,
        "query_gpus",
        lambda fields: [
            {"index": "0", "pci.bus_id": "00000000:01:00.0"},
            {"index": "1", "pci.bus_id": "00000000:21:00.0"},
        ],
    )
    monkeypatch.setattr(
        wl,
        "probe_models",
        lambda port: {"ids": ["DeepSeek-V4-Flash-Vision-Exp"], "max_model_len": 262144},
    )

    data = wl.gather(tmp_path)
    (srv,) = data["servers"]
    assert srv["framework"] == "vllm" and srv["gpus"] == ["0", "1"] and srv["processes"] == 2
    assert srv["used_gib"] == 180.0 and srv["endpoint"] == "http://127.0.0.1:8000"
    assert srv["serving"]["max_model_len"] == 262144
    assert "sk-SECRET" not in json.dumps(data)  # command lines are parsed, never stored
    assert wl.busy_gpus(data) == {"0": 92160, "1": 92160 + 300}  # all memory held per GPU
    assert wl.busy_gpus({"processes": [{"gpu": "3", "used_mib": 300}]}) == {}  # small: not busy
    assert wl.summarize(data).startswith(
        "vllm deepseek-ai/DeepSeek-V4-Flash-Vision-Exp on GPU 0,1 (180.0 GiB"
    )
    assert wl.summarize({"servers": [], "processes": []}) == "GPUs idle: no compute processes"


def test_serve_tests_the_running_server_instead_of_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hwb import cli
    from hwb import serve as serve_mod

    seen: dict[str, Any] = {}
    running = {
        "servers": [{"framework": "vllm", "model": "m", "served_name": "dsv4", "port": 8000, "gpus": ["0", "1"],
                     "used_gib": 180.0, "processes": 2, "endpoint": "http://127.0.0.1:8000",
                     "serving": {"ids": ["dsv4"], "max_model_len": None}}],
        "processes": [{"gpu": "0", "pid": 1, "process": "python3", "used_mib": 92160}],
    }  # fmt: skip
    monkeypatch.setattr(serve_mod.workloads, "gather", lambda: running)
    monkeypatch.setattr(
        serve_mod,
        "_run_endpoint",
        lambda ctx, base, model, opts, section: seen.update(base=base, model=model),
    )
    assert cli.main(["--out", str(tmp_path / "a"), "serve"]) == 0
    assert seen == {"base": "http://127.0.0.1:8000", "model": "dsv4"}

    # A preset was asked for, but the GPUs are taken: refuse, and point at the running server.
    assert (
        cli.main(
            [
                "--out",
                str(tmp_path / "b"),
                "serve",
                "--preset",
                "glm-4.7-flash",
                "--force",
                "--dry-run",
            ]
        )
        == 0
    )
    running["servers"][0]["serving"] = None  # not answering: nothing to test, GPUs still held
    two_pro = [
        serve_mod.presets.Gpu("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 95.6, "12.0")
    ] * 2
    monkeypatch.setattr(serve_mod, "detected_gpus", lambda: two_pro)
    assert cli.main(["--out", str(tmp_path / "c"), "serve", "--preset", "glm-4.7-flash"]) == 0
    (run_dir,) = (p for p in (tmp_path / "c").iterdir() if p.is_dir())
    (r,) = json.loads((run_dir / "report.json").read_text())["results"]
    assert r["status"] == "skip" and "needs 2 free GPUs; in use: GPU0 90 GiB" in r["summary"]


def test_serve_launches_on_the_free_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from hwb import cli
    from hwb import serve as serve_mod

    # hwd02-like: someone's job holds GPU4; a 2-GPU preset still fits on GPUs 0 and 1.
    running = {
        "servers": [],
        "processes": [{"gpu": "4", "pid": 36052, "process": "python", "used_mib": 7782}],
    }
    monkeypatch.setattr(serve_mod.workloads, "gather", lambda: running)
    monkeypatch.setattr(
        serve_mod,
        "detected_gpus",
        lambda: [serve_mod.presets.Gpu("NVIDIA GeForce RTX 4090 D", 24, "8.9")] * 5,
    )
    assert (
        cli.main(
            ["--out", str(tmp_path), "serve", "--preset", "glm-4.7-flash", "--force", "--dry-run"]
        )
        == 0
    )
    assert (
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 VLLM_SKIP_P2P_CHECK=0 vllm serve zai-org/GLM-4.7-Flash"
        in capsys.readouterr().out
    )
