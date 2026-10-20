"""Model weights on the machine: cache layouts, completeness, preset choice, shared downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hwb.serve import presets, weights

PRO = presets.Gpu("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 95.6, "12.0")


def hf_snapshot(
    hub: Path, model_id: str, rev: str, shards: dict[str, int], *, missing: tuple[str, ...] = ()
) -> Path:
    """The real HF cache layout: snapshots/<rev>/<file> are links into blobs/<sha>."""
    repo = hub / ("models--" + model_id.replace("/", "--"))
    snap = repo / "snapshots" / rev
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir(exist_ok=True)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(rev)
    config = json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}).encode()
    files = {"config.json": config, **{name: b"\0" * size for name, size in shards.items()}}
    if len(shards) > 1:
        files["model.safetensors.index.json"] = json.dumps(
            {"weight_map": {f"w{i}": n for i, n in enumerate(shards)}}
        ).encode()
    for name, data in files.items():
        blob = repo / "blobs" / hashlib.sha256(name.encode() + data).hexdigest()
        if name not in missing:
            blob.write_bytes(data)
        (snap / name).symlink_to(blob)  # dangling when "missing": an interrupted download
    return snap


@pytest.fixture
def machine(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    hub = root / "home" / "alice" / ".cache" / "huggingface" / "hub"
    hf_snapshot(
        hub,
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "a1b2c3",
        {"model-00001-of-00002.safetensors": 3000, "model-00002-of-00002.safetensors": 2000},
    )
    hf_snapshot(hub, "zai-org/GLM-4.7-Flash", "d4e5f6", {"model-00001-of-00002.safetensors": 3000, "model-00002-of-00002.safetensors": 2000},
                missing=("model-00002-of-00002.safetensors",))  # fmt: skip
    gguf = hub / "models--unsloth--Foo-GGUF" / "snapshots" / "r1"
    gguf.mkdir(parents=True)
    (gguf / "foo-Q4_K_M.gguf").write_bytes(b"\0" * 1000)
    ms = (
        root
        / "home"
        / "alice"
        / ".cache"
        / "modelscope"
        / "hub"
        / "models"
        / "deepseek-ai"
        / "DeepSeek-V4-Flash-0731"
    )
    ms.mkdir(parents=True)
    (ms / "config.json").write_text("{}")
    (ms / "model.safetensors").write_bytes(b"\0" * 4000)
    local = root / "models" / "my-finetune"
    local.mkdir(parents=True)
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"\0" * 1000)
    man = root / "usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/llama3.3/70b"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"layers": [{"size": 42_000_000_000}, {"size": 1000}]}))
    return root


def test_finds_weights_from_every_tool_and_judges_completeness(machine: Path) -> None:
    found = {(w.id, w.source): w for w in weights.find_all(machine)}
    assert set(found) == {
        ("Qwen/Qwen3.6-35B-A3B-FP8", "huggingface"),
        ("zai-org/GLM-4.7-Flash", "huggingface"),
        ("unsloth/Foo-GGUF", "huggingface"),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "modelscope"),
        ("my-finetune", "local"),
        ("llama3.3:70b", "ollama"),
    }
    assert found["Qwen/Qwen3.6-35B-A3B-FP8", "huggingface"].vllm_runnable
    assert found["Qwen/Qwen3.6-35B-A3B-FP8", "huggingface"].revision == "a1b2c3"
    assert (
        found["Qwen/Qwen3.6-35B-A3B-FP8", "huggingface"].architecture
        == "Qwen3_5MoeForConditionalGeneration"
    )
    assert not found["zai-org/GLM-4.7-Flash", "huggingface"].complete  # a shard never arrived
    assert found["unsloth/Foo-GGUF", "huggingface"].format == "gguf"
    assert found["llama3.3:70b", "ollama"].size_gb == pytest.approx(42.0, abs=0.01)
    report = json.dumps([w.to_dict() for w in found.values()])
    assert "alice" not in report and "/home" not in report  # no paths, no user names


def test_default_hub_follows_hugging_face_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert weights.default_hub() == tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub2"))
    assert weights.default_hub() == tmp_path / "hub2"


def test_preset_choice_prefers_weights_already_on_disk(machine: Path) -> None:
    from hwb import serve

    hw = presets.matching_hardware([PRO, PRO])
    found = weights.find_all(machine)
    preset, why = serve._choose_preset(None, hw, found)
    assert preset is not None and preset.name == "deepseek-v4-flash"  # ModelScope copy on disk
    assert "already on this machine" in why
    # GLM-4.7-Flash is only partly downloaded: never chosen as "on disk".
    assert weights.find(found, "zai-org/GLM-4.7-Flash") is None
    preset, why = serve._choose_preset(None, hw, [])
    assert preset is not None and preset.default and "will be downloaded" in why


def test_weights_plan_reuses_any_cache_and_checks_disk(
    machine: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hwb import serve

    found = weights.find_all(machine)
    qwen = presets.by_name("qwen3.6-35b-a3b-fp8")
    assert qwen is not None
    # In alice's cache, but our default cache is elsewhere: point vLLM at hers, no download.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "my-hub"))
    model_arg, env, plan, problem = serve._weights_plan(qwen, found, force=False)
    assert problem is None and model_arg is None and plan["cached"]
    assert env == {
        "HF_HUB_CACHE": str(machine / "home/alice/.cache/huggingface/hub")
    }  # no tokenizer: stay online
    snap = (
        machine
        / "home/alice/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/a1b2c3"
    )
    (snap / "tokenizer.json").write_text("{}")
    assert (
        serve._weights_plan(qwen, found, force=False)[1]["HF_HUB_OFFLINE"] == "1"
    )  # all on disk: offline
    # The ModelScope copy is used by path.
    ds = presets.by_name("deepseek-v4-flash")
    assert ds is not None
    model_arg, env, plan, problem = serve._weights_plan(ds, found, force=False)
    assert (
        model_arg
        and model_arg.endswith("DeepSeek-V4-Flash-0731")
        and plan["source"] == "modelscope"
    )
    # Nothing on disk and too little space: refuse before a long, doomed download.
    glm = presets.by_name("glm-4.5-air-fp8")
    assert glm is not None
    monkeypatch.setattr(weights, "free_gb", lambda path: 50.0)
    _, _, plan, problem = serve._weights_plan(glm, found, force=False)
    assert problem and "50 GB free" in problem[0] and "HF_HOME" in problem[1]
    assert serve._weights_plan(glm, found, force=True)[3] is None


def test_auto_preset_sizes_tensor_parallel(machine: Path) -> None:
    from hwb import serve

    big = weights.Weights("org/big", "huggingface", machine, 168.0, "safetensors", True, "r", None)
    small = weights.Weights(
        "org/small", "huggingface", machine, 30.0, "safetensors", True, "r", None
    )
    gguf = weights.Weights("org/q", "huggingface", machine, 30.0, "gguf", True, "r", None)
    p, tp, _ = serve._auto_preset("org/big", [big], [PRO, PRO], ["0", "1"])
    assert p is not None and tp == 2 and p.args[:2] == ("--tensor-parallel-size", "2")
    assert serve._auto_preset("org/small", [small], [PRO, PRO], ["0", "1"])[1] == 1
    assert serve._auto_preset("org/big", [big], [PRO, PRO], ["1"])[0] is None  # one GPU free
    p, _, why = serve._auto_preset("org/q", [gguf], [PRO], ["0"])
    assert p is None and "--endpoint" in why
    assert "hf download org/none" in serve._auto_preset("org/none", [], [PRO], ["0"])[2]


def test_serve_model_from_a_local_directory(
    machine: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hwb import cli, serve

    monkeypatch.setattr(serve.workloads, "gather", lambda: {"servers": [], "processes": []})
    monkeypatch.setattr(serve, "detected_gpus", lambda: [PRO, PRO])
    monkeypatch.setattr(weights, "ROOT", machine)
    local = machine / "models" / "my-finetune"
    assert (
        cli.main(["--out", str(tmp_path / "runs"), "serve", "--model", str(local), "--dry-run"])
        == 0
    )
    out = capsys.readouterr().out
    assert (
        f"vllm serve {local} --served-model-name my-finetune" in out
        and "--tensor-parallel-size 1" in out
    )
    assert "weights: local directory" in out
