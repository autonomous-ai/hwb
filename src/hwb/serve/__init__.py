"""`hwb serve`: host a real model with vLLM (or test a running server) and measure it.

Two modes:
- launch: pick a preset for the detected GPUs (presets.py), start ``vllm serve`` in its own
  process group, wait for /health, test, and always tear it down;
- endpoint: test a server that is already running (``--endpoint http://host:8000``).

Either way: answer checks first (suite.CHECKS), then a load sweep over concurrency levels,
with the sampler recording GPU/CPU temperature, power, clocks and PCIe links throughout.
vLLM itself lives in an isolated uv venv under $HWB_HOME (``--install-vllm``) or comes from
PATH / ``--vllm``; it never touches the core environment.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import nvsmi
from ..core import proc
from ..core.context import RunContext
from ..core.errors import HwbError
from ..core.result import CheckResult, Status
from ..metrics import workloads
from ..metrics.sampler import Sampler
from . import client, presets, suite

log = logging.getLogger(__name__)

ENV_API_KEY = "HWB_LLM_API_KEY"
READY_POLL_S = 5.0
LOG_TAIL = 3000
_LOCAL = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class ServeOptions:
    preset: str | None = None
    endpoint: str | None = None
    model: str | None = None  # served model name, for --endpoint
    port: int = 8000
    ready_timeout_s: int = 3600
    concurrency: tuple[int, ...] = (1, 4, 16)
    requests_per_level: int = 4  # x concurrency
    max_tokens: int = 256
    interval_s: float = 1.0
    vllm: str | None = None
    install_vllm: bool = False
    dry_run: bool = False
    force: bool = False


def hwb_home() -> Path:
    return Path(os.environ.get("HWB_HOME") or Path.home() / ".cache" / "hwb")


def detected_gpus() -> list[presets.Gpu]:
    try:
        rows = nvsmi.query_gpus(("name", "memory.total", "compute_cap"))
    except HwbError:
        return []
    return [
        presets.Gpu(
            str(r.get("name") or ""),
            float(r.get("memory.total") or 0) / 1024,
            str(r.get("compute_cap") or ""),
        )
        for r in rows
    ]


def vllm_venv() -> Path:
    return hwb_home() / "venvs" / "vllm"


def find_vllm(explicit: str | None) -> str | None:
    if explicit:
        return explicit if proc.which(explicit) or Path(explicit).exists() else None
    venv = vllm_venv() / "bin" / "vllm"
    if venv.exists():
        return str(venv)
    return proc.which("vllm")


def install_vllm(preset: presets.Preset) -> str:
    """vLLM into an isolated uv venv under $HWB_HOME (never the system or core Python)."""
    if not proc.have("uv"):
        raise HwbError("--install-vllm needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    venv = vllm_venv()
    spec = presets.VLLM_SPEC
    if preset.vllm_extras:
        name, _, version = spec.partition("==")
        spec = f"{name}[{','.join(preset.vllm_extras)}]=={version}"
    log.info("installing %s into %s (several GB; one time)", spec, venv)
    for argv in (
        ["uv", "venv", str(venv), "--python", "3.12", "--allow-existing"],
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), spec],
    ):
        res = proc.run(argv, timeout=3600)
        if not res.ok:
            raise HwbError(f"{' '.join(argv[:3])} failed: {(res.stderr or res.stdout)[-500:]}")
    return str(venv / "bin" / "vllm")


def _tail(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-LOG_TAIL:]
    except OSError:
        return ""


def wait_ready(base: str, child: proc.Child, log_path: Path, timeout_s: float) -> float:
    """Seconds until /health answered; raises with the log tail if vLLM died or timed out."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if child.returncode is not None:
            raise HwbError(
                f"vllm exited with {child.returncode} before it was ready:\n{_tail(log_path)}"
            )
        if client.is_ready(base):
            return time.monotonic() - t0
        child.wait(READY_POLL_S)
    raise HwbError(f"vllm not ready after {timeout_s:.0f}s:\n{_tail(log_path)}")


def _summary(level: dict[str, Any]) -> str:
    parts = [f"{level['output_tok_per_s']} tok/s at {level['concurrency']} concurrent"]
    if level.get("ttft_ms_p50") is not None:
        parts.append(f"TTFT p50 {level['ttft_ms_p50']:.0f} ms")
    if level.get("per_request_tok_per_s"):
        parts.append(f"{level['per_request_tok_per_s']} tok/s per request")
    return ", ".join(parts)


def _test(
    ctx: RunContext,
    base: str,
    model: str,
    preset: presets.Preset | None,
    opts: ServeOptions,
    section: dict[str, Any],
    sample: bool,
) -> None:
    api_key = os.environ.get(ENV_API_KEY)
    no_thinking = preset.no_thinking if preset else suite.NO_THINKING_ANY
    tools = preset.tools if preset else True
    with Sampler(ctx.run_dir, name="serve", interval=opts.interval_s, gpus=sample) as s:
        s.phase("checks")
        checks = suite.run_checks(
            base, model, no_thinking=no_thinking, tools=tools, api_key=api_key
        )
        levels = []
        offset = 0
        for c in opts.concurrency:
            s.phase(f"c={c}")
            n = max(c * opts.requests_per_level, 1)
            levels.append(
                suite.run_level(
                    base, model, concurrency=c, requests=n, max_tokens=opts.max_tokens,
                    no_thinking=no_thinking, api_key=api_key, offset=offset,
                )
            )  # fmt: skip
            offset += n
    section |= {"checks": checks, "load": levels}
    if sample:
        section["timeseries"] = s.result.to_report()

    failed = [c for c in checks if c["status"] == "fail"]
    passed = [c for c in checks if c["status"] == "pass"]
    ctx.add(
        CheckResult(
            "serve.checks",
            Status.FAIL if failed else Status.PASS,
            (f"{len(failed)} of {len(passed) + len(failed)} answer checks failed: "
             + "; ".join(f"{c['name']}: {c['detail']}" for c in failed))
            if failed else f"{len(passed)} answer checks passed ({', '.join(c['name'] for c in passed)})",
            data={"checks": checks},
            hint="wrong answers from a healthy server usually mean the chat template, the tool/reasoning "
            "parser or the quantization: compare with the model card" if failed else None,
        )
    )  # fmt: skip
    ok_levels = [lv for lv in levels if lv["ok"]]
    errors = sum(lv["errors"] for lv in levels)
    if not ok_levels:
        status, text = (
            Status.FAIL,
            f"every load request failed: {levels[0]['first_error'] if levels else '?'}",
        )
    else:
        best = max(ok_levels, key=lambda lv: lv["output_tok_per_s"] or 0)
        status = Status.WARN if errors else Status.PASS
        text = _summary(best) + (f"; {errors} requests failed" if errors else "")
    ctx.add(
        CheckResult(
            "serve.load", status, text, data={"levels": levels},
            hint="see vllm.log in the run directory" if errors else None,
        )
    )  # fmt: skip


def _run_endpoint(
    ctx: RunContext, base: str, model: str | None, opts: ServeOptions, section: dict[str, Any]
) -> None:
    try:
        model = (
            model or (client.served_models(base, api_key=os.environ.get(ENV_API_KEY)) or [""])[0]
        )
    except (OSError, ValueError) as exc:
        ctx.add(CheckResult("serve.checks", Status.FAIL, f"cannot reach {base}: {exc}",
                            hint="check the URL and that the server is listening there"))  # fmt: skip
        return
    # A known model gets its preset's settings (how to switch reasoning off, tool parser).
    preset = presets.by_name(opts.preset) if opts.preset else presets.match_model(model)
    section |= {"endpoint": base, "model": model, "preset": preset.to_dict() if preset else None}
    _test(ctx, base, model, preset, opts, section, sample=urlsplit(base).hostname in _LOCAL)


def _testable(server: dict[str, Any]) -> bool:
    return server["framework"] in workloads.OPENAI_COMPATIBLE and bool(server.get("serving"))


def run(ctx: RunContext, opts: ServeOptions) -> None:
    section: dict[str, Any] = {"mode": "endpoint" if opts.endpoint else "launch"}
    ctx.sections.setdefault("bench", {})["serve"] = section

    if opts.endpoint:
        _run_endpoint(ctx, opts.endpoint.rstrip("/"), opts.model, opts, section)
        return

    # Never fight a model that already holds the GPUs: test it instead, or refuse to launch.
    running = workloads.gather()
    section["running"] = running["servers"]
    local = next((s for s in running["servers"] if _testable(s)), None)
    if local and not opts.preset:
        section["mode"] = "endpoint (already running)"
        log.info(
            "testing the %s server already running on port %s", local["framework"], local["port"]
        )
        _run_endpoint(ctx, local["endpoint"], local["serving"]["ids"][0], opts, section)
        return
    gpus = detected_gpus()
    hw = presets.matching_hardware(gpus)
    preset = (
        presets.by_name(opts.preset)
        if opts.preset
        else next((p for p in presets.for_hardware(hw) if p.default), None)
    )
    section["hardware"] = hw.id if hw else None
    if preset is None:
        ctx.add(CheckResult(
            "serve.vllm", Status.SKIP,
            f"no preset {'named ' + repr(opts.preset) if opts.preset else 'for these GPUs'}",
            hint="hwb serve --list shows the presets; --endpoint URL tests a server you already run",
        ))  # fmt: skip
        return
    section["preset"] = preset.to_dict()
    if preset.hardware != (hw.id if hw else None) and not opts.force:
        want = presets.HARDWARE[preset.hardware].label
        have = ", ".join(sorted({g.name for g in gpus})) or "no NVIDIA GPUs"
        ctx.add(CheckResult(
            "serve.vllm", Status.SKIP, f"preset {preset.name} is for {want}; this machine has {have}",
            hint="--force tries it anyway (it will likely run out of GPU memory)",
        ))  # fmt: skip
        return

    # Other workloads on some GPUs: launch on the free ones (same numbering as nvidia-smi),
    # refuse only when too few are left.
    busy = workloads.busy_gpus(running)
    needed = presets.HARDWARE[preset.hardware].gpus
    free = [str(i) for i in range(len(gpus)) if str(i) not in busy]
    env: dict[str, str] = {}
    if busy:
        if len(free) < needed and not opts.force:
            held = ", ".join(f"GPU{g} {m / 1024:.0f} GiB" for g, m in sorted(busy.items()))
            hint = (
                f"test it with: hwb serve --endpoint {local['endpoint']}"
                if local
                else "stop that workload first, or --force to launch anyway (it will likely run out of memory)"
            )
            ctx.add(
                CheckResult(
                    "serve.vllm",
                    Status.SKIP,
                    f"{preset.name} needs {needed} free GPUs; in use: {held} ({workloads.summarize(running)})",
                    hint=hint,
                )
            )
            return
        if len(free) >= needed:
            env = {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ",".join(free[:needed]),
            }
            section["gpus"] = free[:needed]
            log.info(
                "GPUs %s are busy; launching on GPUs %s",
                ",".join(sorted(busy)),
                ",".join(free[:needed]),
            )

    vllm = find_vllm(opts.vllm)
    if vllm is None and opts.install_vllm and not opts.dry_run:
        vllm = install_vllm(preset)
    argv = preset.command(host="127.0.0.1", port=opts.port, vllm=vllm or "vllm")
    section["command"] = argv
    shown = " ".join(f"{k}={v}" for k, v in env.items())
    print("vllm command:\n  " + (shown + " " if shown else "") + " ".join(argv))
    if opts.dry_run:
        ctx.add(CheckResult("serve.vllm", Status.SKIP, f"dry run: {preset.name} ({preset.status})"))
        return
    if vllm is None:
        ctx.add(CheckResult(
            "serve.vllm", Status.SKIP, "vllm not found",
            hint=f"hwb serve --install-vllm (uv venv in {vllm_venv()}), or --vllm PATH",
        ))  # fmt: skip
        return

    base = f"http://127.0.0.1:{opts.port}"
    log_path = ctx.run_dir / "vllm.log"
    with proc.spawn(argv, stdout=log_path, stderr=log_path, env=env) as child:
        ready_s = wait_ready(base, child, log_path, opts.ready_timeout_s)
        section |= {"model": preset.name, "endpoint": base, "ready_s": round(ready_s, 1)}
        ctx.add(CheckResult(
            "serve.vllm", Status.PASS, f"{preset.name} ({preset.model}) ready in {ready_s:.0f}s",
            data={"preset": preset.to_dict()},
        ))  # fmt: skip
        _test(ctx, base, preset.name, preset, opts, section, sample=True)
