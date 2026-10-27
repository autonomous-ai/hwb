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
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import nvsmi
from ..bench import analysis
from ..core import proc
from ..core.context import RunContext
from ..core.errors import HwbError, ToolMissingError
from ..core.platform import is_root
from ..core.result import CheckResult, Status
from ..metrics import workloads
from ..metrics.sampler import Sampler
from . import client, presets, suite, weights

log = logging.getLogger(__name__)

ENV_API_KEY = "HWB_LLM_API_KEY"
READY_POLL_S = 5.0
WEIGHTS_SHARE = 0.85  # auto launch: weights may use this share of the GPUs' memory
LOG_TAIL = 3000
# --load-format instanttensor reads through io_uring with registered (locked) buffers and stages
# shards on the GPU: it needs a large locked-memory limit and GPUs nobody else holds memory on.
INSTANTTENSOR_MEMLOCK = 1 << 30
INSTANTTENSOR_HELD_MIB = 512
# A worker's own exception or a compiler error, not the "see root cause above" wrappers.
_CAUSE = re.compile(r"\b(?:\w+(?:Error|Exception)|fatal error): .+")
_WRAPPERS = ("WorkerProc initialization failed", "Engine core initialization failed")
_CAUSE_HINTS = (
    ("Python.h", "Triton compiles a helper at first start and needs the Python headers of the "
                 "venv's Python: sudo apt install python3.X-dev (X = that Python), or recreate the "
                 "venv with --install-vllm, which uses uv's own Python with headers included"),
    ("FlashInfer", "FlashInfer has no prebuilt kernels in the venv and no nvcc to build them: "
                   "run vllm download-kernels from the vLLM venv"),
    ("home_path", "DeepGEMM's JIT found no CUDA toolkit: set CUDA_HOME to one (the vLLM venv "
                  "has one from pip under site-packages/nvidia/cu*), or install nvcc"),
)  # fmt: skip
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
    profiles: tuple[suite.Profile, ...] = ()  # extra input/output-length tests after the sweep


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


def nvcc_major(nvcc: str) -> int | None:
    """CUDA major version of an nvcc ("release 12.0, V12.0.140" -> 12), None if it won't say."""
    res = proc.run([nvcc, "--version"], timeout=30)
    m = re.search(r"release (\d+)\.", res.stdout) if res.ok else None
    return int(m.group(1)) if m else None


def venv_env(vllm: str, environ: dict[str, str] | None = None) -> dict[str, str]:
    """What activating vLLM's venv would give it: its bin on PATH (JIT builds call ninja from
    there), and CUDA_HOME at the venv's pip toolkit (nvidia/cu*/bin/nvcc) when the rig has no
    CUDA toolkit, since DeepGEMM and FlashInfer don't look for that one on their own. A system
    toolkit older than the pip one also loses: FlashInfer takes the CUDA version from the nvcc
    on PATH, and Ubuntu's nvidia-cuda-toolkit (12.0) knows no Blackwell, so its JIT then finds
    "No supported CUDA architectures". An explicit CUDA_HOME/CUDA_PATH is always kept."""
    environ = dict(os.environ) if environ is None else environ
    venv = Path(vllm).resolve().parent.parent
    env: dict[str, str] = {}
    if (venv / "pyvenv.cfg").exists():
        env["PATH"] = os.pathsep.join(p for p in (str(venv / "bin"), environ.get("PATH")) if p)
    if environ.get("CUDA_HOME") or environ.get("CUDA_PATH"):
        return env
    found = sorted(venv.glob("lib/python3*/site-packages/nvidia/cu*/bin/nvcc"))
    if not found:
        return env
    pip_major = (
        int(m.group(1)) if (m := re.fullmatch(r"cu(\d+)", found[-1].parent.parent.name)) else 0
    )
    system = proc.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None
    )
    if system and (nvcc_major(system) or 0) >= pip_major:
        return env
    if system:
        log.info(
            "%s is older than CUDA %d: JIT kernels use the vLLM venv's toolkit", system, pip_major
        )
    env["CUDA_HOME"] = str(found[-1].parent.parent)
    complete_pip_toolkit(found[-1].parent.parent)
    return env


def complete_pip_toolkit(home: Path) -> None:
    """Give the pip toolkit the layout a JIT link step expects of a CUDA home: lib64/ and the
    unversioned libX.so names (pip ships lib/ and libX.so.N only, so `-lcudart` fails)."""
    lib = home / "lib"
    if not lib.is_dir():
        return
    try:
        if not (home / "lib64").exists():
            (home / "lib64").symlink_to("lib")
        for so in lib.glob("lib*.so.*"):
            plain = lib / (so.name.split(".so.")[0] + ".so")
            if not plain.exists() and not plain.is_symlink():
                plain.symlink_to(so.name)
    except OSError as e:
        log.warning("could not complete the CUDA toolkit layout in %s: %s", home, e)


def loader_problem(memlock: int | None, held_mib: dict[str, float]) -> str | None:
    """Why --load-format instanttensor would fail here (memlock None = unlimited), or None."""
    if memlock is not None and memlock < INSTANTTENSOR_MEMLOCK:
        return (f"the locked-memory limit is {memlock // 1024} KiB, too small for its io_uring "
                "buffers (ulimit -l unlimited, or memlock in /etc/security/limits.conf)")  # fmt: skip
    held = {g: m for g, m in held_mib.items() if m > INSTANTTENSOR_HELD_MIB}
    if held:
        where = ", ".join(f"GPU{g} {m:.0f} MiB" for g, m in sorted(held.items()))
        return f"memory is already in use ({where}; a desktop?) and its GPU staging runs out"
    return None


def _memlock() -> int | None:
    """This process's locked-memory limit, the soft one first raised to the hard one (vLLM
    inherits it); None when unlimited."""
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != hard:
        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        except (ValueError, OSError):
            return None if soft == resource.RLIM_INFINITY else soft
    return None if hard == resource.RLIM_INFINITY else hard


def _held_mib(indices: list[str]) -> dict[str, float]:
    try:
        rows = nvsmi.query_gpus(("index", "memory.used"))
    except (ToolMissingError, nvsmi.NvidiaSmiError):
        return {}
    return {
        str(r.get("index")): float(r.get("memory.used") or 0)
        for r in rows
        if str(r.get("index")) in indices
    }


# The pip CUDA toolkit pieces a JIT compile uses; vLLM's dependencies pull the newest nvcc while
# torch pins its runtime headers, and CCCL refuses to compile when the two differ.
_PIP_NVCC = ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-cccl")


def _match_pip_nvcc(venv: Path) -> None:
    py = str(venv / "bin" / "python")
    res = proc.run([py, "-c", "import torch; print(torch.version.cuda or '')"], timeout=300)
    cuda = res.stdout.strip() if res.ok else ""
    if not cuda or not list(venv.glob("lib/python3*/site-packages/nvidia/cu*/bin/nvcc")):
        return
    pins = [f"{name}=={cuda}.*" for name in _PIP_NVCC]
    res = proc.run(["uv", "pip", "install", "--python", py, *pins], timeout=1800)
    if not res.ok:
        log.warning("could not pin the venv's nvcc to CUDA %s: %s", cuda, res.stderr[-500:])


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
    # uv's own Python ships its headers; a distro Python needs python3.X-dev, which Triton's
    # first-start compile of a C helper fails without.
    for argv in (
        ["uv", "venv", str(venv), "--python", "3.12", "--managed-python", "--allow-existing"],
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), spec],
    ):
        res = proc.run(argv, timeout=3600)
        if not res.ok:
            raise HwbError(f"{' '.join(argv[:3])} failed: {(res.stderr or res.stdout)[-500:]}")
    vllm = str(venv / "bin" / "vllm")
    _match_pip_nvcc(venv)
    # Prebuilt FlashInfer kernels: a fresh OS has no nvcc to build them at startup.
    res = proc.run([vllm, "download-kernels"], timeout=3600)
    if not res.ok:
        log.warning("vllm download-kernels failed: %s", (res.stderr or res.stdout)[-500:])
    return vllm


def failure_causes(text: str, limit: int = 5) -> list[str]:
    """The distinct root-cause lines in a vLLM log; its tail is mostly the wrapping tracebacks."""
    causes: list[str] = []
    for line in text.splitlines():
        m = _CAUSE.search(line)
        if m and not any(w in line for w in _WRAPPERS):
            cause = m.group(0).strip()[:400]
            if cause not in causes:
                causes.append(cause)
    return causes[:limit]


def _tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    out = text[-LOG_TAIL:]
    causes = failure_causes(text)
    if causes:
        out += "\nRoot cause:\n" + "\n".join(f"  {c}" for c in causes)
        hints = [h for key, h in _CAUSE_HINTS if any(key in c for c in causes)]
        out += "".join(f"\nHint: {h}" for h in hints)
    return out + f"\nFull log: {path}"


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


def _run_profiles(
    base: str,
    model: str,
    opts: ServeOptions,
    sampler: Sampler,
    no_thinking: dict[str, Any],
    api_key: str | None,
    offset: int,
) -> list[dict[str, Any]]:
    """Each --profiles entry over the same concurrency levels, with prompts as long as it says."""
    if not opts.profiles:
        return []
    per_word, overhead = suite.tokens_per_word(
        base, model, no_thinking=no_thinking, api_key=api_key
    )
    limit = suite.max_model_len(base, model, api_key=api_key)
    out: list[dict[str, Any]] = []
    for prof in opts.profiles:
        entry: dict[str, Any] = {
            "name": prof.name, "about": prof.about, "input_tokens": prof.input_tokens,
            "output_tokens": prof.output_tokens, "levels": [],
        }  # fmt: skip
        out.append(entry)
        if limit and prof.input_tokens + prof.output_tokens > limit:
            entry["skipped"] = (
                f"needs {prof.input_tokens + prof.output_tokens} tokens of context; "
                f"the server allows {limit}"
            )
            continue
        words = max(1, round((prof.input_tokens - overhead) / per_word))
        for c in opts.concurrency:
            sampler.phase(f"{prof.name} c={c}")
            n = max(c * opts.requests_per_level, 1)
            entry["levels"].append(
                suite.run_level(
                    base, model, concurrency=c, requests=n, max_tokens=prof.output_tokens,
                    no_thinking=no_thinking, api_key=api_key, offset=offset, prompt_words=words,
                )
            )  # fmt: skip
            offset += n
    return out


def _profile_result(prof: dict[str, Any]) -> CheckResult:
    name = f"serve.profile.{prof['name']}"
    if prof.get("skipped"):
        return CheckResult(name, Status.SKIP, f"{prof['about']}: {prof['skipped']}")
    levels = prof["levels"]
    ok_levels = [lv for lv in levels if lv["ok"]]
    errors = sum(lv["errors"] for lv in levels)
    if not ok_levels:
        first = levels[0]["first_error"] if levels else "?"
        return CheckResult(
            name, Status.FAIL, f"{prof['about']}: every request failed: {first}",
            data={"levels": levels}, hint="see vllm.log in the run directory",
        )  # fmt: skip
    best = max(ok_levels, key=lambda lv: lv["total_tok_per_s"] or lv["output_tok_per_s"] or 0)
    text = f"{prof['about']}: "
    if best.get("total_tok_per_s"):
        text += f"{best['total_tok_per_s']} tok/s in+out ({best['output_tok_per_s']} out)"
    else:
        text += f"{best['output_tok_per_s']} tok/s out"
    text += f" at {best['concurrency']} concurrent"
    if best.get("ttft_ms_p50") is not None:
        text += f", TTFT p50 {best['ttft_ms_p50']:.0f} ms"
    if errors:
        text += f"; {errors} requests failed"
    return CheckResult(
        name, Status.WARN if errors else Status.PASS, text, data={"levels": levels},
        hint="see vllm.log in the run directory" if errors else None,
    )  # fmt: skip


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
        profiles = _run_profiles(base, model, opts, s, no_thinking, api_key, offset)
    section |= {"checks": checks, "load": levels}
    if profiles:
        section["profiles"] = profiles
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
    for prof in profiles:
        ctx.add(_profile_result(prof))
    if sample and s.result.meta.get("pcie"):
        # serving is often the heaviest PCIe traffic a rig sees (tensor-parallel all-reduce)
        found = analysis.pcie_findings(s.result)
        section["findings"] = [vars(f) for f in found]
        level = analysis.verdict(found)
        ctx.add(
            CheckResult(
                "serve.pcie",
                Status.FAIL if level == "fail" else Status.WARN if found else Status.PASS,
                "; ".join(f.text for f in found) if found
                else "PCIe links at full speed with no errors while serving",
                hint=found[0].hint if found else None,
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
    section["hardware"] = hw.id if hw else None
    found = weights.find_all()
    section["weights_on_disk"] = [w.to_dict() for w in found]
    busy = workloads.busy_gpus(running)
    free = [str(i) for i in range(len(gpus)) if str(i) not in busy]

    if opts.model:  # any model on this machine, without a preset
        preset, needed, why = _auto_preset(opts.model, found, gpus, free)
    else:
        preset, why = _choose_preset(opts.preset, hw, found)
        needed = presets.HARDWARE[preset.hardware].gpus if preset else 0
    if preset is None:
        ctx.add(CheckResult("serve.vllm", Status.SKIP, why,
                            hint="hwb serve --list shows presets and the models on this machine; "
                            "--endpoint URL tests a server you already run"))  # fmt: skip
        return
    if why:
        log.info("%s", why)
    section["preset"] = preset.to_dict()
    if (
        preset.hardware in presets.HARDWARE
        and preset.hardware != (hw.id if hw else None)
        and not opts.force
    ):
        want = presets.HARDWARE[preset.hardware].label
        have = ", ".join(sorted({g.name for g in gpus})) or "no NVIDIA GPUs"
        ctx.add(CheckResult(
            "serve.vllm", Status.SKIP, f"preset {preset.name} is for {want}; this machine has {have}",
            hint="--force tries it anyway (it will likely run out of GPU memory)",
        ))  # fmt: skip
        return

    # Other workloads on some GPUs: launch on the free ones (same numbering as nvidia-smi),
    # refuse only when too few are left.
    env: dict[str, str] = {}
    if busy:
        if len(free) < needed and not opts.force:
            held = ", ".join(f"GPU{g} {m / 1024:.0f} GiB" for g, m in sorted(busy.items()))
            hint = (
                f"test it with: hwb serve --endpoint {local['endpoint']}"
                if local
                else "stop that workload first, or --force to launch anyway (it will likely run out of memory)"
            )
            ctx.add(CheckResult(
                "serve.vllm", Status.SKIP,
                f"{preset.name} needs {needed} free GPUs; in use: {held} ({workloads.summarize(running)})",
                hint=hint,
            ))  # fmt: skip
            return
        if len(free) >= needed:
            env |= {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ",".join(free[:needed]),
            }
            section["gpus"] = free[:needed]
            log.info(
                "GPUs %s are busy; launching on GPUs %s",
                ",".join(sorted(busy)),
                ",".join(free[:needed]),
            )

    model_arg, weights_env, plan, problem = _weights_plan(preset, found, force=opts.force)
    section["weights"] = plan
    if problem:
        ctx.add(CheckResult("serve.vllm", Status.SKIP, problem[0], hint=problem[1]))
        return
    env |= weights_env

    vllm = find_vllm(opts.vllm)
    if vllm is None and opts.install_vllm and not opts.dry_run:
        vllm = install_vllm(preset)
    if vllm:
        env |= venv_env(vllm)
    # vLLM trusts the driver's P2P report unless told to test it; with the IOMMU translating
    # (Ubuntu's default on Intel), P2P reads return garbage and TP all-reduce hangs in warmup.
    env.setdefault("VLLM_SKIP_P2P_CHECK", os.environ.get("VLLM_SKIP_P2P_CHECK", "0"))
    argv = preset.command(host="127.0.0.1", port=opts.port, vllm=vllm or "vllm", model=model_arg)
    section["command"] = argv
    if "instanttensor" in argv and not opts.dry_run:
        used = env.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(needed))).split(",")
        no_fast_load = loader_problem(_memlock(), _held_mib(used))
        if no_fast_load:
            at = argv.index("--load-format")
            del argv[at : at + 2]
            section["loader"] = f"vLLM's default loader: instanttensor would fail, {no_fast_load}"
            log.warning("not using --load-format instanttensor: %s", no_fast_load)
    shown = " ".join(f"{k}={v}" for k, v in env.items() if k != "PATH")  # PATH: the venv's bin
    print(f"weights: {plan['note']}")
    print("vllm command:\n  " + (shown + " " if shown else "") + " ".join(argv))
    if opts.dry_run:
        ctx.add(
            CheckResult(
                "serve.vllm",
                Status.SKIP,
                f"dry run: {preset.name} ({preset.status}); {plan['note']}",
            )
        )
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
            "serve.vllm", Status.PASS, f"{preset.name} ({preset.model}) ready in {ready_s:.0f}s; {plan['note']}",
            data={"preset": preset.to_dict(), "weights": plan},
        ))  # fmt: skip
        _test(ctx, base, preset.name, preset, opts, section, sample=True)


def _choose_preset(
    name: str | None, hw: presets.Hardware | None, found: list[weights.Weights]
) -> tuple[presets.Preset | None, str]:
    """The named preset; otherwise, for this hardware, one whose weights are already on disk
    (default first, then verified), else the default (which downloads)."""
    if name:
        p = presets.by_name(name)
        return p, "" if p else f"no preset named {name!r}"
    candidates = presets.for_hardware(hw)
    if not candidates:
        return None, "no preset for these GPUs"
    on_disk = [p for p in candidates if weights.find(found, p.model, p.revision)]
    ranked = sorted(on_disk, key=lambda p: (not p.default, p.status != presets.VERIFIED))
    if ranked:
        return ranked[0], f"using {ranked[0].name}: its weights are already on this machine"
    default = next(p for p in candidates if p.default)
    return (
        default,
        f"no preset's weights are on this machine yet; {default.name} will be downloaded",
    )


def _auto_preset(
    model: str, found: list[weights.Weights], gpus: list[presets.Gpu], free: list[str]
) -> tuple[presets.Preset | None, int, str]:
    """A preset for a model on this machine (HF id in a cache, or a directory) that has none:
    as few GPUs as fit its weights, safe default flags, no tool parser assumed."""
    path = Path(model).expanduser().resolve()
    if path.is_dir():
        complete, fmt, size = weights.snapshot_state(path)
        hit: weights.Weights | None = weights.Weights(path.name, "local", path, size, fmt, complete)
    else:
        hit = weights.find(found, model) or next(
            (w for w in found if w.id.lower() == model.lower()), None
        )
    if hit is None:
        return None, 0, (f"{model} is not on this machine: fetch it into the shared cache with "
                         f"`hf download {model}`, or pick a preset")  # fmt: skip
    if not hit.vllm_runnable:
        what = "incomplete download" if not hit.complete else f"{hit.format} weights"
        return (
            None,
            0,
            f"{model}: {what}, not runnable by vLLM (GGUF/Ollama: test its own server with --endpoint)",
        )
    mem = min((g.mem_gib for i, g in enumerate(gpus) if str(i) in free), default=0) * 1.0737
    if mem <= 0:
        return None, 0, "no free NVIDIA GPU"
    # Weights within 85 % of the chosen GPUs' memory, the rest for KV cache and CUDA graphs
    # (the verified DeepSeek preset runs 168 GB on 2 x 96 GB this way).
    tp = 1
    while hit.size_gb > tp * mem * WEIGHTS_SHARE and tp < 64:
        tp *= 2  # vLLM splits attention heads evenly: powers of two
    if tp > len(free):
        return (
            None,
            0,
            f"{model} ({hit.size_gb:.0f} GB) needs {tp} free GPUs of {mem:.0f} GB; {len(free)} free",
        )
    preset = presets.Preset(
        name=hit.id.rsplit("/", 1)[-1],
        family="auto",
        hardware="this machine",
        model=str(hit.path) if hit.source != "huggingface" else hit.id,
        revision=hit.revision if hit.source == "huggingface" else None,
        weights_gb=round(hit.size_gb, 1),
        quant="as stored",
        context=0,
        status="auto",
        tools=False,
        no_thinking=suite.NO_THINKING_ANY,
        args=(
            "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", "0.92",
            "--enable-prefix-caching", "--enable-chunked-prefill",
        ),
    )  # fmt: skip
    return preset, tp, f"{model}: no preset, launching on {tp} GPU(s) with default flags"


def _weights_plan(
    preset: presets.Preset, found: list[weights.Weights], *, force: bool
) -> tuple[str | None, dict[str, str], dict[str, Any], tuple[str, str] | None]:
    """(model argument, env, report entry, problem): reuse a copy on disk where possible
    (pointing vLLM at whichever cache holds it), else download into the shared HF cache."""
    if Path(preset.model).is_absolute():  # an ad-hoc preset for a local directory
        return (
            preset.model,
            {},
            {"cached": True, "source": "local", "note": "local directory"},
            None,
        )
    hit = weights.find(found, preset.model, preset.revision)
    if hit and hit.source == "huggingface":
        env = {"HF_HUB_CACHE": str(hit.hub)} if hit.hub and hit.hub != weights.default_hub() else {}
        where = "another Hugging Face cache on this machine" if env else "the Hugging Face cache"
        if weights.has_tokenizer(hit.path):
            # Everything vLLM loads is on disk: no Hub lookups, no lock files in someone's cache.
            env["HF_HUB_OFFLINE"] = "1"
        plan = {"cached": True, "source": hit.source, "offline": "HF_HUB_OFFLINE" in env,
                "note": f"already in {where}, no download"}  # fmt: skip
        return None, env, plan, None
    if hit:  # ModelScope / plain directory copy of the same model
        return (
            str(hit.path),
            {},
            {
                "cached": True,
                "source": hit.source,
                "note": f"{hit.source} copy on disk, no download",
            },
            None,
        )
    other = weights.find(found, preset.model)
    target = weights.default_hub()
    need = preset.weights_gb * (0.2 if other else 1.05)
    free = weights.free_gb(target)
    note = ("other revision cached: only changed files download" if other
            else f"downloads ~{preset.weights_gb:.0f} GB into the shared Hugging Face cache")  # fmt: skip
    plan = {
        "cached": False,
        "source": "huggingface",
        "note": note,
        "free_gb": round(free),
        "need_gb": round(need),
    }
    if free < need and not force:
        return (
            None,
            {},
            plan,
            (
                f"not enough disk for {preset.name}: {free:.0f} GB free in the Hugging Face cache, ~{need:.0f} GB needed",
                "point HF_HOME at a bigger disk (vLLM, transformers and `hf` all follow it), or --force",
            ),
        )
    if is_root() and os.environ.get("SUDO_USER"):
        log.warning("running as root: the download goes to root's Hugging Face cache, not %s's. "
                    "Run hwb serve without sudo (it needs no root) to share one cache.",
                    os.environ["SUDO_USER"])  # fmt: skip
    return None, {}, plan, None
