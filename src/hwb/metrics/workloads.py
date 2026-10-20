"""What is already running on the GPUs: compute processes and the model servers behind them.

nvidia-smi lists the processes holding GPU memory. Serving frameworks run their GPU work in
worker processes, so each one is followed up its parent chain to the server that started it,
which is identified (vLLM, SGLang, llama.cpp, Ollama, TGI, Triton) and its model, served name
and port read from well-known flags. A local server is then asked for its models
(/v1/models, read-only GET) to confirm.

Privacy: full command lines are never stored (they can carry --api-key and paths); only the
parsed model id (local model paths reduced to their last component), served name and port.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import nvsmi
from ..core import proc
from ..core.errors import HwbError

APP_FIELDS = ("pid", "process_name", "used_memory", "gpu_bus_id")
_PARENT_HOPS = 6
_PROBE_TIMEOUT_S = 2.0
ROOT = Path("/")
# Frameworks that speak the OpenAI API, so `hwb serve --endpoint` can test them.
OPENAI_COMPATIBLE = ("vllm", "sglang", "llama.cpp", "tgi", "ollama")


@dataclass(frozen=True)
class Framework:
    name: str
    match: re.Pattern[str]  # on the joined command line
    model_flags: tuple[str, ...]
    default_port: int
    positional_after: str | None = None  # `vllm serve MODEL`


FRAMEWORKS = (
    Framework("vllm", re.compile(r"(?:^|[/\s])vllm(?:\s|$)|vllm\.entrypoints"), ("--model",), 8000, "serve"),
    Framework("sglang", re.compile(r"sglang\.launch_server|(?:^|[/\s])sglang\s+serve"), ("--model-path", "--model"), 30000),
    Framework("llama.cpp", re.compile(r"(?:^|/)llama-server(?:\s|$)"), ("-m", "--model", "-hf", "--hf-repo"), 8080),
    Framework("ollama", re.compile(r"(?:^|/)ollama(?:\s|$)"), (), 11434),
    Framework("tgi", re.compile(r"text-generation-launcher|text-generation-router"), ("--model-id",), 80),
    Framework("triton", re.compile(r"(?:^|/)tritonserver(?:\s|$)"), ("--model-repository",), 8000),
)  # fmt: skip


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def cmdline(root: Path, pid: int) -> list[str]:
    text = _read_text(root / "proc" / str(pid) / "cmdline") or ""
    return [a for a in text.split("\0") if a]


def parent_pid(root: Path, pid: int) -> int | None:
    stat = _read_text(root / "proc" / str(pid) / "stat") or ""
    fields = stat.rpartition(")")[2].split()  # comm may contain spaces and parens
    return (
        int(fields[1]) if len(fields) > 1 and fields[1].isdigit() and int(fields[1]) > 1 else None
    )


def identify(argv: list[str]) -> Framework | None:
    joined = " ".join(argv)
    return next((f for f in FRAMEWORKS if f.match.search(joined)), None)


def _flag(argv: list[str], names: tuple[str, ...]) -> str | None:
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def _model_name(value: str | None) -> str | None:
    """HF ids stay as they are; local paths shrink to their last part (paths carry user names)."""
    if not value:
        return None
    if value.startswith(("/", "~", ".")) or "\\" in value:
        return Path(value.rstrip("/")).name or None
    return value


def describe(argv: list[str], fw: Framework) -> dict[str, Any]:
    model = _flag(argv, fw.model_flags)
    if model is None and fw.positional_after and fw.positional_after in argv:
        i = argv.index(fw.positional_after)
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            model = argv[i + 1]
    port = _flag(argv, ("--port",))
    tp = _flag(argv, ("--tensor-parallel-size", "-tp", "--tp-size", "--tp"))
    return {
        "framework": fw.name,
        "model": _model_name(model),
        "served_name": _flag(argv, ("--served-model-name",)),
        "port": int(port) if port and port.isdigit() else fw.default_port,
        "tensor_parallel": int(tp) if tp and tp.isdigit() else None,
    }


def parse_apps(text: str) -> list[dict[str, Any]]:
    rows = []
    for raw in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(raw) != len(APP_FIELDS) or not raw[0].strip().isdigit():
            continue
        pid, name, mem, bus = (c.strip() for c in raw)
        rows.append(
            {
                "pid": int(pid),
                "process": Path(name).name,  # the binary, never its full path
                "used_mib": int(mem) if mem.isdigit() else None,
                "bus_id": bus,
            }
        )
    return rows


def compute_apps() -> list[dict[str, Any]]:
    if not proc.have("nvidia-smi"):
        return []
    res = proc.run(
        [
            "nvidia-smi",
            f"--query-compute-apps={','.join(APP_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    return parse_apps(res.stdout) if res.ok else []


def probe_models(port: int) -> dict[str, Any] | None:
    """Ask a local OpenAI-compatible server what it serves (read-only)."""
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    models = [m for m in (data or {}).get("data", []) if isinstance(m, dict)]
    return {
        "ids": [str(m.get("id")) for m in models],
        "max_model_len": next(
            (m.get("max_model_len") for m in models if m.get("max_model_len")), None
        ),
    }


@dataclass
class Server:
    info: dict[str, Any]
    root_pid: int
    pids: set[int] = field(default_factory=set)
    gpus: set[str] = field(default_factory=set)
    used_mib: int = 0


def gather(root: Path | None = None, *, probe: bool = True) -> dict[str, Any]:
    root = ROOT if root is None else root
    apps = compute_apps()
    try:
        index = {
            nvsmi_bus(r["pci.bus_id"]): str(r["index"])
            for r in nvsmi.query_gpus(("index", "pci.bus_id"))
        }
    except HwbError:
        index = {}
    servers: dict[int, Server] = {}
    processes = []
    for app in apps:
        gpu = index.get(nvsmi_bus(app["bus_id"]), app["bus_id"])
        processes.append(
            {"gpu": gpu, "pid": app["pid"], "process": app["process"], "used_mib": app["used_mib"]}
        )
        pid: int | None = app["pid"]
        for _ in range(_PARENT_HOPS):  # the GPU process is often a worker of the server
            if pid is None:
                break
            argv = cmdline(root, pid)
            fw = identify(argv)
            if fw:
                srv = servers.setdefault(pid, Server(describe(argv, fw), pid))
                srv.pids.add(app["pid"])
                srv.gpus.add(gpu)
                srv.used_mib += app["used_mib"] or 0
                break
            pid = parent_pid(root, pid)
    out = []
    for srv in servers.values():
        info = dict(srv.info)
        info |= {
            "gpus": sorted(srv.gpus),
            "used_gib": round(srv.used_mib / 1024, 1),
            "processes": len(srv.pids),
            "endpoint": f"http://127.0.0.1:{info['port']}",
        }
        if probe and info["framework"] in OPENAI_COMPATIBLE:
            info["serving"] = probe_models(int(info["port"]))
        out.append(info)
    return {"servers": out, "processes": processes}


def nvsmi_bus(bus_id: Any) -> str:
    """Bus ids as nvidia-smi prints them in different queries, normalised for matching."""
    s = str(bus_id or "").strip().lower()
    dom, _, rest = s.partition(":")
    return f"{dom[-4:].zfill(4)}:{rest}" if rest else s


def summarize(data: dict[str, Any]) -> str:
    servers = data.get("servers") or []
    if servers:
        parts = [
            f"{s['framework']} {s.get('served_name') or s.get('model') or '?'} on GPU {','.join(s['gpus'])} "
            f"({s['used_gib']} GiB, port {s['port']})"
            for s in servers
        ]
        return "; ".join(parts)
    procs = data.get("processes") or []
    if procs:
        return f"{len(procs)} GPU process(es), no known model server"
    return "GPUs idle: no compute processes"


def busy_gpus(data: dict[str, Any], min_mib: int = 1024) -> dict[str, int]:
    """GPU -> MiB held by other processes (ignores small allocations)."""
    held: dict[str, int] = {}
    for p in data.get("processes") or []:
        held[p["gpu"]] = held.get(p["gpu"], 0) + (p.get("used_mib") or 0)
    return {g: m for g, m in held.items() if m >= min_mib}
