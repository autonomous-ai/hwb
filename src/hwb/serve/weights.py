"""Model weights already on this machine, and where new downloads go.

hwb never keeps its own model store. Weights live where the common tools look, so a model
downloaded once serves every tool:

- Hugging Face hub cache (vLLM, SGLang, transformers, `hf download`): $HF_HUB_CACHE, else
  $HF_HOME/hub, else ~/.cache/huggingface/hub; as root also every user's cache, so a model a
  user already downloaded is found instead of fetched again.
- ModelScope (~/.cache/modelscope/hub), plain model directories ($HWB_MODEL_DIRS, /models,
  /data/models): safetensors that vLLM loads from the path.
- Ollama, LM Studio, llama.cpp caches: GGUF, listed so they can be tested through their own
  server (`hwb serve --endpoint`).

A Hugging Face snapshot counts as complete when its config and every safetensors shard named
in the index resolve to real files; interrupted downloads leave dangling links or no shards.
Reports carry ids, sizes and completeness, never paths (they contain user names).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.platform import is_root

ROOT = Path("/")
ENV_MODEL_DIRS = "HWB_MODEL_DIRS"  # extra directories with models, colon-separated
COMMON_MODEL_DIRS = ("models", "data/models", "opt/models")  # relative to ROOT
_MAX_DEPTH = 2  # plain model directories are searched this deep


@dataclass(frozen=True)
class Weights:
    id: str  # HF id ("org/name"), Ollama "name:tag", or a local directory's name
    source: str  # huggingface | modelscope | local | ollama | lmstudio | llama.cpp
    path: Path  # snapshot / model directory (never reported)
    size_gb: float
    format: str  # safetensors | gguf | ollama
    complete: bool
    revision: str | None = None
    hub: Path | None = None  # the HF hub cache it lives in (HF_HUB_CACHE for vLLM)

    @property
    def architecture(self) -> str | None:
        """config.json's architecture (e.g. Qwen3_5ForConditionalGeneration): tells a text
        LLM vLLM can serve from, say, a music or vision-only model."""
        try:
            arch = json.loads((self.path / "config.json").read_text()).get("architectures") or []
        except (OSError, ValueError, AttributeError):
            return None
        return str(arch[0]) if arch else None

    @property
    def vllm_runnable(self) -> bool:
        return self.complete and self.format == "safetensors"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "size_gb": round(self.size_gb, 1),
            "format": self.format,
            "architecture": self.architecture,
            "complete": self.complete,
            "revision": self.revision,
        }


def _homes(root: Path) -> list[Path]:
    """Home directories to search: your own; as root every user's (a model a user already
    downloaded must not be fetched again). A test tree (root != /) is searched alone."""
    if root != Path("/"):
        return [root / "root", *sorted((root / "home").glob("*"))]
    homes = [Path.home()]
    if is_root():
        homes += [Path("/root"), *sorted(Path("/home").glob("*"))]
    out: list[Path] = []
    for h in homes:
        if h not in out:
            out.append(h)
    return out


def default_hub() -> Path:
    """Where Hugging Face tools (and so vLLM) download to, exactly as they resolve it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_hubs(root: Path) -> list[Path]:
    hubs = [default_hub()] if root == Path("/") else []
    hubs += [h / ".cache" / "huggingface" / "hub" for h in _homes(root)]
    seen, out = set(), []
    for h in hubs:
        try:
            key = h.resolve()
        except OSError:
            continue
        if key not in seen and h.is_dir():
            seen.add(key)
            out.append(h)
    return out


def _size(files: list[Path]) -> float:
    total = 0
    for f in files:
        with contextlib.suppress(OSError):  # a dangling link: an interrupted download
            total += f.stat().st_size  # follows the snapshot's links into blobs/
    return total / 1e9


def snapshot_state(snap: Path) -> tuple[bool, str, float]:
    """(complete, format, size in GB) of one model directory or HF snapshot."""
    files = [f for f in snap.rglob("*") if not f.is_dir() or f.is_symlink()]
    size = _size(files)
    if any(f.suffix == ".gguf" and f.exists() for f in files):
        return True, "gguf", size
    index = snap / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError):
            return False, "safetensors", size
        ok = (snap / "config.json").exists() and all((snap / s).exists() for s in shards)
        return ok, "safetensors", size
    single = [f for f in files if f.suffix == ".safetensors"]
    ok = bool(single) and all(f.exists() for f in single) and (snap / "config.json").exists()
    return ok, "safetensors", size


def scan_hf_hub(hub: Path) -> list[Weights]:
    out = []
    for repo in sorted(hub.glob("models--*")):
        parts = repo.name.split("--")[1:]
        if len(parts) < 2:
            continue
        model_id = f"{parts[0]}/{'--'.join(parts[1:])}"
        for snap in sorted((repo / "snapshots").glob("*")):
            if not snap.is_dir():
                continue
            complete, fmt, size = snapshot_state(snap)
            if size == 0:
                continue
            out.append(Weights(model_id, "huggingface", snap, size, fmt, complete, snap.name, hub))
    return out


def _model_dirs(base: Path, depth: int = _MAX_DEPTH) -> list[Path]:
    """Directories under base (up to depth levels) that hold a model."""
    if not base.is_dir():
        return []
    found, frontier = [], [base]
    for _ in range(depth + 1):
        nxt = []
        for d in frontier:
            if (d / "config.json").exists() or any(d.glob("*.gguf")):
                found.append(d)
                continue
            nxt += [c for c in sorted(d.iterdir()) if c.is_dir() and not c.name.startswith(".")]
        frontier = nxt
    return found


def scan_dirs(root: Path) -> list[Weights]:
    out = []
    homes = _homes(root)
    sources: list[tuple[str, Path, bool]] = [  # (source, base, id is org/name)
        *(("modelscope", h / ".cache" / "modelscope" / "hub" / "models", True) for h in homes),
        *(("modelscope", h / ".cache" / "modelscope" / "hub", True) for h in homes),
        *(("lmstudio", h / ".lmstudio" / "models", True) for h in homes),
        *(("llama.cpp", h / ".cache" / "llama.cpp", False) for h in homes),
        *(("local", root / d, False) for d in COMMON_MODEL_DIRS),
        *(("local", Path(d), False) for d in os.environ.get(ENV_MODEL_DIRS, "").split(":") if d),
    ]
    seen: set[Path] = set()
    for source, base, org_name in sources:
        for d in _model_dirs(base):
            if d in seen or d == base:
                continue
            seen.add(d)
            complete, fmt, size = snapshot_state(d)
            if size == 0:
                continue
            rel = d.relative_to(base).parts
            model_id = "/".join(rel[-2:]) if org_name and len(rel) >= 2 else d.name
            out.append(Weights(model_id, source, d, size, fmt, complete))
    return out


def scan_ollama(root: Path) -> list[Weights]:
    out = []
    bases = [h / ".ollama" / "models" for h in _homes(root)] + [
        root / "usr/share/ollama/.ollama/models"
    ]
    for base in bases:
        manifests = base / "manifests"
        for m in sorted(manifests.glob("*/*/*/*")) if manifests.is_dir() else []:
            try:
                layers = json.loads(m.read_text()).get("layers", [])
            except (OSError, ValueError):
                continue
            ns, name, tag = m.parts[-3], m.parts[-2], m.name
            model_id = f"{name}:{tag}" if ns == "library" else f"{ns}/{name}:{tag}"
            size = sum(int(layer.get("size") or 0) for layer in layers) / 1e9
            out.append(Weights(model_id, "ollama", m, size, "ollama", True))
    return out


def find_all(root: Path | None = None) -> list[Weights]:
    root = ROOT if root is None else root
    found = [w for hub in hf_hubs(root) for w in scan_hf_hub(hub)]
    return found + scan_dirs(root) + scan_ollama(root)


def find(found: list[Weights], model_id: str, revision: str | None = None) -> Weights | None:
    """A complete, vLLM-runnable copy of model_id (at revision, if given)."""
    want = model_id.lower()
    for w in found:
        if (
            w.vllm_runnable
            and w.id.lower() == want
            and (revision is None or w.revision == revision)
        ):
            return w
    return None


def free_gb(path: Path) -> float:
    """Free space where path is (or would be created)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def summarize(found: list[Weights]) -> str:
    return summarize_dicts([w.to_dict() for w in found])


def summarize_dicts(models: list[dict[str, Any]]) -> str:
    """summarize() for the report form (no paths)."""
    runnable = [m for m in models if m.get("complete") and m.get("format") == "safetensors"]
    if not models:
        return "no model weights found"
    head = ", ".join(f"{m['id']} ({m['size_gb']:.0f} GB)" for m in runnable[:4])
    extra = len(models) - len(runnable)
    return (f"{len(runnable)} runnable with vLLM" + (f": {head}" if head else "")
            + (f"; {extra} other (GGUF/Ollama or incomplete)" if extra else ""))  # fmt: skip


def has_tokenizer(path: Path) -> bool:
    return any(
        (path / f).exists() for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    )
