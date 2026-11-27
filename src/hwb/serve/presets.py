"""Known-good vLLM launch presets per hardware profile.

Presets are data: model, pinned revision where known, the exact `vllm serve` arguments, and
how sure we are. "verified" means the command ran on that hardware; "untested" means it was
derived from the model card, the weight size and vLLM's source, and still needs a real run.
Model ids, revisions and flags drift: keep them here, nowhere else (like install/plan.py).

A preset may carry a container template (image + how to run it): `hwb serve` then starts the
server with docker instead of a local vLLM, which is the quick way to run builds that are not on
PyPI (model-specific vLLM images, ROCm forks). `--docker` runs any other preset in the stock
vLLM image (DEFAULT_IMAGE).

Sizing rule for a profile: weights + KV cache + CUDA graphs must fit the GPUs' memory; on
2x RTX PRO 6000 (2 x 96 GB) the verified DeepSeek run carries 168 GB of weights, which is
about the ceiling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

VLLM_SPEC = "vllm==0.32.0"  # what the presets were checked against (vLLM main, 2026-10)

DEFAULT_IMAGE = "vllm/vllm-openai:v" + VLLM_SPEC.split("==")[1]  # --docker for a plain preset

VERIFIED = "verified"
UNTESTED = "untested"


@dataclass(frozen=True)
class Hardware:
    id: str
    label: str
    gpu_name: str  # regex on nvidia-smi's name
    gpus: int
    min_mem_gib: float
    compute_cap: str  # NVIDIA: "12.0"; AMD: the gfx target, "gfx1201"
    vendor: str = "nvidia"  # nvidia | amd


@dataclass(frozen=True)
class Container:
    """How a preset runs in docker. The image's entrypoint is `vllm serve` (the official vLLM
    images and most forks) unless ``own_command``: then the image's default command serves the
    model by itself and hwb passes no arguments."""

    image: str
    port: int = 8000  # where the server listens inside the container
    env: tuple[tuple[str, str], ...] = ()
    own_command: bool = False
    cache: str | None = None  # container path for a writable per-preset cache (own weights)
    root: bool = False  # run as the image's user (its launcher does) instead of the invoking user


@dataclass(frozen=True)
class Preset:
    name: str
    family: str
    hardware: str
    model: str
    weights_gb: float
    quant: str
    context: int
    args: tuple[str, ...]
    status: str = UNTESTED
    revision: str | None = None
    vllm_extras: tuple[str, ...] = ()
    tools: bool = True  # a tool-call parser is configured
    # chat_template_kwargs that switch reasoning off, for short deterministic test answers
    no_thinking: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    default: bool = False
    container: Container | None = None  # run in docker with this template
    sources: tuple[str, ...] = ()  # where the recipe comes from

    def command(
        self,
        *,
        host: str = "0.0.0.0",  # noqa: S104 - a server preset
        port: int = 8000,
        vllm: str = "vllm",
        model: str | None = None,  # a local copy's path instead of the HF id
    ) -> list[str]:
        argv = [vllm, "serve", model or self.model]
        if self.revision and model is None:
            argv += ["--revision", self.revision]
        return [
            *argv,
            "--served-model-name",
            self.name,
            "--host",
            host,
            "--port",
            str(port),
            *self.args,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "hardware": self.hardware,
            "model": self.model,
            "revision": self.revision,
            "weights_gb": self.weights_gb,
            "quant": self.quant,
            "context": self.context,
            "status": self.status,
            "notes": self.notes,
            **({"image": self.container.image} if self.container else {}),
        }


HARDWARE = {
    h.id: h
    for h in (
        Hardware(
            "2x-rtx-pro-6000-blackwell",
            "2x NVIDIA RTX PRO 6000 Blackwell (2 x 96 GB, sm_120)",
            r"RTX PRO 6000 Blackwell",
            gpus=2,
            min_mem_gib=90,
            compute_cap="12.0",
        ),
        Hardware(
            "1x-rtx-pro-6000-blackwell",
            "1x NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120)",
            r"RTX PRO 6000 Blackwell",
            gpus=1,
            min_mem_gib=90,
            compute_cap="12.0",
        ),
        Hardware(
            "1x-rtx-5090",
            "1x NVIDIA GeForce RTX 5090 (32 GB, sm_120)",
            r"RTX 5090",
            gpus=1,
            min_mem_gib=30,
            compute_cap="12.0",
        ),
        Hardware(
            "1x-radeon-ai-pro-r9700",
            "1x AMD Radeon AI PRO R9700 (32 GB, RDNA4 gfx1201)",
            r"gfx1201",
            gpus=1,
            min_mem_gib=30,  # the 16 GB RX 9070 XT is gfx1201 too
            compute_cap="gfx1201",
            vendor="amd",
        ),
    )
}

# Shared by the large presets on this profile: FP8 KV cache, prefix caching, chunked prefill.
_PRO6000 = (
    "--tensor-parallel-size", "2",
    "--gpu-memory-utilization", "0.92",
    "--kv-cache-dtype", "fp8",
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--enable-prompt-tokens-details",
)  # fmt: skip


# Inkling 2 Mini (Thinking Machines, 2026-11-20): looped MoE, 26.4B total / 3.4B active, one
# ~20 GB checkpoint (NVFP4 experts in layers 0-19, MXFP8 in 20-23) with a bundled DSpark drafter.
# Support is in no vLLM release yet (upstream PR vllm#69028): it runs from images.
INKLING = "thinkingmachines/Inkling-2-Mini"
INKLING_REVISION = "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6"
# vLLM nightly 0.34.0rc1.dev1235, the image Hugging Face's hill-climb ran on one RTX PRO 6000
# (vllm/vllm-openai:inkling-2-mini is the tag from vLLM's launch post).
INKLING_IMAGE = (
    "vllm/vllm-openai@sha256:eaa63fd6ee3aac77f3158e80ed79ec425db3a53f7cd666bf1954e96abfaf19a7"
)
# Eliovp's Paiton vLLM plugin for RDNA4; its launcher runs the image's own command (port 8042).
PAITON = "ghcr.io/eliovp/paiton-vllm-plugin"
PAITON_W3_IMAGE = (
    f"{PAITON}:inkling-2-mini-w3-rdna4-v1.2.0"
    "@sha256:7f1eaad71457b00f548d97fa66af9389ee6730dc8e718339cfeee5d7ed3a449e"
)
_DSPARK7 = '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'


def _paiton(image: str) -> Container:
    """Eliovp's launcher (serve-docker.sh): the image's own command, port 8042, cache in
    /models/cache, root, one hardware queue."""
    return Container(image, port=8042, env=(("GPU_MAX_HW_QUEUES", "1"),), own_command=True,
                     cache="/models/cache", root=True)  # fmt: skip


# fmt: off
PRESETS: tuple[Preset, ...] = (
    Preset(
        name="deepseek-v4-flash-vision",
        family="deepseek",
        hardware="2x-rtx-pro-6000-blackwell",
        model="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        revision="6821d6ad3681a4b137b066b76094fa82ebd0a380",
        weights_gb=167.8,
        quant="fp8 + packed fp4 experts",
        context=262144,
        vllm_extras=("b12x",),
        no_thinking={"thinking": False},
        notes="Ran in production on this profile with --linear-backend b12x on another vLLM build. "
              f"With {VLLM_SPEC} that flag leaves wo_a 2-D and DeepGEMM's fp8 o_proj einsum asserts "
              "(layout.hpp t.dim() == N) at warmup, so it is left out: linear layers use DeepGEMM, "
              "attention and MoE b12x (SM120/SM121 kernels). Started and passed the answer checks "
              "on 2x RTX PRO 6000 (2026-10-27). DSpark speculative decoding.",
        args=(
            "--trust-remote-code",
            "--tensor-parallel-size",
            "2",
            "--decode-context-parallel-size",
            "1",
            "--max-model-len",
            "262144",
            "--max-num-seqs",
            "16",
            "--max-num-batched-tokens",
            "4096",
            "--gpu-memory-utilization",
            "0.95",
            "--kv-cache-dtype",
            "fp8",
            "--block-size",
            "256",
            "--load-format",
            "instanttensor",
            "--max-cudagraph-capture-size",
            "64",
            "--compilation-config",
            '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}',
            "--async-scheduling",
            "--no-scheduler-reserve-full-isl",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--prefix-cache-retention-interval",
            "4096",
            "--tokenizer-mode",
            "deepseek_v4",
            "--tool-call-parser",
            "deepseek_v4",
            "--reasoning-parser",
            "deepseek_v4",
            "--enable-auto-tool-choice",
            "--enable-prompt-tokens-details",
            "--enable-force-include-usage",
            "--enable-request-id-headers",
            "--default-chat-template-kwargs.thinking=true",
            "--default-chat-template-kwargs.reasoning_effort=high",
            "--enable-flashinfer-autotune",
            "--speculative-config",
            '{"model":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp","method":"dspark","num_speculative_tokens":3,'
            '"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}',
            "--attention-backend",
            "B12X",
            "--moe-backend",
            "b12x",
        ),
    ),
    Preset(
        name="deepseek-v4-flash",
        family="deepseek",
        hardware="2x-rtx-pro-6000-blackwell",
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        weights_gb=166.9,
        quant="fp8 + packed fp4 experts",
        context=262144,
        vllm_extras=("b12x",),
        status=VERIFIED,
        no_thinking={"thinking": False},
        notes="Text-only sibling of the vision preset: same weights layout and flags. Verified on "
              "2x RTX PRO 6000 Blackwell WS with vllm 0.32.0 (2026-10-28): 5/5 answer checks, 671 tok/s "
              "at 16 concurrent (256-token answers), 1319 tok/s out at 1k in / 8k out, PCIe clean.",
        args=(
            "--trust-remote-code",
            "--tensor-parallel-size",
            "2",
            "--max-model-len",
            "262144",
            "--max-num-seqs",
            "16",
            "--max-num-batched-tokens",
            "4096",
            "--gpu-memory-utilization",
            "0.95",
            "--kv-cache-dtype",
            "fp8",
            "--block-size",
            "256",
            "--async-scheduling",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--tokenizer-mode",
            "deepseek_v4",
            "--tool-call-parser",
            "deepseek_v4",
            "--reasoning-parser",
            "deepseek_v4",
            "--enable-auto-tool-choice",
            "--enable-prompt-tokens-details",
            "--enable-force-include-usage",
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}',
            "--attention-backend",
            "B12X",
            "--moe-backend",
            "b12x",
        ),
    ),
    Preset(
        name="qwen3.5-122b-a10b-nvfp4",
        family="qwen",
        hardware="2x-rtx-pro-6000-blackwell",
        model="nvidia/Qwen3.5-122B-A10B-NVFP4",
        weights_gb=83.5,
        quant="nvfp4 (modelopt)",
        context=262144,
        no_thinking={"enable_thinking": False},
        status=VERIFIED,
        default=True,
        notes="NVFP4 runs natively on Blackwell; leaves ~90 GB for KV cache.",
        args=(
            *_PRO6000,
            "--trust-remote-code",
            "--quantization",
            "modelopt_fp4",
            "--max-model-len",
            "262144",
            "--max-num-seqs",
            "32",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
        ),
    ),
    Preset(
        name="qwen3.8-flash-next-nvfp4",
        family="qwen",
        hardware="2x-rtx-pro-6000-blackwell",
        model="nvidia/Qwen3.8-Flash-Next-NVFP4",
        weights_gb=132.7,
        quant="nvfp4 (modelopt)",
        context=262144,
        tools=False,
        no_thinking={"enable_thinking": False},
        notes="Newest Qwen that fits; MTP speculative decoding as on the model card. The card configures no tool parser.",
        args=(
            *_PRO6000,
            "--trust-remote-code",
            "--quantization",
            "modelopt",
            "--max-model-len",
            "262144",
            "--max-num-seqs",
            "16",
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":1}',
            "--reasoning-parser",
            "qwen3",
        ),
    ),
    Preset(
        name="qwen3.6-35b-a3b-fp8",
        family="qwen",
        hardware="2x-rtx-pro-6000-blackwell",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        weights_gb=37.5,
        quant="fp8",
        context=262144,
        no_thinking={"enable_thinking": False},
        notes="Small and fast: one full copy per GPU (data parallel) for the most requests per second.",
        args=(
            "--data-parallel-size",
            "2",
            "--gpu-memory-utilization",
            "0.92",
            "--kv-cache-dtype",
            "fp8",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--enable-prompt-tokens-details",
            "--max-model-len",
            "262144",
            "--max-num-seqs",
            "64",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
        ),
    ),
    Preset(
        name="glm-4.5-air-fp8",
        family="glm",
        hardware="2x-rtx-pro-6000-blackwell",
        model="zai-org/GLM-4.5-Air-FP8",
        weights_gb=112.6,
        quant="fp8 (compressed-tensors)",
        context=131072,
        no_thinking={"enable_thinking": False},
        notes="Largest GLM that fits; GLM-5.x (>= 204 GB even as NVFP4) does not.",
        args=(
            *_PRO6000,
            "--max-model-len",
            "131072",
            "--max-num-seqs",
            "32",
            "--tool-call-parser",
            "glm45",
            "--reasoning-parser",
            "glm45",
            "--enable-auto-tool-choice",
        ),
    ),
    Preset(
        name="glm-4.7-flash",
        family="glm",
        hardware="2x-rtx-pro-6000-blackwell",
        model="zai-org/GLM-4.7-Flash",
        weights_gb=62.4,
        quant="bf16",
        context=131072,
        no_thinking={"enable_thinking": False},
        notes="Parsers and MTP as on the model card (glm47 tools, glm45 reasoning).",
        args=(
            *_PRO6000,
            "--max-model-len",
            "131072",
            "--max-num-seqs",
            "32",
            "--speculative-config.method",
            "mtp",
            "--speculative-config.num_speculative_tokens",
            "1",
            "--tool-call-parser",
            "glm47",
            "--reasoning-parser",
            "glm45",
            "--enable-auto-tool-choice",
        ),
    ),
    Preset(
        name="inkling-2-mini",
        family="inkling",
        hardware="1x-rtx-pro-6000-blackwell",
        model=INKLING,
        revision=INKLING_REVISION,
        weights_gb=20.0,
        quant="nvfp4 + mxfp8 (as released)",
        context=262144,
        container=Container(INKLING_IMAGE),
        notes="Hugging Face's best single-RTX-PRO-6000 config: DSpark k=7 probabilistic, 190.6 tok/s "
              "one stream, 716 tok/s at 8 streams, 20.7K tok/s prefill (thinking on, 512-token answers). "
              "262K context: the default 1M needs 52.6 GiB of KV cache; 0.85 memory leaves room for the "
              "drafter. Reasoning cannot be switched off fully (model card).",
        sources=("https://huggingface.co/blog/inkling-2-mini-inference-hillclimb",
                 "https://vllm.ai/blog/2026-11-20-inkling2-mini"),
        args=(
            "--max-model-len", "262144",
            "--gpu-memory-utilization", "0.85",
            "--reasoning-parser", "inkling",
            "--tool-call-parser", "inkling",
            "--enable-auto-tool-choice",
            "--speculative-config", _DSPARK7,
        ),
    ),
    Preset(
        name="inkling-2-mini-5090",
        family="inkling",
        hardware="1x-rtx-5090",
        model=INKLING,
        revision=INKLING_REVISION,
        weights_gb=20.0,
        quant="nvfp4 + mxfp8 (as released)",
        context=65536,
        status=VERIFIED,
        container=Container(INKLING_IMAGE),
        notes="The RTX PRO 6000 recipe on a 32 GB card, the peer of the Radeon AI PRO R9700. KV cache "
              "is tight: 64K of context needs 6.66 GiB, which fits only with 16 sequences at most "
              "(vLLM's default 256 reserves ~1.4 GiB more for activations and CUDA graphs) and 0.94 "
              "of the card. Images built for CUDA 13.2 run on R580 drivers with "
              "NVIDIA_DISABLE_REQUIRE=1. Reasoning cannot be switched off fully (model card).",
        sources=("https://huggingface.co/blog/inkling-2-mini-inference-hillclimb",),
        args=(
            "--max-model-len", "65536",
            "--max-num-seqs", "16",
            "--gpu-memory-utilization", "0.94",
            "--reasoning-parser", "inkling",
            "--tool-call-parser", "inkling",
            "--enable-auto-tool-choice",
            "--speculative-config", _DSPARK7,
        ),
    ),
    Preset(
        name="inkling-2-mini-r9700",
        family="inkling",
        hardware="1x-radeon-ai-pro-r9700",
        model=INKLING,
        revision=INKLING_REVISION,
        weights_gb=20.0,
        quant="nvfp4 + mxfp8 (as released)",
        context=262144,
        tools=False,  # the image's own command decides the parsers
        container=_paiton(f"{PAITON}:inkling-2-mini-rdna4-v1.2.0"),
        notes="Eliovp Paiton image for RDNA4, exact model output: 93.9 tok/s one stream, 249.7 at 4 "
              "streams, 3.9K tok/s prefill at 8K; peak 29.9 GiB of the card's 32 GB. The image fetches "
              "its own weights into its cache.",
        sources=("https://huggingface.co/EliovpAI/Inkling-2-Mini-W3-INT3-Paiton-RDNA4",
                 "https://github.com/Eliovp-BV/paiton-vllm-plugin"),
        args=(),
    ),
    Preset(
        name="inkling-2-mini-r9700-w3",
        family="inkling",
        hardware="1x-radeon-ai-pro-r9700",
        model=INKLING,
        revision=INKLING_REVISION,
        weights_gb=20.0,
        quant="3-bit routed experts (Paiton W3), 8-bit attention",
        context=262144,
        tools=False,
        container=_paiton(PAITON_W3_IMAGE),
        notes="Faster Paiton start, not the exact model output: 179.7 tok/s one stream, 400.5 at 4 "
              "streams, 5.7K tok/s prefill at 8K (34/35 greedy on Eliovp's task set, same as default).",
        sources=("https://huggingface.co/EliovpAI/Inkling-2-Mini-W3-INT3-Paiton-RDNA4",),
        args=(),
    ),
)
# fmt: on


def by_name(name: str) -> Preset | None:
    return next((p for p in PRESETS if p.name == name), None)


def match_model(model_id: str) -> Preset | None:
    """The preset a running server's model id belongs to: preset name, HF id, or the HF id's
    last part (vLLM's default served name), case-insensitive."""
    want = model_id.strip().lower()
    for p in PRESETS:
        if want in (p.name.lower(), p.model.lower(), p.model.rsplit("/", 1)[-1].lower()):
            return p
    return None


@dataclass(frozen=True)
class Gpu:
    name: str
    mem_gib: float
    compute_cap: str


def matching_hardware(gpus: list[Gpu]) -> Hardware | None:
    """The largest profile these GPUs satisfy (enough matching cards, memory, architecture)."""
    for hw in sorted(HARDWARE.values(), key=lambda h: -h.gpus):
        ok = [
            g
            for g in gpus
            if re.search(hw.gpu_name, g.name)
            and g.mem_gib >= hw.min_mem_gib
            and g.compute_cap == hw.compute_cap
        ]
        if len(ok) >= hw.gpus:
            return hw
    return None


def covers(have: Hardware | None, want: Hardware) -> bool:
    """A machine of profile ``have`` runs presets made for ``want``: same cards, no more of them
    (a single-card preset runs on a 2-card rig)."""
    return bool(
        have
        and have.gpu_name == want.gpu_name
        and have.compute_cap == want.compute_cap
        and have.min_mem_gib >= want.min_mem_gib
        and have.gpus >= want.gpus
    )


def for_hardware(hw: Hardware | None) -> list[Preset]:
    """Presets this profile runs: its own first, then those for fewer of the same cards."""
    fits = [p for p in PRESETS if covers(hw, HARDWARE[p.hardware])]
    return sorted(fits, key=lambda p: hw is None or p.hardware != hw.id)
