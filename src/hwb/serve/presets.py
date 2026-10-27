"""Known-good vLLM launch presets per hardware profile.

Presets are data: model, pinned revision where known, the exact `vllm serve` arguments, and
how sure we are. "verified" means the command ran on that hardware; "untested" means it was
derived from the model card, the weight size and vLLM's source, and still needs a real run.
Model ids, revisions and flags drift: keep them here, nowhere else (like install/plan.py).

Sizing rule for a profile: weights + KV cache + CUDA graphs must fit the GPUs' memory; on
2x RTX PRO 6000 (2 x 96 GB) the verified DeepSeek run carries 168 GB of weights, which is
about the ceiling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

VLLM_SPEC = "vllm==0.32.0"  # what the presets were checked against (vLLM main, 2026-10)

VERIFIED = "verified"
UNTESTED = "untested"


@dataclass(frozen=True)
class Hardware:
    id: str
    label: str
    gpu_name: str  # regex on nvidia-smi's name
    gpus: int
    min_mem_gib: float
    compute_cap: str


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
    """The profile these GPUs satisfy (enough matching cards, memory, architecture)."""
    for hw in HARDWARE.values():
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


def for_hardware(hw: Hardware | None) -> list[Preset]:
    return [p for p in PRESETS if hw and p.hardware == hw.id]
