# Serving and testing a model

`hwb serve` answers two questions about a GPU server: does a real model run on it correctly, and how fast? It either tests a model server that's **already running**, or **launches** vLLM with a preset made for the hardware, tests it, and shuts it down again.

Full option reference: [usage → `serve`](usage.md#serve).

## 0. Install and look around

```bash
curl -fsSL https://autonomous.ai/hwb-install.sh | sudo sh
sudo hwb          # driver checks + inventory, including "Models on the GPUs"
hwb serve --list  # presets; "fits = yes" marks the ones for this machine
```

The inventory's **Models on the GPUs** section shows any model server already holding the GPUs (vLLM, SGLang, llama.cpp, Ollama, TGI, Triton), with its model, port, GPUs and memory.

## Model weights: reuse what's on disk, share what's downloaded

hwb has **no model store of its own**. It finds weights where the common tools keep them, and downloads into the cache those tools share, so each model is downloaded once per machine.

**Found automatically** (`hwb serve --list`, and "Model weights on disk" in `sudo hwb`):

| where | used by | runnable with vLLM? |
|---|---|---|
| Hugging Face cache: `$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`; **as root, every user's** | vLLM, SGLang, transformers, `hf download` | yes (safetensors) |
| ModelScope `~/.cache/modelscope/hub` | ModelScope, vLLM | yes, loaded by path |
| `$HWB_MODEL_DIRS` (colon-separated), `/models`, `/data/models`, `/opt/models` | your own copies | yes, if they hold safetensors + `config.json` |
| Ollama `~/.ollama/models`, `/usr/share/ollama/.ollama/models` | Ollama | no (GGUF): test through Ollama with `--endpoint` |
| LM Studio `~/.lmstudio/models`, llama.cpp `~/.cache/llama.cpp` | LM Studio, llama.cpp | no (GGUF): test through their server |

- **Each model is listed with** its size, architecture (from `config.json`; it tells a text LLM from, say, a music model) and whether the download is **complete**: every shard in the safetensors index is present, with no dangling links from an interrupted download.
- **Privacy:** reports never include the paths, which contain user names.

**How `hwb serve` uses them:**
- **No `--preset`:** for this hardware it prefers a preset whose weights are already complete on disk. The default comes first, then verified presets. If none is on disk, it uses the default, which downloads.
- **Already on disk:** vLLM loads that copy, even from another user's cache or a ModelScope or plain directory (by path), and nothing is downloaded. If the tokenizer is there too, `HF_HUB_OFFLINE=1` is set, so vLLM doesn't contact the Hub or write lock files into someone else's cache.
- **`hwb serve --model <HF id | directory>`** launches any model found on the machine without a preset. It uses as few GPUs as fit the weights in 85% of their memory, safe default flags, and no tool-call parser.
- **New downloads** go to the standard Hugging Face cache, shared with every other tool. Move it to a bigger disk with `HF_HOME=/data/hf` (vLLM, transformers and `hf` all follow it). Before downloading, hwb checks for free space and refuses rather than fail partway through, e.g. after 150 GB of a 168 GB model (`--force` overrides).
- **Run `hwb serve` without `sudo`.** It doesn't need root, and under `sudo` downloads would go to root's cache instead of yours. hwb warns when that happens.

To download a model ahead of time, into the same shared cache:
```bash
hf download zai-org/GLM-4.5-Air-FP8      # pip install -U huggingface_hub
```

## 1. A model is already running: test it

```bash
hwb serve                                    # finds the running server and tests it
hwb serve --endpoint http://127.0.0.1:8000   # or name it explicitly
hwb serve --endpoint http://other-host:8000  # a server on another machine
```

- **Nothing is restarted or reconfigured.** hwb only sends requests, but those requests do load the model fully, so don't run this while the server is handling real users.
- **The model name picks the preset's settings.** The name the server reports (e.g. `DeepSeek-V4-Flash-Vision-Exp`) matches a preset, which supplies how to switch reasoning off for the checks and whether a tool-call parser is expected. For models with no preset, hwb sends both common switches (`enable_thinking: false` and `thinking: false`). `--preset NAME` forces a particular preset.
- `HWB_LLM_API_KEY` is sent as a Bearer token if the server requires one.

## 2. Launch a preset

The GPUs must be free, so stop any running model server first. If other processes hold some of the GPUs, hwb uses only the free ones, and refuses if too few are left.

```bash
hwb serve --preset qwen3.5-122b-a10b-nvfp4 --dry-run        # print the exact vllm command, start nothing
hwb serve --preset qwen3.5-122b-a10b-nvfp4 --install-vllm   # first time: vLLM into ~/.cache/hwb/venvs/vllm
hwb serve --preset qwen3.5-122b-a10b-nvfp4 --vllm "$(command -v vllm)"   # or use the vLLM you already have
hwb serve                                                    # no preset: the verified one for this hardware
```

hwb then:
1. downloads the model on the first run (minutes to an hour, depending on its size; `HF_TOKEN` is used if set);
2. starts vLLM in its own process group on `127.0.0.1`;
3. waits up to `--ready-timeout` (default 1 hour) until `/health` answers, failing fast with the end of `vllm.log` if vLLM exits;
4. runs the checks and the load test;
5. **always stops vLLM**, including on Ctrl-C or SIGTERM.

## 3. What it measures

**Answer checks.** They run with reasoning off and temperature 0, so answers are short and deterministic. They catch a server that's up but wrong.

| check | passes when | a failure usually means |
|---|---|---|
| `arithmetic` | `17 × 23` → `391` | broken weights, quantization or chat template |
| `json` | JSON only, `city` = Paris | the template or the model ignores instructions |
| `code` | writes `def is_prime` | the same |
| `long_context` | finds a code hidden in ~6000 words | context or attention problems (`--max-model-len`, KV cache) |
| `tool_call` | calls `get_weather(city="Hanoi")` | wrong or missing `--tool-call-parser` / `--enable-auto-tool-choice` |

If an answer comes back empty because the model spent its whole token budget reasoning, the failure says so. Pass `--preset` so hwb knows how to switch reasoning off for that model.

**Load test.** At each `--concurrency` level (default 1, 4, 16) it sends `--requests × concurrency` distinct streaming requests, each producing exactly `--max-tokens` (default 256) output tokens. It records:
- output tokens/s in total, and per request;
- time to first token (TTFT) p50/p90/p99;
- time per output token (TPOT);
- the longest stall between chunks;
- errors.

The sampler records GPU temperature, power, clocks and PCIe links throughout.

```bash
hwb serve --concurrency 1,8,32,64 --requests 8 --max-tokens 512   # a heavier sweep
```

## 4. Read and share the results

- **Terminal:** `serve.checks` (passed or failed, with reasons), `serve.load`, for example "2100 tok/s at 16 concurrent, TTFT p50 900 ms, 140 tok/s per request", and `serve.pcie` for a local server: a GPU link below its generation while busy (with the cause when the BIOS limits the slot), a narrowed link, or PCIe errors.
- **`hwb-runs/report.html` → Benchmarks → Model serving:** the summary numbers, each check with the model's actual answer, a table per concurrency level, the exact vLLM command, and GPU temperature, power, clock and PCIe charts.
- **Run directory:** `vllm.log`, plus `serve.csv` and `serve-gpu.csv` with every sample.
- **`hwb submit`:** uploads a sanitized copy and gives you a link to share.

## 5. Presets

Presets live in `src/hwb/serve/presets.py`, one set per hardware profile.
- **verified** means the exact command has run on that hardware.
- **untested** means it was derived from the model card, the weight size and vLLM's source.

After a clean run of an untested preset (all checks pass, no load errors), change its `status` to `VERIFIED`, then release.
