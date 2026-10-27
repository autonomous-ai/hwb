# Using hwb

```
hwb [--version] [-v] [--out DIR] [command] [options]
```

With no command, `hwb` runs `info`: the GPU driver checks plus the hardware inventory.

The examples write the command as `hwb`. Substitute `python3 hwb.pyz` or `./hwb.sh`, depending on [how you installed it](install.md).

**Global options go before the command:** `hwb -v --out /data/runs info` works, `hwb info -v` does not.

| option | default | meaning |
|---|---|---|
| `-v`, `--verbose` | off | debug logging on stderr, including every command executed |
| `--out DIR` | `./hwb-runs` | base directory for run output |
| `--version` | | print version and exit |

## Recommended workflow for a fresh server

```bash
# 1. Is the GPU stack healthy, and what is in the box?
sudo hwb

# 2. If the driver checks FAIL: review the plan, apply it, reboot, re-check
hwb install driver
sudo hwb install driver --apply
sudo reboot
sudo hwb

# 3. Benchmark tools (optional: stress-ng makes CPU scores comparable), then benchmarks
sudo hwb install tools --apply
sudo hwb bench cpu
sudo hwb bench thermal --duration 600

# 5. Real-model smoke test (work in progress)
hwb serve --model Qwen/Qwen3-8B

# Or run everything in one go:
hwb all --duration 300 --model Qwen/Qwen3-8B
```

Attach `hwb-runs/report.html` to the server's acceptance ticket. It holds every run in that folder, with an overview tab. Paste `report.md` from a run directory if the ticket needs plain text.

---

## Commands

### `info` (default)

Read-only. First the GPU driver and CUDA health checks (see the [check reference](checks.md) for what each result means and how to fix it), then the hardware inventory:

- **system:** OS, kernel, uptime
- **board:** vendor, model, BIOS version and date
- **cpu:** model, sockets, cores and threads, ISA flags (AVX-512, AMX), governor, max clock
- **memory:** total, swap, hugepages; per-DIMM size, type, speed and part number when run as root with `dmidecode`
- **numa:** nodes and their CPU lists
- **disks:** name, model, size, rotational, transport
- **network:** physical NICs, link state, speed, driver
- **sensors:** all hwmon temperatures
- **gpus:** nvidia-smi inventory
- **workloads:** what's already running on the GPUs.
  - Each GPU process is traced to the model server that started it (vLLM, SGLang, llama.cpp, Ollama, TGI, Triton), giving its model, served name, port, GPUs and memory.
  - Each server's own `/v1/models` on localhost is asked what it serves.
  - Full command lines are never stored, because they can contain API keys; local model paths are reduced to their last part.

MAC addresses, IPs and serial numbers are deliberately not collected. The one exception is each GPU's serial (from nvidia-smi, `gpu_serial`): bus IDs belong to slots, so the serial is what tells which physical card sat where across reports and swaps. It stays in shared reports.

```bash
hwb                  # same as: hwb info
sudo hwb             # adds DIMM details and scans the kernel log for Xid errors
```

It exits 1 when a check FAILs (see [exit codes](#exit-codes)), so `hwb || alert ...` works in scripts.

### `install`

Prints the install commands for this distro. **By default it changes nothing.**

```
hwb install {driver,cuda,tools} [--apply] [-y] [--branch N] [--cuda-version X.Y]
```

| target | Ubuntu | Debian | Arch / Omarchy |
|---|---|---|---|
| `driver` | `linux-headers-$(uname -r)` + `nvidia-driver-<branch>-server-open` (signed, works with Secure Boot) | headers + NVIDIA CUDA repo keyring + `nvidia-open` | `pacman -Syu` + `linux*-headers` + `nvidia-open-dkms nvidia-utils` |
| `cuda` | NVIDIA repo + `cuda-toolkit[-X-Y]` | same | `cuda` |
| `tools` | `fio stress-ng sysbench lm-sensors pciutils dmidecode nvtop curl` | same | same, with `lm_sensors` |

| option | meaning |
|---|---|
| `--apply` | actually run the steps. Requires root on Linux |
| `-y`, `--yes` | skip the `Proceed? [y/N]` prompt. Required when stdin is not a terminal |
| `--branch N` | Ubuntu driver branch (default `580`) |
| `--cuda-version X.Y` | pin a toolkit version, e.g. `12.8` (default: the latest in the repo) |

Steps run in order and **stop at the first failure**. The failing step's stderr is shown in the result, and the full output is in `hwb.log.jsonl`. After `install driver --apply`, reboot and run `hwb` again.

Notes:
- All drivers use the **open kernel modules**. Blackwell (RTX 50xx, RTX PRO 6000 Blackwell) requires them, and NVIDIA recommends them for Ampere (RTX A6000) as well.
- The CUDA toolkit (`nvcc`) is **optional**. PyTorch and vLLM wheels bundle their own CUDA runtime and only need the driver.
- On Debian with Secure Boot on, enroll the DKMS MOK key (`mokutil --import`) or disable Secure Boot.
- Omarchy usually installs the NVIDIA driver during OS setup. Run `hwb` before installing anything.

### `bench`

```
hwb bench {cpu,memory,disk,thermal,gpu,all} [--duration SECONDS] [--interval SECONDS] [--path DIR]
```

While a benchmark runs, a sampler records a time series every `--interval` seconds (default 1):
- **CPU:** utilisation, average clock, package temperatures (hwmon `coretemp`/`k10temp`), package power (RAPL, needs root) and the kernel's thermal-throttle counters.
- **System:** memory use and load average.
- **GPUs:** temperature, utilisation, power, SM clock and throttle reasons, from `nvidia-smi -lms`.

Every sample goes to `bench-<kind>.csv` and `bench-<kind>-gpu.csv` in the run directory. The report keeps up to 300 points per series (temperatures keep their peaks). The viewer's **Benchmarks** tab draws them as charts, with each phase shaded.

| kind | what it does | judged on |
|---|---|---|
| `cpu` | idle baseline → single-thread → all-core for `--duration` → cooldown | throughput (bogo ops/s with `stress-ng`, else Python loop iterations/s), scaling, clock, power, peak temp. WARN if the CPUs are under 90% busy at full load, or on throttle events |
| `thermal` | idle → all-core load for `--duration` (use 300–900 s) → cooldown | FAIL within 3 °C of the sensor's limit (Intel 100 °C, or `temp*_crit`; AMD Tctl 95 °C). WARN within 10 °C, on throttle events, on ≥10% clock sag over the run, or on GPU thermal/hardware slowdowns. A temperature still rising at the end is noted: run longer |
| `memory`, `disk`, `gpu` | not implemented yet (SKIP) | planned: stress-ng/sysbench per NUMA node; fio on a *file* in `--path`, never raw devices; GEMM/bandwidth/P2P/NCCL in an isolated uv venv |

Load comes from `stress-ng --cpu-method matrixprod` when it is installed (`hwb install tools`). Without it, a stdlib fallback runs one busy Python process per thread: the results are still valid for temperatures and stability, but the scores only compare with other fallback runs. All load processes run in their own process group and are always stopped, including on Ctrl-C. `thermal` monitors GPUs but doesn't load them; GPU load comes with `bench gpu`.

**GPU PCIe links** are recorded too, for each GPU the host driver owns:
- **Link state:** the generation and width the kernel negotiated (`current_link_speed`/`width` in sysfs, with `nvidia-smi` as a fallback).
- **Error counters:** the kernel's PCIe error counts (AER) for the GPU and for its upstream port.
- **Replays:** the GPU's packet-retransmission count from `nvidia-smi -q`.

Judged after the run:
- **WARN** when a link runs below its maximum generation *while that GPU is busy* (≥ 50% utilisation). Idle GPUs drop to Gen1 on purpose, so that's ignored.
- **WARN** when a link narrows, or correctable errors or replays rise.
- **FAIL** on uncorrectable errors.

The Benchmarks tab adds a link-generation chart, an error chart and a per-GPU table with the address, upstream port and counters.

**Run benchmarks on an otherwise idle machine.** Other workloads compete for the CPUs and heat them, which skews both the scores and the temperatures.

### `monitor`

```
hwb monitor [--duration SECONDS] [--interval SECONDS]
```

Records the same time series as the benchmarks (CPU and GPU temperature, utilisation, clocks, power, GPU PCIe links) for `--duration` seconds (default 600) **without adding any load**, while the machine runs its own workload. It applies the same temperature-limit, throttling, GPU-slowdown and PCIe judgements. Ctrl-C stops early and keeps what was recorded.

Use it for problems that only show under real jobs after a while. Example: a customer reports GPUs dropping a PCIe generation after some time (Gen5 → Gen4 on Blackwell cards, Gen4 → Gen3 on Ampere/Ada). Run `sudo hwb monitor --duration 3600` during their workload. If a busy GPU's link drops a generation while its error count rises and temperatures climb, that's a marginal PCIe channel (riser, retimer, slot, contacts) losing margin as the chassis warms. CPU throttling itself doesn't change link speed.

### `serve`

```
hwb serve --list
hwb serve [--preset NAME | --model HF_ID_OR_DIR] [--dry-run] [--install-vllm | --vllm PATH] [--force]
hwb serve --endpoint http://host:8000 [--model NAME] [--preset NAME]
          [--concurrency 1,4,16] [--requests 4] [--max-tokens 256] [--ready-timeout 3600]
          [--profiles chat,rag,reasoning,agent | all | IN/OUT]
```

Hosts a real model with vLLM, or tests one that's already running, and measures it. **Step-by-step guide: [Serving and testing a model](serve.md).**

1. **Answer checks**, run with reasoning off and temperature 0. They catch a server that's up but wrong: bad quantization, the wrong chat template, a missing tool-call or reasoning parser.
   - arithmetic (`17 × 23`);
   - JSON-only output;
   - writing a Python function;
   - recalling a code hidden in a ~6000-word context;
   - a tool call (`get_weather("Chicago")`), which only passes when the tool-call parser is set up.
2. **Load test** at each `--concurrency` level, sending `--requests × concurrency` distinct streaming requests with a fixed output length (`--max-tokens`, forced with vLLM's `ignore_eos`). It records:
   - output tokens/s in total and per request;
   - time to first token (p50/p90/p99);
   - time per output token;
   - the longest pause between chunks;
   - errors.

   This default sweep sends one-sentence prompts, so it measures decode. It stays the default, so runs compare across rigs and over time.
3. **Profiles** (`--profiles`, optional) repeat the load test at the same concurrency levels with the input and output lengths published benchmarks use (vLLM `bench serve`, NVIDIA, SemiAnalysis InferenceMAX):

   | profile | input / output tokens | stands for |
   |---|---|---|
   | `chat` | 1k / 1k | chat |
   | `rag` | 8k / 1k | long prompts: RAG, summaries, code review |
   | `reasoning` | 1k / 8k | long answers: reasoning, generation |
   | `agent` | 32k / 1k | a coding agent's context |

   `all` runs the four; `IN/OUT` such as `4k/512` sets your own. Each prompt is distinct text (no prefix-cache shortcut), sized with the server's own token counts so it is as long as the profile says. A profile that needs more context than the server allows is skipped. Profiles report input, output and total tokens/s besides the latency figures, as `serve.profile.<name>` results. Long prompts load prefill, and with tensor parallelism the PCIe links between the GPUs, harder than the default sweep.
4. **Sampling throughout:** the sampler records GPU temperature, power, clocks and PCIe links for every phase. The Benchmarks tab shows the checks, one load table per test (the default sweep, then each profile) and the charts. On the Overview, **All benchmark runs** lines up every run of the report, grouped by benchmark and, for serving, by model, one row per test.

**What it launches:**
- `--endpoint URL` tests that server and starts nothing. The model name it reports selects the matching preset's settings, such as how to switch reasoning off and whether a tool-call parser is expected. With no matching preset, both common switches are sent (`enable_thinking` and `thinking` set to false).
- Otherwise hwb first looks for a model server already running on the GPUs (see `info`). If it finds one and no `--preset` was given, **it tests that one rather than launching another**.
- To launch, it picks the preset for the detected hardware, or the one named with `--preset`. It starts `vllm serve` in its own process group on `127.0.0.1`, waits for `/health` (`--ready-timeout`), tests it, and **always stops it** afterwards.
- If other processes hold some GPUs, it launches on the free ones (`CUDA_VISIBLE_DEVICES`, numbered as in `nvidia-smi`). It refuses only when too few GPUs are free.
- `--dry-run` prints the exact command and where the weights come from, then stops.
- **Model weights:** hwb finds weights already on the machine (Hugging Face caches including other users', ModelScope, plain directories, plus Ollama and LM Studio as GGUF). With no `--preset`, it prefers a preset that's already on disk. `--model` launches any model found here without a preset. New downloads go to the shared Hugging Face cache (`HF_HOME`), after a free-space check. Details: [Serving and testing a model](serve.md#model-weights-reuse-whats-on-disk-share-whats-downloaded).

**vLLM** comes from `--vllm PATH`, then `$HWB_HOME/venvs/vllm`, then `PATH`. `--install-vllm` creates that venv with uv's own Python (it ships the headers Triton compiles against), installs the pinned version plus extras such as `b12x` for Blackwell kernels, pins the venv's pip `nvcc` to the CUDA version torch was built for, and fetches FlashInfer's prebuilt kernels (`vllm download-kernels`). Nothing is installed into the system or core Python. At launch hwb puts the venv's `bin` on `PATH`, points `CUDA_HOME` at the venv's pip toolkit when the rig has no CUDA toolkit or an older one (Ubuntu's `nvidia-cuda-toolkit` is CUDA 12.0, which knows no Blackwell), so JIT-compiled kernels (DeepGEMM, FlashInfer) build for the GPU. A `CUDA_HOME` you set yourself is kept. It also sets `VLLM_SKIP_P2P_CHECK=0`, so vLLM tests GPU peer-to-peer copies instead of trusting the driver. If vLLM fails to start, hwb prints the worker's own error under "Root cause", with a hint when it recognizes it. Hugging Face downloads use `HF_TOKEN` if it's set; the presets' models are public, so it's optional. `HWB_LLM_API_KEY` is sent as a Bearer token to servers that require one.

**Presets** (`hwb serve --list`) are per hardware profile:
- **"verified"** means the exact command ran on that hardware.
- **"untested"** means it was derived from the model card, the weight size and vLLM's source.

The presets for **2× RTX PRO 6000 Blackwell** (2 × 96 GB, sm_120):

| preset | model | weights | status |
|---|---|---|---|
| `deepseek-v4-flash-vision` | deepseek-ai/DeepSeek-V4-Flash-Vision-Exp @ `6821d6a` | 168 GB fp8 + fp4 experts | ran in production on another vLLM build; with vLLM 0.32.0 it starts and passes the answer checks without `--linear-backend b12x` |
| `deepseek-v4-flash` | deepseek-ai/DeepSeek-V4-Flash-0731 | 167 GB | same flags as the vision preset; untested |
| `qwen3.5-122b-a10b-nvfp4` (default) | nvidia/Qwen3.5-122B-A10B-NVFP4 | 84 GB | verified (886 tok/s at 16 concurrent) |
| `qwen3.8-flash-next-nvfp4` | nvidia/Qwen3.8-Flash-Next-NVFP4 | 133 GB | untested |
| `qwen3.6-35b-a3b-fp8` | Qwen/Qwen3.6-35B-A3B-FP8 (one copy per GPU) | 38 GB | untested |
| `glm-4.5-air-fp8` | zai-org/GLM-4.5-Air-FP8 | 113 GB | untested |
| `glm-4.7-flash` | zai-org/GLM-4.7-Flash | 62 GB bf16 | untested |

Models that don't fit 192 GB with room for KV cache are left out: GLM-5.x (≥ 204 GB even as NVFP4), Qwen3-235B FP8, DeepSeek V4.1/V4-Pro. Presets live in `src/hwb/serve/presets.py`. When an untested one has run cleanly on the hardware, mark it verified.

### `all`

```
hwb all [--duration SECONDS] [--interval SECONDS] [--preset NAME]
```

Runs `info`, then `bench all`, plus `serve` with that preset if `--preset` is given. Everything goes into a single run directory and report.

### `update`

```
hwb update [--apply] [-y]
```

Shows the installed and the latest release.
- **`--apply`** installs the newer release over the running `hwb`: download, verify `SHA256SUMS`, test-run the new file, then an atomic rename. It asks first; `-y` skips the prompt, and a non-interactive run needs it.
- **Permissions:** it needs write access to the file, so use `sudo` for `/usr/local/bin/hwb`.
- **Other installs:** it only works on the release install (see [installation](install.md#updating)).

Interactive runs of any command also check for a newer release, at most once a day, and print one line when there is one. `HWB_NO_UPDATE_CHECK=1` disables that.

### `submit`

```
hwb [--out DIR] submit [--gateway URL] [--label TEXT] [--run DIR ...] [-y] [--dry-run]
hwb [--out DIR] submit --delete ID [-y]
```

Shares the runs in `--out` through the public [hwb gateway](../gateway/README.md) (`https://hwb-gateway.autonomous-ai.workers.dev`, no account or token needed) and prints the link plus a QR code to scan. The page looks just like `report.html`.

1. **Sanitize, on the rig.** The host name becomes a stable alias (`rig-1a2b3c4d`). GPU UUIDs, serials (except GPU serials, kept to track a card), MAC/IP/email addresses, home directory paths and Xid process names are removed. The gateway checks all of this again.
2. **Preview.** hwb lists the runs, how the rig will appear, what was removed and the destination.
3. **Confirm.** Nothing is sent until you answer `y`. Without a terminal, `--yes` is required. `--dry-run` writes the exact upload to `<out>/submission-preview.json` and sends nothing.
4. **Upload.** You get a QR code with the link `https://<gateway>/r/<uuid>` printed right below it, so both stay on screen. The delete command and the image link come above the code. The report is unlisted: only people with the link can open it.
   - The QR code is drawn with colored spaces, so it scans on any terminal, including SSH sessions with a C locale, the Linux console and IPMI/KVM consoles. On a terminal too short for it, hwb draws a half-height version with Unicode half blocks if the terminal supports them.
   - On a monitor plugged straight into the rig (the Linux console, often only 33–67 lines tall) the code needs about 45 lines. hwb narrows the white border when that helps, and tells you when the screen is still too short: pick a smaller console font with `sudo dpkg-reconfigure console-setup` or open the image link.
   - Force a style with `HWB_QR=full|half|ascii|off`. The same code is always available as an image at `https://<gateway>/r/<uuid>/qr.svg`.

The delete token is appended to `<out>/submissions.jsonl` (mode 0600). `hwb submit --delete <id>` removes the report from the gateway. The gateway also deletes reports after its retention period (90 days by default).

| env | |
|---|---|
| `HWB_GATEWAY` | another gateway (`https://…`; `http://` only for localhost), or `off` to disable sharing and the prompt |
| `HWB_GATEWAY_TOKEN` | only for a private gateway, or to skip the public gateway's rate limit |

Limits on the public gateway: uploads up to 512 kB (hwb refuses bigger ones before sending; use `--run` to send fewer runs), 30 uploads per hour per address, and 90 days of retention. When a limit is hit, hwb says how long to wait.

When hwb runs in a terminal, every command offers to submit when it finishes. Answering anything but `y` skips it, and the command's exit code is unaffected. `HWB_GATEWAY=off` turns the offer off.

To submit every test without being asked, set `hwb config set submit auto` (below). This also works when nobody watches the terminal, e.g. a long `hwb serve` in tmux or nohup.

### `config`

```
hwb config                       # in a terminal: a menu to change the settings
hwb config show                  # print the settings and the file they live in
hwb config set KEY VALUE         # for scripts
hwb config unset KEY
```

The menu lists the settings with numbers. Type a number to change one: `submit` offers its three choices by number, and `label` and `gateway` take text (Enter keeps the value, `-` clears it). Each change is saved at once; `q` quits. Without a terminal, `hwb config` prints the settings like `show`.

| key | values | |
|---|---|---|
| `submit` | `ask` (default), `auto`, `off` | after a run: offer to submit (interactive runs only), submit without asking, or say nothing |
| `label` | text | label for submitted reports, e.g. a rack position (`--label` wins) |
| `gateway` | URL | gateway for submissions (`--gateway` and `HWB_GATEWAY` win) |

With `submit auto`, each finished run is sanitized and uploaded as soon as it ends, together with the newest `info` run in `--out` for the hardware context, and the link and QR code are printed. Runs stopped with Ctrl-C, `install` plans and `--dry-run` runs are not sent. Runs that ended in an error are sent, so someone off-site can see why. A failed upload (offline, rate limit) is reported and leaves the exit code alone; `hwb submit` sends it later.

The settings live in `~/.config/hwb/config.json` (`$XDG_CONFIG_HOME`, or the file `$HWB_CONFIG` names). Under `sudo`, hwb reads the invoking user's file when root has none, so `sudo hwb info` and `hwb serve` follow the same settings. `HWB_SUBMIT=ask|auto|off` overrides `submit` for one run, and `HWB_GATEWAY=off` still turns sharing off entirely.

---

## Reading the results

Each check produces one line:

```
PASS   gpu.kernel_module  nvidia 580.65.06 (open kernel module)
WARN   gpu.pcie_link      PCIe link narrower than supported: GPU1 x8/x16
                          ↳ check slot wiring/bifurcation and risers; ...
FAIL   gpu.nvidia_smi     nvidia-smi sees 1 of 2 PCI GPUs
```

| status | meaning |
|---|---|
| **PASS** | fine |
| **WARN** | works, but an operator should look at it |
| **FAIL** | server is not acceptable as-is. The `↳` hint says what to do |
| **SKIP** | not applicable, or a prerequisite is missing (no root, tool absent, feature not implemented) |
| **ERROR** | hwb itself crashed on this check. Says nothing about the hardware. Please report it with `hwb.log.jsonl` |

### Exit codes

| code | meaning |
|---|---|
| `0` | no FAIL and no ERROR (WARN and SKIP allowed) |
| `1` | at least one FAIL |
| `2` | hwb error: crashed check, bad environment, refused install, etc. |
| `130` | interrupted (Ctrl-C or SIGTERM). The report is still written |

Use these in scripts, for example `hwb || alert "rig01 GPU stack broken"`.

### Output files

`<--out>/report.html` is the **one** report page for that output directory, regenerated after every run from all `*/report.json` beside it. It is self-contained (inline CSS/JS, no network) and opens in any current browser with JavaScript on.

- **Run tabs** (top bar): **Overview** first, then each run, newest first. The overview keeps the latest result of each check across runs, says which run it came from, and lists the runs. Runs from several hosts in one folder get one overview per host.
- **Section tabs** (per run or overview): Summary (verdict, counts, "needs attention" with hints), Checks (filter by status and text, expandable data), Inventory, GPU driver, Install plan, Raw JSON (with download).
- The current tab lives in the URL (`report.html#/<run>/<tab>`), so links and the back button work. Printing includes every section of the current tab's run.
- Everything in that folder goes into the page. Use a separate `--out` per rig if you share the file outside your team.

Each run writes to `<--out>/<YYYYMMDDTHHMMSSZ>-<host>-<command>/`:

- `report.md`: the results table.
- `report.json`: `{schema, tool, command, host, started_at, finished_at, exit_code, results[], sections{}}`. `sections` holds the raw data: `inventory`, `gpu_facts` and `install_plan`.
- `hwb.log.jsonl`: one JSON object per line. Records with `"msg": "exec"` / `"exit"` form the audit trail of every external command: argv, return code, duration, and the last part of stdout/stderr.

Quick queries:

```bash
jq -r '.results[] | select(.status=="fail") | "\(.name): \(.summary)"' hwb-runs/*/report.json
jq -c 'select(.msg=="exit") | {argv, rc, duration_s}' hwb-runs/*/hwb.log.jsonl
```
