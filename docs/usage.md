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

Serial numbers, MAC addresses and IPs are deliberately not collected.

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
- All drivers use the **open kernel modules**. RTX 50xx (Blackwell) requires them, and NVIDIA recommends them for Ampere (A6000) as well.
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

Use it for problems that only show under real jobs after a while. Example: a customer reports GPUs dropping from Gen4 to Gen3 after some time. Run `sudo hwb monitor --duration 3600` during their workload. If a busy GPU's link drops a generation while its error count rises and temperatures climb, that's a marginal PCIe channel (riser, retimer, slot, contacts) losing margin as the chassis warms. CPU throttling itself doesn't change link speed.

### `serve` 🚧

```
hwb serve [--model HF_ID] [--tp N] [--port 8000]
```

Not implemented yet: it reports SKIP. Planned behavior:
1. Install vLLM into an isolated venv.
2. Serve `--model` with tensor parallelism across `--tp` GPUs (default: all of them).
3. Run a load test and record throughput, TTFT and inter-token latency.
4. Always shut the server down afterwards.

### `all`

```
hwb all [--duration SECONDS] [--interval SECONDS] [--model HF_ID]
```

Runs `info`, then `bench all`, plus `serve` if `--model` is given. Everything goes into a single run directory and report.

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

1. **Sanitize, on the rig.** The host name becomes a stable alias (`rig-1a2b3c4d`). GPU UUIDs, serials, MAC/IP/email addresses, home directory paths and Xid process names are removed. The gateway checks all of this again.
2. **Preview.** hwb lists the runs, how the rig will appear, what was removed and the destination.
3. **Confirm.** Nothing is sent until you answer `y`. Without a terminal, `--yes` is required. `--dry-run` writes the exact upload to `<out>/submission-preview.json` and sends nothing.
4. **Upload.** You get `https://<gateway>/r/<uuid>` and a QR code. The report is unlisted: only people with the link can open it.

The delete token is appended to `<out>/submissions.jsonl` (mode 0600). `hwb submit --delete <id>` removes the report from the gateway. The gateway also deletes reports after its retention period (90 days by default).

| env | |
|---|---|
| `HWB_GATEWAY` | another gateway (`https://…`; `http://` only for localhost), or `off` to disable sharing and the prompt |
| `HWB_GATEWAY_TOKEN` | only for a private gateway, or to skip the public gateway's rate limit |

Limits on the public gateway: uploads up to 512 kB (hwb refuses bigger ones before sending; use `--run` to send fewer runs), 30 uploads per hour per address, and 90 days of retention. When a limit is hit, hwb says how long to wait.

When hwb runs in a terminal, every command offers to submit when it finishes. Answering anything but `y` skips it, and the command's exit code is unaffected. `HWB_GATEWAY=off` turns the offer off.

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
