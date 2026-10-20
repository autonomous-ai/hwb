# Using hwb

```
hwb [--version] [-v] [--out DIR] <command> [options]
```

The examples write the command as `hwb`. Substitute `python3 hwb.pyz` or `./hwb.sh`, depending on [how you installed it](install.md).

**Global options go before the command:** `hwb -v --out /data/runs detect` works, `hwb detect -v` does not.

| option | default | meaning |
|---|---|---|
| `-v`, `--verbose` | off | debug logging on stderr, including every command executed |
| `--out DIR` | `./hwb-runs` | base directory for run output |
| `--version` | | print version and exit |

## Recommended workflow for a fresh server

```bash
# 1. What is in the box?
hwb info

# 2. Is the GPU stack healthy?
hwb detect

# 3. If detect FAILs on the driver: review the plan, apply it, reboot, re-check
hwb install driver
sudo hwb install driver --apply
sudo reboot
hwb detect

# 4. Benchmark tools, then benchmarks (benchmarks are work in progress)
sudo hwb install tools --apply
hwb bench all --duration 300

# 5. Real-model smoke test (work in progress)
hwb serve --model Qwen/Qwen3-8B

# Or run everything in one go:
hwb all --duration 300 --model Qwen/Qwen3-8B
```

Attach `hwb-runs/report.html` to the server's acceptance ticket. It holds every run in that folder, with an overview tab. Paste `report.md` from a run directory if the ticket needs plain text.

---

## Commands

### `info`

Collects the hardware inventory without changing anything:

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
hwb info
sudo hwb info        # adds DIMM details
```

### `detect`

Runs the GPU driver and CUDA health checks. It's read-only. See the [check reference](checks.md) for what each result means and how to fix it.

```bash
hwb detect
sudo hwb detect      # also scans the kernel log for Xid errors
```

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

Steps run in order and **stop at the first failure**. The failing step's stderr is shown in the result, and the full output is in `hwb.log.jsonl`. After `install driver --apply`, reboot and run `hwb detect`.

Notes:
- All drivers use the **open kernel modules**. RTX 50xx (Blackwell) requires them, and NVIDIA recommends them for Ampere (A6000) as well.
- The CUDA toolkit (`nvcc`) is **optional**. PyTorch and vLLM wheels bundle their own CUDA runtime and only need the driver.
- On Debian with Secure Boot on, enroll the DKMS MOK key (`mokutil --import`) or disable Secure Boot.
- Omarchy usually installs the NVIDIA driver during OS setup. Run `hwb detect` before installing anything.

### `bench` 🚧

```
hwb bench {cpu,memory,disk,thermal,gpu,all} [--duration SECONDS] [--path DIR]
```

Not implemented yet: every kind reports SKIP. Planned behavior:
- **cpu, memory:** stress-ng / sysbench, measured per NUMA node.
- **disk:** fio on a *file* in `--path` (default `/var/tmp`). It never writes to raw devices.
- **thermal:** sustained CPU+GPU load for `--duration`. FAILs on thermal throttling or Xid errors.
- **gpu:** GEMM TFLOPS per dtype, memory bandwidth, PCIe host↔device bandwidth, GPU-to-GPU P2P and NCCL all-reduce. Runs in an isolated uv environment.

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
hwb all [--duration SECONDS] [--model HF_ID]
```

Runs `info`, `detect`, then `bench all`, plus `serve` if `--model` is given. Everything goes into a single run directory and report.

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

Use these in scripts, for example `hwb detect || alert "rig01 GPU stack broken"`.

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
