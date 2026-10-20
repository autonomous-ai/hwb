# hwb — GPU server acceptance toolkit

[![CI](https://github.com/autonomous-ai/hwb/actions/workflows/ci.yml/badge.svg)](https://github.com/autonomous-ai/hwb/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/autonomous-ai/hwb)](https://github.com/autonomous-ai/hwb/releases)

Run hwb on a freshly racked NVIDIA GPU server (Xeon or Threadripper/EPYC with 2× RTX A6000, RTX 4090/5090 or similar, on Ubuntu, Debian or Omarchy). It answers three questions:

1. **Is the GPU driver stack healthy, and what hardware is in the box?** (`hwb`)
2. **Does it perform and stay cool under sustained load?** (`hwb bench cpu`, `hwb bench thermal`)
3. **Does a real model serve?** (`hwb serve`, work in progress)

The only thing hwb needs on the server is `python3` 3.10 or newer. Nothing is installed into the system Python, and nothing on the system changes unless you pass `--apply`.

## Quick start

```bash
# 1. Get the single-file release onto the server
curl -fsSLO https://github.com/autonomous-ai/hwb/releases/latest/download/hwb.pyz

# 2. Check the driver stack and collect the inventory (read-only; root adds DIMMs, Xid scan, CPU power)
sudo python3 hwb.pyz

# 3. No driver yet, or driver checks FAIL? Review the plan, apply it, reboot, check again
python3 hwb.pyz install driver
sudo python3 hwb.pyz install driver --apply && sudo reboot
sudo python3 hwb.pyz

# 4. Benchmark on an otherwise idle machine
sudo python3 hwb.pyz install tools --apply    # optional: stress-ng makes CPU scores comparable
sudo python3 hwb.pyz bench cpu
sudo python3 hwb.pyz bench thermal --duration 600

# 5. Share the results: sanitized, previewed, confirmed, then a link and a QR code
python3 hwb.pyz submit
```

Or run steps 2 and 4 in one go: `sudo python3 hwb.pyz all --duration 600`.

Open **`./hwb-runs/report.html`**: one self-contained page for every run in that folder. It works offline, so you can attach it to a ticket or send it to the vendor.
- **Overview tab:** merges all runs into the latest result of each check, with one verdict for the rig.
- **One tab per run**, split into Summary · Checks · Benchmarks · Inventory · GPU driver · Install plan · Raw JSON.
- **Benchmarks** charts temperatures, utilisation, clocks and power over time, for both CPU and GPUs.

Each run also writes its own files to `./hwb-runs/<UTC time>-<host>-<command>/`:

| file | contents |
|---|---|
| `report.md` | readable summary table: paste it into a ticket |
| `report.json` | machine-readable results, inventory and benchmark time series |
| `bench-*.csv` | every benchmark sample at full resolution (CPU, and GPUs in `*-gpu.csv`) |
| `hwb.log.jsonl` | audit log: every command hwb executed, with exit code, duration and output |

`hwb submit` uploads to the public [hwb gateway](gateway/README.md), a Cloudflare Worker in this repo. No account is needed.
- It removes host names, addresses, serials, GPU UUIDs and user paths.
- It shows what will be sent and asks for confirmation.
- It prints a link and a QR code.

Interactive runs offer to submit at the end; `HWB_GATEWAY=off` turns that off.

## Documentation

- **[Installation](docs/install.md)**: getting hwb onto a server (bundle, git checkout, pip/uv), prerequisites per distro
- **[Usage](docs/usage.md)**: every command and flag, the recommended fresh-rig workflow, exit codes, output files
- **[Check reference](docs/checks.md)**: what each driver check means and how to fix a FAIL or WARN
- **[Development](docs/development.md)**: dev setup, tests, adding a check, project rules

## Status

| command | status |
|---|---|
| `info` (default): driver checks | ✅ nouveau, kernel module, PCI binding, Secure Boot, headers, nvidia-smi, Blackwell, PCIe width, persistence, libcuda, CUDA toolkit, Xid |
| `info` (default): inventory | ✅ system, board, CPU, memory/DIMMs, NUMA, disks, NICs, temperatures, GPUs |
| `install driver\|cuda\|tools` | ✅ dry-run plans, guarded `--apply` |
| `bench cpu`, `bench thermal` | ✅ load (stress-ng or built-in fallback) with a time series of CPU and GPU temperature, utilisation, clocks and power; charts in the report |
| `monitor` | ✅ the same time series and judgements as the benchmarks, recorded while the machine runs its own workload; includes GPU PCIe links (generation while busy, width, error counters, replays) |
| `bench memory\|disk\|gpu` | 🚧 reports SKIP "not implemented yet" |
| `serve` (vLLM) | 🚧 reports SKIP "not implemented yet" |
| `submit` | ✅ sanitized upload to the public gateway, link and QR code, delete token |
