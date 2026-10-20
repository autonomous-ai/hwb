# hwb — GPU server acceptance toolkit

[![CI](https://github.com/autonomous-ai/hwb/actions/workflows/ci.yml/badge.svg)](https://github.com/autonomous-ai/hwb/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/autonomous-ai/hwb)](https://github.com/autonomous-ai/hwb/releases)

Run hwb on a freshly racked NVIDIA GPU server (Xeon with 2× RTX A6000, RTX 5090 or similar, on Ubuntu, Debian or Omarchy). It answers three questions:

1. **Is the GPU driver installed correctly?** (`hwb detect`)
2. **Is all the hardware there?** (`hwb info`)
3. **Does it perform, and does a real model serve?** (`hwb bench`, `hwb serve`, both work in progress)

The only thing hwb needs on the server is `python3` 3.10 or newer. Nothing is installed into the system Python.

## Quick start

```bash
# On the server: grab the latest single-file release
curl -fsSLO https://github.com/autonomous-ai/hwb/releases/latest/download/hwb.pyz
# (private repo: gh release download --repo autonomous-ai/hwb --pattern hwb.pyz)

python3 hwb.pyz detect               # is the GPU driver healthy?
python3 hwb.pyz info                 # hardware inventory

# No driver yet? Review the plan first, then apply it as root
python3 hwb.pyz install driver
sudo python3 hwb.pyz install driver --apply
sudo reboot
python3 hwb.pyz detect               # should now be all PASS/SKIP
```

Open **`./hwb-runs/report.html`**: one self-contained page for every run in that folder. It works
offline and needs no server, so you can attach it for the vendor. The **Overview** tab merges all runs
(the latest result of each check, with one verdict for the rig). Each run has its own tab, split into
Summary · Checks · Inventory · GPU driver · Install plan · Raw JSON.

Each run also writes its own files to `./hwb-runs/<UTC time>-<host>-<command>/`:

| file | contents |
|---|---|
| `report.md` | readable summary table: paste it into a ticket |
| `report.json` | machine-readable results and the full inventory |
| `hwb.log.jsonl` | audit log: every command hwb executed, with exit code, duration and output |

To share the results, run `hwb submit`. It uploads to the public [hwb gateway](gateway/README.md) (a Cloudflare Worker in this repo; no account needed). It removes host names, addresses, serials and paths, shows what will be sent, asks for confirmation, then prints a link and a QR code. Interactive runs offer this at the end; `HWB_GATEWAY=off` disables it.

## Documentation

- **[Installation](docs/install.md)**: getting hwb onto a server (bundle, git checkout, pip/uv), prerequisites per distro
- **[Usage](docs/usage.md)**: every command and flag, the recommended fresh-rig workflow, exit codes, output files
- **[Check reference](docs/checks.md)**: what each `detect` check means and how to fix a FAIL or WARN
- **[Development](docs/development.md)**: dev setup, tests, adding a check, project rules

## Status

| command | status |
|---|---|
| `detect` | ✅ driver, nouveau, PCI binding, Secure Boot, headers, nvidia-smi, Blackwell, PCIe width, persistence, libcuda, CUDA toolkit, Xid |
| `info` | ✅ system, board, CPU, memory/DIMMs, NUMA, disks, NICs, temperatures, GPUs |
| `install driver\|cuda\|tools` | ✅ dry-run plans, guarded `--apply` |
| `bench cpu\|memory\|disk\|thermal\|gpu` | 🚧 reports SKIP "not implemented yet" |
| `serve` (vLLM) | 🚧 reports SKIP "not implemented yet" |
