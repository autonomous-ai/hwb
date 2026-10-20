# Installing hwb

hwb has **no Python package dependencies**. It needs only `python3` 3.10 or newer on the target server. Pick one of the three methods below.

## Prerequisites on the server

| distro | Python version shipped | if python3 is missing |
|---|---|---|
| Ubuntu 22.04 / 24.04 | 3.10 / 3.12 (preinstalled) | `sudo apt-get install -y python3` |
| Debian 12 / 13 | 3.11 / 3.13 (usually preinstalled) | `sudo apt-get install -y python3` |
| Omarchy / Arch | latest (preinstalled on Omarchy) | `sudo pacman -S --needed python` |

Check with `python3 --version`.

Optional tools make some results richer. hwb works without them and marks the affected checks SKIP:

| tool | used for | package |
|---|---|---|
| `dmidecode` (run as root) | per-DIMM memory details | `dmidecode` |
| `lsblk` | disk models and transport | `util-linux` (always present) |
| `journalctl` (run as root) | Xid error scan | systemd (always present) |
| `stress-ng` | comparable `bench cpu` / `bench thermal` scores (without it, a built-in Python load runs instead) | `hwb install tools` installs it |
| `fio`, `sysbench`, `lm-sensors`, `nvtop` | upcoming benchmarks, interactive monitoring | `hwb install tools` installs them |
| root (`sudo`) | CPU package power (RAPL) in benchmarks, DIMM details, Xid scan | |

`hwb install tools` prints the install commands for these, and `--apply` runs them (see [usage](usage.md#install)).

---

## Method 1 — download a release (recommended)

Every tagged release on [GitHub Releases](https://github.com/autonomous-ai/hwb/releases) ships `hwb.pyz` (the single-file bundle), a wheel, an sdist, and `SHA256SUMS`.

```bash
curl -fsSLO https://github.com/autonomous-ai/hwb/releases/latest/download/hwb.pyz
curl -fsSLO https://github.com/autonomous-ai/hwb/releases/latest/download/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS

sudo python3 hwb.pyz        # driver checks + inventory
```

Every release asset also carries a signed build-provenance attestation. To confirm a file was built by this repo's release workflow:

```bash
gh attestation verify hwb.pyz --repo autonomous-ai/hwb
```

Pin a specific version by replacing `latest/download` with `download/v0.3.0`.

## Method 1b — build the bundle yourself

Build the bundle on any machine with Python 3.10+. The server doesn't need git or network access.

```bash
# workstation, in the repo root
make bundle
```

Copy it over and run it:

```bash
scp dist/hwb.pyz admin@rig01:~
ssh admin@rig01
python3 hwb.pyz --version
chmod +x hwb.pyz && ./hwb.pyz            # also works: it has a shebang
```

The bundle is about 25 KB and needs no other files on the server. Rebuild it after you pull changes.

## Method 2 — git checkout + launcher

Use this when you want to tweak or debug hwb on the server itself.

```bash
git clone https://github.com/autonomous-ai/hwb.git && cd hwb
./hwb.sh --version
./hwb.sh
```

`hwb.sh` uses the newest `python3.13` … `python3.10` it finds and runs straight from `src/`, so nothing gets installed. If no suitable python3 exists but `uv` does, it falls back to a Python managed by uv.

## Method 3 — install as a command (`hwb` on PATH)

From a checkout:

```bash
uv tool install .          # or: pipx install .
hwb --version
```

Avoid `sudo pip install` into the system Python. On Debian and Ubuntu it's blocked by PEP 668 anyway.

---

## Running as root

Some steps need root: `install --apply`, the DIMM details and the Xid scan.

```bash
sudo python3 hwb.pyz install driver --apply
sudo ./hwb.sh
sudo hwb                   # with method 3, only if the uv/pipx bin dir is on root's PATH, which it usually isn't
```

With method 3, run as root with the full path: `sudo "$(command -v hwb)"`.

Output from a root run lands in `./hwb-runs/` and is owned by root. Pass `--out /some/dir` to put it somewhere else.

## Air-gapped servers

`info`, `bench cpu`, `bench thermal` and the install *plan* need no network. `submit` needs HTTPS to the gateway (`hwb-gateway.autonomous-ai.workers.dev`, or your `HWB_GATEWAY`). Applying an install needs network: it needs the distro mirrors, plus `developer.download.nvidia.com` on Debian. The upcoming GPU benchmark and vLLM test need PyPI and Hugging Face, or an offline wheelhouse and model directory (planned).

## Uninstall

- Bundle: `rm hwb.pyz`
- Checkout: `rm -rf hwb/`
- Tool install: `uv tool uninstall hwb` (or `pipx uninstall hwb`)
- Run outputs: `rm -rf hwb-runs/`

hwb never changes the system unless you passed `install … --apply`. Packages installed that way are ordinary distro packages, and you remove them with apt or pacman.
