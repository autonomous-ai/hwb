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

## Method 1 — install a release (recommended)

One command downloads the latest release, verifies it against the release's `SHA256SUMS`, and installs it as an executable `hwb`:

```bash
curl -fsSL https://autonomous.ai/hwb-install.sh | sudo sh     # -> /usr/local/bin/hwb
curl -fsSL https://autonomous.ai/hwb-install.sh | sh          # -> ~/.local/bin/hwb, no root

sudo hwb        # driver checks + inventory
```

`https://autonomous.ai/hwb-install.sh` is a redirect to the latest release's installer. Use the GitHub URL directly if you prefer, or if a proxy only allows github.com:

```bash
curl -fsSL https://github.com/autonomous-ai/hwb/releases/latest/download/install.sh | sudo sh
```

The installer is a short POSIX `sh` script that ships with every release. Read it first if you like: `curl -fsSL https://autonomous.ai/hwb-install.sh | less`.
- **Prerequisites:** only `python3` ≥ 3.10. It downloads with `curl`, `wget`, or `python3` itself when neither is installed.
- **Safe to re-run:** running it again re-installs or updates in place.
- **Options**, set as environment variables (put them after `sudo` so they reach the script, e.g. `| sudo HWB_VERSION=v0.4.0 sh`):

| variable | default | meaning |
|---|---|---|
| `HWB_VERSION` | `latest` | release tag to install, e.g. `v0.4.0` |
| `HWB_INSTALL_DIR` | `/usr/local/bin` as root, else `~/.local/bin` | where `hwb` goes |
| `HWB_BASE_URL` | the GitHub release | a mirror or local directory (`file:///path`) holding `hwb.pyz` and `SHA256SUMS` |

**Optional: download the release by hand.** `hwb` is just the release's `hwb.pyz` with the executable bit set:

```bash
curl -fsSLO https://github.com/autonomous-ai/hwb/releases/latest/download/hwb.pyz
curl -fsSLO https://github.com/autonomous-ai/hwb/releases/latest/download/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
sudo install -m 0755 hwb.pyz /usr/local/bin/hwb      # or just run: sudo python3 hwb.pyz
```

Every release asset also carries a signed build-provenance attestation. To confirm a file was built by this repo's release workflow:

```bash
gh attestation verify hwb.pyz --repo autonomous-ai/hwb
```

### Updating

```bash
hwb update                 # installed vs latest version; changes nothing
sudo hwb update --apply    # download, verify SHA256, test-run, replace /usr/local/bin/hwb in place
```

**Update check:** after an interactive run, hwb looks up the latest release at most once a day.
- It's a 2-second HEAD request to GitHub's `/releases/latest`; nothing about the machine is sent.
- The result is cached in `~/.cache/hwb/update-check.json` (or `$HWB_HOME`).
- When a newer release exists, it prints one line with the command to update.
- Offline rigs never see an error.
- `HWB_NO_UPDATE_CHECK=1` turns the check off.

Running the installer again works too. `hwb update` only applies to the release install: a pip/uv install or a source checkout is updated the way it was installed.

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
sudo hwb install driver --apply   # release install (method 1): /usr/local/bin is on root's PATH
sudo python3 hwb.pyz install driver --apply
sudo ./hwb.sh
```

With method 3, run as root with the full path: `sudo "$(command -v hwb)"`.

Output from a root run lands in `./hwb-runs/` and is owned by root. Pass `--out /some/dir` to put it somewhere else.

## Air-gapped servers

`info`, `bench cpu`, `bench thermal` and the install *plan* need no network. `submit` needs HTTPS to the gateway (`hwb-gateway.autonomous-ai.workers.dev`, or your `HWB_GATEWAY`). Applying an install needs network: it needs the distro mirrors, plus `developer.download.nvidia.com` on Debian. The upcoming GPU benchmark and vLLM test need PyPI and Hugging Face, or an offline wheelhouse and model directory (planned).

## Uninstall

- Release install: `sudo rm /usr/local/bin/hwb` (or `rm ~/.local/bin/hwb`), and the update-check cache `rm -rf ~/.cache/hwb`
- Bundle: `rm hwb.pyz`
- Checkout: `rm -rf hwb/`
- Tool install: `uv tool uninstall hwb` (or `pipx uninstall hwb`)
- Run outputs: `rm -rf hwb-runs/`

hwb never changes the system unless you passed `install … --apply`. Packages installed that way are ordinary distro packages, and you remove them with apt or pacman.
