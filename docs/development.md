# Developing hwb

Development happens on macOS or Linux, and hwb runs on Linux GPU servers. The project rules are in [`CLAUDE.md`](../CLAUDE.md). Read it before changing anything.

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs the dev tools and the right Python versions on its own.

```bash
make test        # pytest
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict
make check       # all three
make bundle      # dist/hwb.pyz
make clean
```

### Report viewer (`ui/`)

`report.html` is a React + TypeScript app built with Vite and
[`vite-plugin-singlefile`](https://github.com/richardtallent/vite-plugin-singlefile) into **one**
self-contained `src/hwb/assets/index.html`, with all JS and CSS inlined. That file is committed and
shipped as package data. At run time `core/report_html.py` gathers every `*/report.json` in the output directory and drops them into its
`<script id="hwb-report" type="application/json">` slot, so the rig, the pyz and the Python tests
never need Node. You need Node 22 only to change the viewer:

```bash
cd ui && npm ci
npm run dev        # live preview: all fixtures as one output directory (Overview + run tabs)
make ui-test       # tsc + vitest
make ui            # rebuild src/hwb/assets/index.html; commit it with your change
```

The fixtures in `ui/src/fixtures/` are real `report.json` files from the simulated rigs in
`tests/test_flow.py`, with the host name anonymised. CI rebuilds the viewer and fails if the
committed `index.html` is stale.

CI-style compatibility check against the oldest supported Python:

```bash
uv run --python 3.10 pytest -q
```

## CI and releases

GitHub Actions workflows live in `.github/workflows/`:

| workflow | trigger | what it does |
|---|---|---|
| `ci.yml` | push to `main`, every PR | ruff + mypy, pytest on Python 3.10–3.13, `make dist`, then a **smoke test** of `hwb.pyz` on the stock Python of `ubuntu:22.04`, `ubuntu:24.04`, `debian:12` and `archlinux` (runs `info`, a dry-run `install driver`, and `detect`, which must exit 1 with no GPU and must not exit 2) |
| `release.yml` | push of a `v*` tag | reruns CI, checks the tag matches the package version, builds `hwb.pyz`, the wheel, the sdist and `SHA256SUMS`, attests build provenance, and creates the GitHub Release with auto-generated notes. Tags like `v1.0.0-rc1` become pre-releases |

Dependabot (`.github/dependabot.yml`) bumps action versions weekly and dev tools monthly.

To cut a release:

```bash
# 1. bump the version in BOTH places
#    pyproject.toml           version = "0.2.0"
#    src/hwb/__init__.py      __version__ = "0.2.0"
# 2. merge to main, then tag the merge commit
git tag v0.2.0 && git push origin v0.2.0
```

If the tag and the versions disagree, the release job fails before it publishes anything. To build release artifacts locally, run `make dist`, which produces `dist/` with the same files.

Optional hardening in the repo settings:
- Add required reviewers to the `release` environment so publishing needs an approval.
- Add a tag protection rule for `v*`.

## Running locally

```bash
./hwb.sh -v --out /tmp/hwb-runs detect
```

On macOS everything degrades to SKIP or FAIL ("no NVIDIA GPU"). That's expected, and a useful check that nothing crashes off-box.

## Layout

```
src/hwb/
  cli.py             subcommands, exit codes, signal handling
  core/              proc (subprocess+audit), log, result, context, report, platform
  nvsmi.py           nvidia-smi wrapper
  detect/gpu.py      gather() -> GpuFacts; check_* pure functions; CHECKS tuple
  metrics/collect.py inventory collectors; COLLECTORS dict
  install/plan.py    per-distro Step lists; execute()
  bench/, serve/     work in progress
  assets/index.html  prebuilt report viewer (generated from ui/, do not edit)
tests/               parser and check tests with synthetic inputs
ui/                  report viewer source (React + Vite, single-file build)
```

`tests/test_flow.py` runs the real CLI end to end against a simulated rig: shell-script stand-ins for
`nvidia-smi`, `ldconfig`, `journalctl`, `nvcc` and `lsblk` on `PATH`, plus a fake root for `/proc`
and `/sys` reads. It checks every output file, including that the one `<out>/report.html` embeds
exactly each run's `report.json` and loads nothing external. Describe a new scenario with the `Rig` dataclass. When a new probe reads the
machine, route it through the fakes there, or the flow test will read the dev machine instead.

## Adding a `detect` check

1. If the check needs new data, add a field to `GpuFacts` and fill it in `gather()` (add an entry to `probes`).
2. Write `check_<name>(f: GpuFacts) -> CheckResult`. Keep it pure, with no I/O. Return FAIL/WARN with a `hint` the operator can act on.
3. Append it to `CHECKS`. Its result name becomes `gpu.<name>`.
4. Add a test in `tests/test_detect.py` with a synthetic `GpuFacts`.
5. Document it in [`docs/checks.md`](checks.md).

## Adding an inventory collector

Write `def foo() -> Inventory` in `metrics/collect.py` and register it in `COLLECTORS`. Prefer sysfs/procfs reads over CLI tools. Put text parsing in a separate `parse_*` function and test that.

## Rules of thumb

- Stdlib only in `src/hwb`. Heavy dependencies go into on-demand uv venvs.
- Write code that runs on Python 3.10.
- Spawn processes only through `hwb.core.proc.run`: argv list, timeout, audit log.
- Never change the system without `--apply`, root and confirmation.
- If you change the shape of `report.json`, bump `REPORT_SCHEMA` in `core/report.py`.
- Keep package names and driver branches as constants at the top of `install/plan.py`.
