"""End-to-end: run the real CLI against a simulated rig and check every output file.

The rig is faked at the OS boundary only: shell-script stand-ins for nvidia-smi & co. on
PATH (so core.proc, parsing and the audit log run for real) and a fake root for the
/proc, /sys reads. Works on macOS and Linux CI alike.
"""

from __future__ import annotations

import functools
import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from hwb import cli
from hwb.core.platform import OSInfo
from hwb.detect import gpu
from hwb.metrics import collect, sampler

UBUNTU = OSInfo("ubuntu", ("debian",), "24.04", "Ubuntu 24.04.1 LTS", "6.8.0-45-generic", "x86_64")

OPEN_580 = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.65.06  Release Build  "
    "(dvs-builder@U16-I3-B03-4-3)  Fri Jul 25 2025\nGCC version:  gcc version 13.3.0"
)
SMI_BANNER = "| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0     |\n"
# Columns follow nvsmi.GPU_FIELDS. GPU1 sits in an x8 slot and has persistence off.
SMI_CSV = (
    "0, NVIDIA GeForce RTX 5090, GPU-00000000-aaaa, [N/A], 00000000:01:00.0, 580.65.06, 32607, 12.0, Enabled, 5, 16, 16, 34, 575.00\n"
    "1, NVIDIA GeForce RTX 5090, GPU-00000000-bbbb, 1652724012345, 00000000:41:00.0, 580.65.06, 32607, 12.0, Disabled, 5, 8, 16, 71, 575.00\n"
)
CPUINFO = "\n\n".join(
    f"processor\t: {i}\nmodel name\t: Intel(R) Xeon(R) Gold 6430\nphysical id\t: {i % 2}\n"
    "cpu cores\t: 32\nflags\t\t: fpu sse avx2 avx512f amx_tile amx_bf16"
    for i in range(4)
)
MEMINFO = "MemTotal:       527958016 kB\nSwapTotal:        8388604 kB\nHugePages_Total:       0\n"
LSBLK = {
    "blockdevices": [
        {
            "name": "nvme0n1",
            "model": "SAMSUNG MZQL23T8HCLS",
            "size": 3840755982336,
            "rota": False,
            "tran": "nvme",
            "type": "disk",
        },
        {
            "name": "sr0",
            "model": "Virtual CD",
            "size": 1073741312,
            "rota": True,
            "tran": "usb",
            "type": "rom",
        },
    ]
}


@dataclass
class Rig:
    """What the simulated machine looks like; defaults are a healthy 2x RTX 5090 box."""

    modules: tuple[str, ...] = ("nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia")
    gpu_driver: str | None = "nvidia"
    driver_version_file: str | None = OPEN_580
    smi_ok: bool = True
    libcuda: bool = True
    xid: tuple[str, ...] = (
        "NVRM: Xid (PCI:0000:41:00): 13, pid=4242, name=python3, Graphics Exception",
    )


def _script(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


def _build(tmp: Path, rig: Rig) -> tuple[Path, Path]:
    """Create fake binaries and a fake root; returns (bin_dir, root)."""
    bin_dir, root, fx = tmp / "bin", tmp / "root", tmp / "fixtures"
    for d in (bin_dir, root, fx):
        d.mkdir()

    def put(rel: str, text: str) -> Path:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    drivers = tmp / "drivers"
    for addr in ("0000:01:00.0", "0000:41:00.0"):
        for fn, cls in ((".0", "0x030000"), (".1", "0x040300")):  # GPU + its HDMI audio
            dev = f"sys/bus/pci/devices/{addr[:-2]}{fn}"
            put(f"{dev}/vendor", "0x10de\n")
            put(f"{dev}/class", cls + "\n")
            put(f"{dev}/device", "0x2b85\n")
            if fn == ".0":  # link state as the kernel reports it
                put(f"{dev}/max_link_speed", "16.0 GT/s PCIe\n")
                put(f"{dev}/max_link_width", "16\n")
                gts = "8.0" if addr.startswith("0000:41") else "16.0"
                put(f"{dev}/current_link_speed", f"{gts} GT/s PCIe\n")
                put(f"{dev}/current_link_width", "16\n")
                put(f"{dev}/aer_dev_correctable", "RxErr 0\nBadTLP 0\nTOTAL_ERR_COR 0\n")
            if fn == ".0" and rig.gpu_driver:
                (drivers / rig.gpu_driver).mkdir(parents=True, exist_ok=True)
                (root / dev / "driver").symlink_to(drivers / rig.gpu_driver)
    put("proc/modules", "".join(f"{m} 106496 0 - Live 0x0\n" for m in rig.modules))
    if rig.driver_version_file:
        put("proc/driver/nvidia/version", rig.driver_version_file)
    put("proc/cpuinfo", CPUINFO)
    put("proc/meminfo", MEMINFO)
    put("proc/uptime", "93784.21 180000.00\n")
    put("proc/stat", "cpu  1000 0 500 8000 100 0 0 0 0 0\n")
    put("proc/loadavg", "0.50 0.40 0.30 1/900 4242\n")
    for i in range(2):  # one coretemp package sensor per socket
        hw = f"sys/class/hwmon/hwmon{i}"
        put(f"{hw}/name", "coretemp\n")
        put(f"{hw}/temp1_label", f"Package id {i}\n")
        put(f"{hw}/temp1_input", f"{48000 + i * 2000}\n")
        put(f"{hw}/temp1_crit", "100000\n")
    cpufreq = "sys/devices/system/cpu/cpu0/cpufreq"
    put(f"{cpufreq}/scaling_governor", "performance\n")
    put(f"{cpufreq}/cpuinfo_max_freq", "3400000\n")
    for k, v in {
        "sys_vendor": "Supermicro",
        "product_name": "SYS-421GE-TNRT",
        "bios_version": "2.1",
    }.items():
        put(f"sys/class/dmi/id/{k}", v + "\n")

    (fx / "smi.csv").write_text(SMI_CSV)
    (fx / "smi.txt").write_text(SMI_BANNER)
    (fx / "lsblk.json").write_text(json.dumps(LSBLK))
    (fx / "journal.txt").write_text("\n".join(("kernel: Linux version 6.8.0", *rig.xid)) + "\n")
    q = shlex.quote
    if rig.smi_ok:
        # -lms streams samples forever, like the real tool; the sampler must stop it.
        # (date has only whole seconds on macOS, so stream for a while like the real tool)
        lms = (
            "i=0; while [ $i -lt 15 ]; do "
            'now=$(date "+%Y/%m/%d %H:%M:%S.000"); '
            # GPU1 is busy at Gen3 although its slot does Gen4: the downtrained-link case.
            'echo "0, $now, 3, 41, 31.5, 210, 2, 1, 16, 0x0000000000000001"; '
            'echo "1, $now, 98, 63, 290.0, 1800, 40000, 3, 16, 0x0000000000000000"; '
            "i=$((i+1)); sleep 0.2; done; exec sleep 60"
        )
        busids = 'echo "0, 00000000:01:00.0"; echo "1, 00000000:41:00.0"'
        _script(
            bin_dir / "nvidia-smi",
            f'case "$*" in *-lms=*) {lms};; *index,pci.bus_id*) {busids};; '
            f"--query-gpu=*) cat {q(str(fx / 'smi.csv'))};; "
            f"*) cat {q(str(fx / 'smi.txt'))};; esac",
        )
    else:
        _script(
            bin_dir / "nvidia-smi",
            'echo "NVIDIA-SMI has failed because it couldn\'t communicate with the NVIDIA driver."; exit 9',
        )
    libcuda = (
        "\tlibcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1" if rig.libcuda else ""
    )
    _script(bin_dir / "ldconfig", f"echo {q('1 libs found in cache')}; echo {q(libcuda)}")
    _script(bin_dir / "journalctl", f"cat {q(str(fx / 'journal.txt'))}")
    _script(bin_dir / "nvcc", "echo 'Cuda compilation tools, release 12.8, V12.8.93'")
    _script(bin_dir / "lsblk", f"cat {q(str(fx / 'lsblk.json'))}")
    _script(
        bin_dir / "stress-ng",
        't=1; while [ $# -gt 0 ]; do [ "$1" = "--timeout" ] && t="${2%s}"; shift; done; sleep "$t"; '
        'echo "stress-ng: metrc: [4242] cpu   12345   $t.00   1.00   0.00   6172.50   6000.00"',
    )
    return bin_dir, root


def _rerooted(real: Callable[[Path], str | None], root: Path) -> Callable[[Path], str | None]:
    def read(path: Path) -> str | None:
        if path.is_absolute() and not path.is_relative_to(root):
            path = root / path.relative_to("/")
        return real(path)

    return read


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Install a simulated rig; returns a function that builds it and gives the --out dir."""

    def install(spec: Rig | None = None) -> Path:
        bin_dir, root = _build(tmp_path, spec or Rig())
        monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
        for mod in (gpu, collect):
            monkeypatch.setattr(mod, "_read", _rerooted(mod._read, root))
            monkeypatch.setattr(mod, "os_info", lambda: UBUNTU)
        monkeypatch.setattr(cli, "os_info", lambda: UBUNTU)
        monkeypatch.setattr(
            gpu, "scan_pci", functools.partial(gpu.scan_pci, root / "sys/bus/pci/devices")
        )
        monkeypatch.setattr(gpu, "secure_boot_enabled", lambda: True)
        monkeypatch.setattr(sampler, "ROOT", root)
        return tmp_path / "runs"

    return install


class _Page(HTMLParser):
    """Collects what the report page would load and the JSON embedded in it."""

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str]] = []
        self.data: str | None = None
        self._in_data = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        self.refs += [(tag, v) for k, v in a.items() if k in ("src", "href") and v]
        if tag == "link":
            self.refs.append((tag, a.get("href") or ""))
        self._in_data = tag == "script" and a.get("id") == "hwb-report"

    def handle_endtag(self, tag: str) -> None:
        self._in_data = False

    def handle_data(self, data: str) -> None:
        if self._in_data:
            self.data = (self.data or "") + data


def _page_runs(out: Path) -> list[dict[str, Any]]:
    """The runs embedded in the one combined <out>/report.html."""
    parser = _Page()
    parser.feed((out / "report.html").read_text())
    parser.close()
    # Self-contained: nothing is fetched (the rig may be offline).
    assert all(tag == "a" and "//" not in ref for tag, ref in parser.refs), parser.refs
    assert parser.data is not None
    runs: list[dict[str, Any]] = json.loads(parser.data)["runs"]
    return runs


def _outputs(out: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    (run_dir,) = (p for p in out.iterdir() if p.is_dir())
    names = {p.name for p in run_dir.iterdir()}
    assert {"report.json", "report.md", "hwb.log.jsonl"} <= names
    assert "report.html" not in names  # one combined page lives in <out>
    report = json.loads((run_dir / "report.json").read_text())
    runs = _page_runs(out)
    # The viewer gets exactly what report.json holds.
    assert runs == [{"id": run_dir.name, "report": report}]
    return run_dir, report, runs


def st_summary(report: dict[str, Any], name: str) -> str:
    return next(str(r["summary"]) for r in report["results"] if r["name"] == name)


def _statuses(report: dict[str, Any]) -> dict[str, str]:
    return {r["name"]: r["status"] for r in report["results"]}


def test_all_on_healthy_rig_with_warnings(rig: Callable[..., Path]) -> None:
    out = rig()
    code = cli.main(["--out", str(out), "all", "--duration", "1", "--interval", "0.25"])
    run_dir, report, _ = _outputs(out)
    st = _statuses(report)

    assert code == 0 == report["exit_code"]
    assert {n.split(".")[0] for n in st} == {"info", "gpu", "bench"}
    assert "error" not in st.values()
    assert all(st[f"info.{c}"] == "pass" for c in collect.COLLECTORS)
    for name in (
        "pci",
        "nouveau",
        "kernel_module",
        "bound",
        "secure_boot",
        "nvidia_smi",
        "blackwell",
        "libcuda",
        "cuda_toolkit",
    ):
        assert st[f"gpu.{name}"] == "pass", name
    assert st["gpu.pcie_link"] == st["gpu.persistence"] == st["gpu.xid"] == "warn"
    assert st["gpu.kernel_headers"] in ("pass", "warn")  # reads the real /lib/modules

    inv = report["sections"]["inventory"]
    assert inv["cpu"]["sockets"] == 2 and inv["cpu"]["physical_cores"] == 64
    assert inv["memory"]["total_gib"] == 503.5
    assert [d["name"] for d in inv["disks"]["devices"]] == ["nvme0n1"]
    assert len(inv["gpus"]["gpus"]) == 2
    assert report["sections"]["gpu_facts"]["kernel_driver"] == {
        "version": "580.65.06",
        "open_module": True,
    }

    assert "| WARN | gpu.pcie_link |" in (run_dir / "report.md").read_text()

    audit = [json.loads(line) for line in (run_dir / "hwb.log.jsonl").read_text().splitlines()]
    ran = {e["argv"][0] for e in audit if e.get("msg") == "exit"}
    assert {"nvidia-smi", "ldconfig", "journalctl", "nvcc", "lsblk", "stress-ng"} <= ran

    # Benchmarks: scores from stress-ng, time series with phases, CPU and GPU temperatures.
    # GPU1 is busy at Gen3 on a Gen4 link: both benchmarks flag it.
    assert st["bench.cpu"] == st["bench.thermal"] == "warn"
    assert "GPU1 PCIe link at Gen3 (max Gen4) while busy" in st_summary(report, "bench.thermal")
    assert "GPU0" not in st_summary(report, "bench.thermal")  # idle GPU0 at Gen1 is fine
    pcie = report["sections"]["bench"]["thermal"]["pcie"]
    assert pcie["1"]["addr"] == "0000:41:00.0" and pcie["1"]["max_gen"] == 4
    assert {"bench.memory", "bench.disk", "bench.gpu"} <= {n for n, s in st.items() if s == "skip"}
    cpu = report["sections"]["bench"]["cpu"]
    assert cpu["method"] == "stress-ng" and cpu["multi_ops_per_s"] == 6172.5
    ts = cpu["timeseries"]
    assert [p["name"] for p in ts["phases"]] == ["idle", "single-core", "all-core", "cooldown"]
    keys = {s["key"] for s in ts["series"]}
    assert {"cpu.temp.Package id 0", "cpu.temp.Package id 1", "gpu0.temp", "gpu1.temp"} <= keys
    assert {"gpu0.pcie_gen", "gpu1.pcie_gen", "gpu1.pcie_errors"} <= keys
    assert all(len(s["values"]) == len(ts["t"]) for s in ts["series"])
    thermal = report["sections"]["bench"]["thermal"]
    assert thermal["cpu_temp_peak_c"] == 50.0 and thermal["cpu_crit_c"] == 100.0
    assert (run_dir / "bench-cpu.csv").exists() and (run_dir / "bench-thermal-gpu.csv").exists()


def test_info_on_broken_rig_fails(rig: Callable[..., Path]) -> None:
    out = rig(
        Rig(
            modules=("nouveau", "drm"),
            gpu_driver="nouveau",
            driver_version_file=None,
            smi_ok=False,
            libcuda=False,
            xid=(),
        )
    )
    code = cli.main(["--out", str(out)])  # info is the default command
    _, report, _ = _outputs(out)
    st = _statuses(report)

    assert code == 1 == report["exit_code"]
    for name in ("nouveau", "kernel_module", "bound", "nvidia_smi", "libcuda"):
        assert st[f"gpu.{name}"] == "fail", name
    assert st["gpu.pci"] == "pass" and st["gpu.xid"] == "pass"
    assert st["gpu.pcie_link"] == st["gpu.blackwell"] == "skip"  # need nvidia-smi rows
    assert all(r["hint"] for r in report["results"] if r["status"] in ("fail", "warn"))
    assert "couldn't communicate with the NVIDIA driver" in st_summary(report, "gpu.nvidia_smi")


def test_install_dry_run_plans_without_running(rig: Callable[..., Path]) -> None:
    out = rig()
    code = cli.main(["--out", str(out), "install", "driver"])
    run_dir, report, _ = _outputs(out)

    assert code == 0
    assert _statuses(report) == {"install.driver": "skip"}
    steps = report["sections"]["install_plan"]["steps"]
    assert any("nvidia-driver-580-server-open" in s["argv"] for s in steps)
    log = (run_dir / "hwb.log.jsonl").read_text()
    assert '"msg": "exec"' not in log  # a dry run spawns nothing


@pytest.mark.parametrize(
    "argv",
    [
        ["info"],
        ["bench", "all", "--duration", "1", "--interval", "0.25"],
        ["serve"],
        ["install", "tools"],
        ["install", "cuda"],
    ],
    ids=lambda a: "-".join(a),
)
def test_every_command_writes_complete_report(rig: Callable[..., Path], argv: list[str]) -> None:
    out = rig()
    code = cli.main(["--out", str(out), *argv])
    _, report, _ = _outputs(out)
    assert code == report["exit_code"] == 0
    assert report["command"] == argv[0]
    assert report["results"]
    assert "error" not in _statuses(report).values()


def test_one_report_html_collects_every_run(rig: Callable[..., Path]) -> None:
    out = rig()
    assert cli.main(["--out", str(out), "info"]) == 0
    assert cli.main(["--out", str(out), "install", "tools"]) == 0
    (out / "19990101T000000Z-junk-info").mkdir()
    (out / "19990101T000000Z-junk-info" / "report.json").write_text("{truncated")
    assert cli.main(["--out", str(out), "serve"]) == 0

    assert [p.name for p in out.glob("**/report.html")] == ["report.html"]
    runs = _page_runs(out)
    assert [r["report"]["command"] for r in runs] == ["info", "install", "serve"]  # oldest first
    for r in runs:
        assert r["report"] == json.loads((out / r["id"] / "report.json").read_text())


def test_monitor_records_without_adding_load(rig: Callable[..., Path]) -> None:
    out = rig()
    code = cli.main(["--out", str(out), "monitor", "--duration", "2", "--interval", "0.25"])
    run_dir, report, _ = _outputs(out)
    st = _statuses(report)
    assert st == {"bench.monitor": "warn"}  # GPU1: busy at Gen3 on a Gen4 link
    assert code == 0
    mon = report["sections"]["bench"]["monitor"]
    assert [p["name"] for p in mon["timeseries"]["phases"]] == ["monitor"]
    assert "cpu_clock_sag_pct" not in mon  # clocks follow the workload, not our load
    log = (run_dir / "hwb.log.jsonl").read_text()
    assert "stress-ng" not in log and "pyload" not in log  # no load generator ran
