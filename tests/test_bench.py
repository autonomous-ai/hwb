"""Sampler probes, load generators, process teardown and the thermal judgement."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from hwb.bench import analysis, load
from hwb.bench.options import scaled
from hwb.core import proc
from hwb.metrics import sampler as sm
from hwb.metrics.sampler import Timeseries


def _put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


# ---------------------------------------------------------------- probes


def test_cpu_utilization_from_proc_stat(tmp_path: Path) -> None:
    _put(tmp_path, "proc/stat", "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 2 3 4\n")
    a = sm.cpu_times(tmp_path)
    _put(tmp_path, "proc/stat", "cpu  175 0 125 850 50 0 0 0 0 0\n")
    b = sm.cpu_times(tmp_path)
    assert sm.utilization(a, b) == pytest.approx(50.0)  # busy +100 of +200 (iowait is idle)
    assert sm.utilization(None, b) is None
    assert sm.cpu_times(tmp_path / "nope") is None


def test_clock_from_cpufreq_else_cpuinfo(tmp_path: Path) -> None:
    _put(tmp_path, "proc/cpuinfo", "cpu MHz\t\t: 2000.0\ncpu MHz\t\t: 3000.0\n")
    assert sm.cpu_mhz(tmp_path) == 2500.0
    for i, khz in enumerate((3_400_000, 3_000_000)):
        _put(tmp_path, f"sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", f"{khz}\n")
    assert sm.cpu_mhz(tmp_path) == 3200.0


def test_temperature_sensors_prefer_package_and_tdie(tmp_path: Path) -> None:
    hw = "sys/class/hwmon"
    _put(tmp_path, f"{hw}/hwmon0/name", "coretemp")
    for n, (label, milli) in enumerate(
        (("Package id 0", 61000), ("Core 0", 58000), ("Package id 1", 64000)), 1
    ):
        _put(tmp_path, f"{hw}/hwmon0/temp{n}_label", label)
        _put(tmp_path, f"{hw}/hwmon0/temp{n}_input", str(milli))
        _put(tmp_path, f"{hw}/hwmon0/temp{n}_crit", "100000")
    _put(tmp_path, f"{hw}/hwmon1/name", "k10temp")
    _put(tmp_path, f"{hw}/hwmon1/temp1_label", "Tctl")
    _put(tmp_path, f"{hw}/hwmon1/temp1_input", "70000")
    _put(tmp_path, f"{hw}/hwmon1/temp2_label", "Tdie")
    _put(tmp_path, f"{hw}/hwmon1/temp2_input", "60000")
    _put(tmp_path, f"{hw}/hwmon2/name", "nvme")
    _put(tmp_path, f"{hw}/hwmon2/temp1_input", "40000")

    sensors = sm.find_cpu_temp_sensors(tmp_path)
    assert [s.label for s in sensors] == ["Package id 0", "Package id 1", "Tdie #1"]
    assert [sm.read_temp(s) for s in sensors] == [61.0, 64.0, 60.0]
    assert sensors[0].crit == 100.0  # from temp*_crit
    assert sensors[2].crit == 95.0  # k10temp publishes none: AMD Tctl throttle point


def test_single_amd_chip_is_just_tctl(tmp_path: Path) -> None:
    # What the Threadripper PRO 5955WX dev rig reports: only Tctl, no limit.
    _put(tmp_path, "sys/class/hwmon/hwmon1/name", "k10temp")
    _put(tmp_path, "sys/class/hwmon/hwmon1/temp1_label", "Tctl")
    _put(tmp_path, "sys/class/hwmon/hwmon1/temp1_input", "94125")
    (s,) = sm.find_cpu_temp_sensors(tmp_path)
    assert (s.label, s.crit, sm.read_temp(s)) == ("Tctl", 95.0, 94.125)


def test_rapl_power_handles_wraparound() -> None:
    assert sm.rapl_watts({"p0": 1_000_000}, {"p0": 3_000_000}, {}, 2.0) == 1.0
    assert sm.rapl_watts({"p0": 9_000_000}, {"p0": 1_000_000}, {"p0": 10_000_000}, 1.0) == 2.0
    assert sm.rapl_watts({}, {"p0": 1.0}, {}, 1.0) is None


def test_misc_probes(tmp_path: Path) -> None:
    _put(tmp_path, "proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
    _put(tmp_path, "proc/loadavg", "12.50 8.00 4.00 3/900 1234\n")
    for i in range(2):
        _put(tmp_path, f"sys/devices/system/cpu/cpu{i}/thermal_throttle/core_throttle_count", "3")
    assert sm.mem_used_pct(tmp_path) == 75.0
    assert sm.load1(tmp_path) == 12.5
    assert sm.throttle_events(tmp_path) == 6


def test_gpu_csv_parsing() -> None:
    wall0 = time.mktime(time.strptime("2026/10/20 07:00:00", "%Y/%m/%d %H:%M:%S"))
    text = (
        "0, 2026/10/20 07:00:01.500, 98, 77, 560.12, 2610, 30000, 4, 16, 0x0000000000000000\n"
        "1, 2026/10/20 07:00:01.501, 97, 89, 575.00, 2400, 30000, 3, 16, 0x0000000000000044\n"
        "garbage line\n"
    )
    rows = sm.parse_gpu_csv(text, wall0)
    assert [(r["index"], r["temp"], round(r["t"], 1)) for r in rows] == [
        ("0", 77.0, 1.5),
        ("1", 89.0, 1.5),
    ]
    assert rows[1]["reasons"] & sm.GPU_SLOWDOWN and rows[1]["reasons"] & sm.GPU_POWER_CAP
    assert (rows[0]["pcie_gen"], rows[1]["pcie_gen"]) == (4.0, 3.0)
    # A driver without the optional fields: parse with the reduced field list.
    old = tuple(f for f in sm.GPU_FIELDS if f not in sm.OPTIONAL_GPU_FIELDS)
    (row,) = sm.parse_gpu_csv("0, 2026/10/20 07:00:01.500, 98, 77, 560, 2610, 30000\n", wall0, old)
    assert row["pcie_gen"] is None and row["reasons"] is None


def test_report_form_downsamples_and_keeps_peaks() -> None:
    ts = Timeseries(1.0, t=[float(i) for i in range(10)])
    ts.add("cpu.temp.Package id 0", "CPU temp", "temperature")[:] = [
        50,
        90,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
    ]
    ts.add("cpu.util", "CPU", "utilization")[:] = [0, 100, 0, 100, 0, 100, 0, 100, 0, 100]
    ts.add("cpu.power", "power", "power")[:] = [None] * 10  # all-missing series are dropped
    ts.phases = [{"name": "load", "start": 0.0, "end": 9.0}]
    out = ts.to_report(max_points=5)
    assert out["t"] == [0, 2, 4, 6, 8] and out["downsampled_from"] == 10 and out["interval_s"] == 2
    by_key = {s["key"]: s for s in out["series"]}
    assert by_key["cpu.temp.Package id 0"]["values"][0] == 90  # temperatures keep the max
    assert by_key["cpu.util"]["values"] == [50, 50, 50, 50, 50]
    assert by_key["cpu.util"]["unit"] == "%" and "cpu.power" not in by_key


def test_sampler_records_phases_and_csv(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _put(root, "proc/stat", "cpu  100 0 100 800 0 0 0 0\n")
    _put(root, "proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 500 kB\n")
    _put(root, "sys/class/hwmon/hwmon0/name", "coretemp")
    _put(root, "sys/class/hwmon/hwmon0/temp1_label", "Package id 0")
    _put(root, "sys/class/hwmon/hwmon0/temp1_input", "55000")
    with sm.Sampler(tmp_path, name="t", interval=0.05, gpus=False, root=root) as s:
        s.phase("idle")
        time.sleep(0.2)
        s.phase("load")
        time.sleep(0.2)
    ts = s.result
    assert [p["name"] for p in ts.phases] == ["idle", "load"]
    assert ts.phases[0]["start"] == 0.0  # the sample taken on entry belongs to "idle"
    assert len(ts.t) >= 4 and ts.values("mem.used")[0] == 50.0
    assert set(ts.values("cpu.temp.Package id 0")) == {55.0}
    header = (tmp_path / "t.csv").read_text().splitlines()[0]
    assert header.startswith("t_s,phase,cpu_util_pct") and "temp_c[Package id 0]" in header


# ---------------------------------------------------------------- load + processes


STRESS_NEW = """stress-ng: info:  [4242] dispatching hogs: 64 cpu
stress-ng: metrc: [4242] stressor       bogo ops real time  usr time  sys time   bogo ops/s     bogo ops/s
stress-ng: metrc: [4242]                           (secs)    (secs)    (secs)   (real time) (usr+sys time)
stress-ng: metrc: [4242] cpu             1234567     60.01   3836.80      0.30     20572.69         321.73
"""
STRESS_OLD = "stress-ng: info:  [99] cpu   5000   10.00   9.90   0.01   500.00   504.54\n"


def test_stress_ng_parsing() -> None:
    assert load.parse_stress_ng(STRESS_NEW) == (1234567, 20572.69)
    assert load.parse_stress_ng(STRESS_OLD) == (5000, 500.0)
    assert load.parse_stress_ng("stress-ng: error: oops") is None


def test_python_fallback_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load, "method", lambda: "python")
    r = load.run_cpu(tmp_path, "t", 2, 0.3)
    assert r.method == "python" and r.error is None and r.ops_per_s and r.ops_per_s > 0
    assert not list(tmp_path.glob("pyload-*"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_spawn_tears_down_the_whole_process_group(tmp_path: Path) -> None:
    # The child forks a grandchild that would outlive a plain kill of the child.
    script = (
        "import os, subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    out = tmp_path / "out"
    with proc.spawn([sys.executable, "-c", script], stdout=out) as child:
        for _ in range(100):
            if out.read_text().strip():
                break
            time.sleep(0.05)
        grandchild = int(out.read_text())
        assert child.returncode is None and _alive(grandchild)
    assert child.returncode is not None
    for _ in range(50):
        if not _alive(grandchild):
            break
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_short_runs_shrink_side_phases() -> None:
    assert scaled(10, 60) == 10 and scaled(10, 600) == 10
    assert scaled(10, 6) == 1.0 and scaled(30, 30) == 15


# ---------------------------------------------------------------- judgement


def _thermal_ts(
    temps: list[float], mhz: list[float], crit: float = 100.0, **meta: object
) -> Timeseries:
    n = len(temps)
    ts = Timeseries(1.0, t=[float(i) for i in range(n)])
    ts.add("cpu.temp.Package id 0", "pkg", "temperature")[:] = temps
    ts.add("cpu.mhz", "clock", "clock")[:] = mhz
    ts.phases = [{"name": "load", "start": 0.0, "end": float(n - 1)}]
    ts.meta = {"cpu_crit_c": crit, **meta}
    return ts


def test_cool_steady_rig_passes() -> None:
    f, facts = analysis.thermal_findings(_thermal_ts([60.0] * 60, [3000.0] * 60), "load")
    assert analysis.verdict(f) == "pass" and facts["cpu_temp_margin_c"] == 40


def test_hitting_the_limit_fails() -> None:
    f, _ = analysis.thermal_findings(_thermal_ts([70.0] * 50 + [99.0] * 10, [3000.0] * 60), "load")
    assert analysis.verdict(f) == "fail" and "at its 100 °C limit" in f[0].text


def test_small_margin_throttling_sag_and_gpu_slowdown_warn() -> None:
    ts = _thermal_ts(
        [88.0] * 60,
        [3400.0] * 12 + [3000.0] * 36 + [2900.0] * 12,
        cpu_throttle_events=7,
        gpu_slowdown_samples={"0": 0, "1": 4},
    )
    f, facts = analysis.thermal_findings(ts, "load")
    texts = " | ".join(x.text for x in f)
    assert analysis.verdict(f) == "warn"
    assert "12 °C below" not in texts and "7 CPU thermal-throttle events" in texts
    assert "clocks sagged 15%" in texts and "GPU1 slowed down" in texts and "GPU0" not in texts
    assert facts["cpu_clock_sag_pct"] == pytest.approx(14.7, abs=0.1)


def test_still_rising_is_a_note_not_a_failure() -> None:
    temps = [50.0 + i * 0.1 for i in range(120)]  # +6 °C/min
    f, facts = analysis.thermal_findings(_thermal_ts(temps, [3000.0] * 120), "load")
    assert analysis.verdict(f) == "pass"
    assert [x.level for x in f] == ["note"] and facts[
        "cpu_temp_end_slope_c_per_min"
    ] == pytest.approx(6.0)


def test_no_sensors_means_no_temperature(tmp_path: Path) -> None:
    assert sm.find_cpu_temp_sensors(tmp_path) == []
    assert analysis.cpu_temp(Timeseries(1.0, t=[0.0, 1.0])) == [None, None]
