"""PCIe link monitoring: sysfs/nvidia-smi parsing and the judgement."""

from __future__ import annotations

from pathlib import Path

import pytest

from hwb.bench import analysis
from hwb.metrics import pcie
from hwb.metrics.sampler import Timeseries

SMI_Q = """
==============NVSMI LOG==============

GPU 00000000:01:00.0
    PCI
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 1
            Link Width
                Max                       : 16x
                Current                   : 16x
        Replays Since Reset               : 3
        Replay Number Rollovers           : 0

GPU 00000000:E1:00.0
    PCI
        Replays Since Reset               : 10
        Replay Number Rollovers           : 1
"""


def test_parsing() -> None:
    assert pcie.normalize_bus_id("00000000:E1:00.0") == "0000:e1:00.0"
    assert [
        pcie.link_gen(s) for s in ("2.5 GT/s PCIe", "8.0 GT/s PCIe", "16.0 GT/s", "", None)
    ] == [1, 3, 4, None, None]
    assert pcie.parse_replays(SMI_Q) == {"0000:01:00.0": 3, "0000:e1:00.0": 10 + 256}


def _sysfs(root: Path, speed: str = "16.0 GT/s PCIe", width: str = "16", cor: int = 0) -> None:
    """A GPU at 41:00.0 behind root port 40:01.1, as the kernel lays it out."""
    port = root / "sys/devices/pci0000:40/0000:40:01.1"
    gpu = port / "0000:41:00.0"
    gpu.mkdir(parents=True, exist_ok=True)
    for d, n in ((gpu, cor), (port, 0)):
        (d / "aer_dev_correctable").write_text(f"RxErr {n}\nBadTLP 0\nTOTAL_ERR_COR {n}\n")
        (d / "aer_dev_nonfatal").write_text("Undefined 0\nTOTAL_ERR_NONFATAL 0\n")
        (d / "aer_dev_fatal").write_text("Undefined 0\nTOTAL_ERR_FATAL 0\n")
    (gpu / "max_link_speed").write_text("16.0 GT/s PCIe\n")
    (gpu / "max_link_width").write_text("16\n")
    (gpu / "current_link_speed").write_text(speed + "\n")
    (gpu / "current_link_width").write_text(width + "\n")
    devs = root / "sys/bus/pci/devices"
    devs.mkdir(parents=True, exist_ok=True)
    if not (devs / "0000:41:00.0").exists():
        (devs / "0000:41:00.0").symlink_to(gpu)
        (devs / "0000:40:01.1").symlink_to(port)


def test_watch_tracks_link_and_error_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pcie, "replay_snapshot", lambda: {})
    _sysfs(tmp_path, cor=5)  # errors from before the run must not count
    w = pcie.PcieWatch(tmp_path, {"2": "0000:41:00.0"})
    assert w.gpus[0].upstream == "0000:40:01.1" and w.gpus[0].max_gen == 4
    w.start()
    assert w.sample()["2"] == {"gen": 4, "width": 16, "errors": 0}
    _sysfs(tmp_path, speed="8.0 GT/s PCIe", width="8", cor=12)
    assert w.sample()["2"] == {"gen": 3, "width": 8, "errors": 7}
    meta = w.finish()["2"]
    assert meta["correctable_errors"] == 7 and meta["uncorrectable_errors"] == 0
    assert (meta["max_width"], meta["min_width"]) == (16, 8)


def _ts(gens: list[float | None], util: list[float], **meta: object) -> Timeseries:
    ts = Timeseries(1.0, t=[float(i) for i in range(len(gens))])
    ts.add("gpu1.pcie_gen", "gen", "pcie")[:] = gens
    ts.add("gpu1.util", "util", "utilization")[:] = util
    ts.meta = {"pcie": {"1": {"max_gen": 4, "max_width": 16, "min_width": 16, **meta}}}
    return ts


def test_idle_link_power_saving_is_not_a_problem() -> None:
    assert analysis.pcie_findings(_ts([1, 1, 1, 1], [0, 0, 2, 0])) == []


def test_link_below_max_while_busy_warns() -> None:
    (f,) = analysis.pcie_findings(_ts([4, 4, 3, 3, 1], [99, 99, 98, 97, 0]))
    assert f.level == "warn"
    assert f.text == "GPU1 PCIe link at Gen3 (max Gen4) while busy, in 2 of 4 busy samples"
    assert "risers" in (f.hint or "")


def test_errors_and_narrowing() -> None:
    found = analysis.pcie_findings(
        _ts(
            [4, 4], [99, 99], min_width=8, correctable_errors=12, uncorrectable_errors=1, replays=40
        )
    )
    assert [f.level for f in found] == ["warn", "fail", "warn", "warn"]
    assert "narrowed to x8 (max x16)" in found[0].text and analysis.verdict(found) == "fail"
