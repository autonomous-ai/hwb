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


def _config(target: int) -> bytes:
    """256 bytes of a root port's config space: capabilities list -> PM (0x40) -> PCIe (0x60)."""
    cfg = bytearray(256)
    cfg[0x06] = 0x10  # status: has a capabilities list
    cfg[0x34] = 0x40
    cfg[0x40:0x42] = bytes([0x01, 0x60])  # power management, next 0x60
    cfg[0x60:0x62] = bytes([0x10, 0x00])  # PCI Express, last
    cfg[0x60 + 0x30] = 0x20 | target  # Link Control 2: target speed in bits 3:0
    return bytes(cfg)


def test_target_link_speed_from_config_space() -> None:
    assert pcie.target_gen(_config(4)) == 4
    assert pcie.target_gen(_config(5)) == 5
    assert pcie.target_gen(_config(4)[:64]) is None  # unprivileged read: header only
    loop = bytearray(_config(4))
    loop[0x41] = 0x40  # a capability pointing at itself must not hang
    assert pcie.target_gen(bytes(loop)) is None


def test_link_limit_reads_card_slot_and_bios(tmp_path: Path) -> None:
    _sysfs(tmp_path)
    port = tmp_path / "sys/devices/pci0000:40/0000:40:01.1"
    gpu = port / "0000:41:00.0"
    (gpu / "max_link_speed").write_text("32.0 GT/s PCIe\n")  # a Gen5 card
    (port / "max_link_speed").write_text("32.0 GT/s PCIe\n")  # in a Gen5 slot
    (port / "config").write_bytes(_config(4))  # that the BIOS set to Gen4
    lim = pcie.link_limit(tmp_path, "0000:41:00.0")
    assert (lim.gpu_gen, lim.port_gen, lim.target_gen, lim.slot_gen) == (5, 5, 4, 5)
    assert lim.bios_capped
    (port / "max_link_speed").write_text("16.0 GT/s PCIe\n")  # a Gen4 slot: nothing to fix
    (port / "config").write_bytes(_config(4))
    lim = pcie.link_limit(tmp_path, "0000:41:00.0")
    assert lim.slot_gen == 4 and not lim.bios_capped


def test_busy_link_below_max_names_the_bios_setting() -> None:
    (f,) = analysis.pcie_findings(
        _ts([4, 4], [99, 99], max_gen=5, port_max_gen=5, port_target_gen=4)
    )
    assert f.text.endswith("; the BIOS limits the slot to Gen4")
    assert f.hint == analysis.BIOS_GEN_HINT


def test_gen5_card_in_a_gen4_slot_is_fine() -> None:
    assert analysis.pcie_findings(_ts([4, 4], [99, 99], max_gen=5, port_max_gen=4)) == []
