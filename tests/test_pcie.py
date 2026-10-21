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
    assert lim.errors == 0 and lim.bios_capped
    _sysfs(tmp_path, cor=913)  # the same, after the link took errors: a fallback, not a setting
    (gpu / "max_link_speed").write_text("32.0 GT/s PCIe\n")
    lim = pcie.link_limit(tmp_path, "0000:41:00.0")
    assert lim.errors == 913 and lim.target_capped and not lim.bios_capped
    _sysfs(tmp_path)
    (gpu / "max_link_speed").write_text("32.0 GT/s PCIe\n")
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


def test_busy_link_below_max_after_errors_is_not_the_bios() -> None:
    (f,) = analysis.pcie_findings(
        _ts([4, 4], [99, 99], max_gen=5, port_max_gen=5, port_target_gen=4, errors_since_boot=913)
    )
    assert "BIOS" not in f.text and f.hint == analysis.PCIE_HINT


KERNEL = """\
[    2.200000] myhost kernel: acpi PNP0A08:00: _OSC: OS now controls [PCIeHotplug SHPCHotplug PME AER PCIeCapability LTR DPC]
[    2.436359] myhost kernel: AMD-Vi: Extended features (0x25bf, 0x1d): PPR X2APIC NX GT [5] IA GA PC
[    4.120000] myhost kernel: pci 0000:70:00.0: 252.048 Gb/s available PCIe bandwidth, limited by 16.0 GT/s PCIe x16 link at 0000:6f:01.0 (capable of 504.112 Gb/s with 32.0 GT/s PCIe x16 link)
[    9.500000] myhost kernel: NVRM: GPU at PCI:0000:70:00: GPU-12345678-1234-1234-1234-123456789abc
[   12.000000] myhost kernel: usb 1-1: new high-speed USB device number 2
[   12.100000] myhost kernel: ata3: SATA link down (SStatus 0 SControl 300)
[ 1180.200000] myhost kernel: pcieport 0000:6f:01.0: AER: Correctable error message received from 0000:6f:01.0
[ 1180.200100] myhost kernel: pcieport 0000:6f:01.0: PCIe Bus Error: severity=Correctable, type=Physical Layer, (Receiver ID)
[ 1180.200200] myhost kernel: pcieport 0000:6f:01.0:   device [8086:352a] error status/mask=00000001/00002000
[ 1180.200300] myhost kernel: pcieport 0000:6f:01.0:    [ 0] RxErr                  (First)
"""


def test_kernel_pcie_events_are_counted_and_scrubbed() -> None:
    k = pcie.parse_kernel_events(KERNEL)
    assert k["aer_native"] is True
    assert k["lines"] == 6 and k["corrected"] == 1 and k["uncorrected"] == 0
    assert k["error_types"] == {"RxErr": 1}
    assert k["devices"]["0000:6f:01.0"] == 4
    assert (k["first_s"], k["last_s"]) == (4.12, 1180.2003)
    text = " ".join(m for _, m in k["sample"])
    assert "myhost" not in text and "12345678-1234" not in text and "GPU-[uuid]" in text
    assert "usb" not in text


SMI_PCI = """
GPU 00000000:70:00.0
    Serial Number                         : 1234567890123
    PCI
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 5
                Host Max                  : 4
            Link Width
                Max                       : 16x
                Current                   : 16x
        Replays Since Reset               : 7
        Replay Number Rollovers           : 0
        Tx Throughput                     : 1200 KB/s
        Rx Throughput                     : 800 KB/s
"""


def test_nvidia_smi_pci_view_keeps_only_link_numbers() -> None:
    assert pcie.parse_smi_pci(SMI_PCI) == {
        "0000:70:00.0": {
            "gen_max": 4, "gen_current": 4, "gen_device_current": 4, "gen_device_max": 5,
            "gen_host_max": 4, "width_max": 16, "width_current": 16, "replays": 7,
            "replay_rollovers": 0, "tx_kb_s": 1200, "rx_kb_s": 800,
        }
    }  # fmt: skip


def test_link_registers_and_aer_by_type(tmp_path: Path) -> None:
    cfg = bytearray(_config(5))
    cfg[0x60 + 0x0C : 0x60 + 0x10] = (5 | 16 << 4).to_bytes(4, "little")  # LnkCap Gen5 x16
    cfg[0x60 + 0x12 : 0x60 + 0x14] = (4 | 16 << 4 | 1 << 13 | 1 << 14).to_bytes(2, "little")
    cfg[0x60 + 0x32] = 1 << 1  # equalization complete
    regs = pcie.link_regs(bytes(cfg))
    assert regs is not None
    assert (regs["cap_gen"], regs["gen"], regs["width"], regs["target_gen"]) == (5, 4, 16, 5)
    assert regs["dl_active"] and regs["bw_mgmt_status"] and regs["eq_complete"]
    assert pcie.link_regs(bytes(cfg[:64])) is None
    _sysfs(tmp_path, cor=913)
    aer = pcie.aer_counts(tmp_path, "0000:41:00.0")
    assert aer is not None and aer["correctable"] == {"RxErr": 913, "total": 913}
    assert aer["nonfatal"] == {"total": 0}
    (snap,) = pcie.debug_snapshot(tmp_path, ["0000:41:00.0"])
    assert snap["port"]["addr"] == "0000:40:01.1" and snap["gpu"]["gen"] == 4
    firmware = "[ 2.2] h kernel: acpi PNP0A08:00: _OSC: platform does not support [AER LTR DPC]\n"
    assert pcie.parse_kernel_events(firmware)["aer_native"] is False


def test_cooling_devices_report_the_thermal_zones_that_drive_them(tmp_path: Path) -> None:
    thermal = tmp_path / "sys/class/thermal"
    for n, kind, state in ((0, "Processor", 0), (1, "PCIe_Port_Link_Speed_0000:6f:01.0", 4),
                           (2, "PCIe_Port_Link_Speed_0000:15:01.0", 1)):  # fmt: skip
        d = thermal / f"cooling_device{n}"
        d.mkdir(parents=True)
        (d / "type").write_text(kind + "\n")
        (d / "cur_state").write_text(f"{state}\n")
        (d / "max_state").write_text("4\n")
    zone = thermal / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("acpitz\n")
    (zone / "cdev0").symlink_to(thermal / "cooling_device2")
    (zone / "cdev0_trip_point").write_text("1\n")
    idle, bound = pcie.cooling_devices(tmp_path)
    assert idle == {
        "type": "PCIe_Port_Link_Speed_0000:6f:01.0", "cur_state": 4, "max_state": 4, "zones": [],
    }  # fmt: skip
    assert bound["zones"] == ["acpitz"]


def test_aer_bit_lines_without_a_severity_header_still_count() -> None:
    log = (
        "[    8.3] h kernel: pcieport 0000:6f:01.0:    [ 0] RxErr                  (First)\n"
        "[    8.3] h kernel: pci 0000:70:00.0:    [13] NonFatalErr\n"
        "[  204.1] h kernel: pcieport 0000:6f:01.0:    [ 0] RxErr                  (First)\n"
        "[  205.0] h kernel: pcieport 0000:6f:01.0:    [ 4] DLP                    (First)\n"
    )
    k = pcie.parse_kernel_events(log)
    assert (k["corrected"], k["uncorrected"]) == (3, 1)
    assert k["error_types"] == {"RxErr": 2, "NonFatalErr": 1, "DLP": 1}
