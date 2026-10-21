from pathlib import Path

from hwb.core.result import Status
from hwb.detect.gpu import (
    GpuFacts,
    KernelDriver,
    PciGpu,
    check_blackwell,
    check_bound,
    check_nvidia_smi,
    check_pci,
    check_pcie_kernel,
    check_pcie_link,
    parse_driver_version,
    scan_pci,
)
from hwb.metrics.pcie import LinkLimit

OPEN_570 = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.86.15  Release Build  (dvs-builder@U16-I3-B03-4-3)  Wed Jan 22 2025\nGCC version:  gcc version 13.3.0"
PROP_550 = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.14  Thu Feb 22 01:44:30 UTC 2024"


def test_parse_driver_version() -> None:
    assert parse_driver_version(OPEN_570) == KernelDriver("570.86.15", True)
    assert parse_driver_version(PROP_550) == KernelDriver("550.54.14", False)
    assert parse_driver_version("garbage") is None


def _dev(root: Path, addr: str, vendor: str, cls: str, device: str = "0x2230") -> None:
    d = root / addr
    d.mkdir(parents=True)
    (d / "vendor").write_text(vendor + "\n")
    (d / "class").write_text(cls + "\n")
    (d / "device").write_text(device + "\n")


def test_scan_pci_filters_vendor_and_class(tmp_path: Path) -> None:
    _dev(tmp_path, "0000:01:00.0", "0x10de", "0x030000")  # GPU
    _dev(tmp_path, "0000:01:00.1", "0x10de", "0x040300")  # its HDMI audio
    _dev(tmp_path, "0000:02:00.0", "0x8086", "0x030000")  # Intel iGPU
    gpus = scan_pci(tmp_path)
    assert [g.address for g in gpus] == ["0000:01:00.0"]
    assert gpus[0].driver is None


def _row(**kw: str | None) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "index": "0",
        "name": "NVIDIA RTX A6000",
        "compute_cap": "8.6",
        "pcie.link.width.current": "16",
        "pcie.link.width.max": "16",
    }
    base.update(kw)
    return base


def test_smi_count_mismatch_fails() -> None:
    pci = [PciGpu(f"0000:0{i}:00.0", "0x2230", "0x030000", "nvidia") for i in (1, 2)]
    f = GpuFacts(pci=pci, smi_rows=[_row()])
    assert check_nvidia_smi(f).status is Status.FAIL


def test_unbound_gpu_fails() -> None:
    f = GpuFacts(pci=[PciGpu("0000:01:00.0", "0x2b85", "0x030000", None)])
    assert check_bound(f).status is Status.FAIL


def test_blackwell_requires_open_module() -> None:
    rows = [_row(name="NVIDIA GeForce RTX 5090", compute_cap="12.0")]
    assert (
        check_blackwell(GpuFacts(smi_rows=rows, kernel_driver=KernelDriver("575.51", False))).status
        is Status.FAIL
    )
    assert (
        check_blackwell(GpuFacts(smi_rows=rows, kernel_driver=KernelDriver("565.1", True))).status
        is Status.FAIL
    )
    assert (
        check_blackwell(
            GpuFacts(smi_rows=rows, kernel_driver=KernelDriver("580.65.06", True))
        ).status
        is Status.PASS
    )
    assert check_blackwell(GpuFacts(smi_rows=[_row()])).status is Status.SKIP


def test_pcie_width_degraded_warns() -> None:
    f = GpuFacts(smi_rows=[_row(), _row(index="1", **{"pcie.link.width.current": "4"})])
    r = check_pcie_link(f)
    assert r.status is Status.WARN
    assert "GPU1 x4/x16" in r.summary


def _limit(addr: str, gpu: int, port: int, target: int | None, errors: int | None = 0) -> LinkLimit:
    return LinkLimit(addr, gpu, port, target, errors)


def test_pcie_generation_capped_in_bios_warns() -> None:
    rows = [
        _row(**{"pci.bus_id": "00000000:16:00.0", "pcie.link.gen.max": "5"}),
        _row(index="1", **{"pci.bus_id": "00000000:70:00.0", "pcie.link.gen.max": "4"}),
    ]
    links = [_limit("0000:16:00.0", 5, 5, 5), _limit("0000:70:00.0", 5, 5, 4)]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=links))
    assert r.status is Status.WARN
    assert r.summary == "PCIe generation limited: GPU1 Gen4 of Gen5: set in the BIOS"
    assert "Auto" in (r.hint or "")
    # not root: the slot's setting is unreadable, nvidia-smi still shows the cap
    links = [_limit("0000:16:00.0", 5, 5, None), _limit("0000:70:00.0", 5, 5, None)]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=links))
    assert r.status is Status.WARN and "GPU1 Gen4 of Gen5" in r.summary
    assert "root" in (r.hint or "")


def test_gen5_card_in_gen4_slot_passes_with_a_note() -> None:
    rows = [_row(**{"pci.bus_id": "00000000:16:00.0", "pcie.link.gen.max": "4"})]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=[_limit("0000:16:00.0", 5, 4, 4)]))
    assert r.status is Status.PASS
    assert "the slot is Gen4" in r.summary


# The layout of a host that passes three of its eight GPUs through to QEMU VMs.
MIXED = [
    PciGpu(addr, "0x2684", "0x030000", drv)
    for addr, drv in (
        ("0000:01:00.0", "nvidia"),
        ("0000:21:00.0", "nvidia"),
        ("0000:41:00.0", "nvidia"),
        ("0000:61:00.0", "nvidia"),
        ("0000:81:00.0", "vfio-pci"),
        ("0000:a1:00.0", "vfio-pci"),
        ("0000:c1:00.0", "vfio-pci"),
        ("0000:e1:00.0", "nvidia"),
    )
]
HOST_ROWS = [_row(index=str(i), name="NVIDIA GeForce RTX 4090") for i in range(5)]


def test_gpus_passed_through_to_vms_are_accepted() -> None:
    f = GpuFacts(pci=MIXED, smi_rows=HOST_ROWS, modules={"nvidia"})
    bound = check_bound(f)
    assert bound.status is Status.PASS
    assert bound.summary == "5 bound to nvidia, 3 passed through to VMs (vfio-pci)"
    # nvidia-smi only sees the host's GPUs: 5 of 8 is correct, not a missing GPU.
    assert check_nvidia_smi(f).status is Status.PASS
    assert "3 passed through to VMs" in check_pci(f).summary


def test_missing_host_gpu_still_fails_next_to_passthrough() -> None:
    f = GpuFacts(pci=MIXED, smi_rows=HOST_ROWS[:4], modules={"nvidia"})
    r = check_nvidia_smi(f)
    assert r.status is Status.FAIL
    assert r.summary == "nvidia-smi sees 4 of 5 host GPUs (3 passed through to VMs (vfio-pci))"


def test_gpu_bound_to_nothing_still_fails() -> None:
    pci = [*MIXED[:7], PciGpu("0000:e1:00.0", "0x2684", "0x030000", None)]
    r = check_bound(GpuFacts(pci=pci))
    assert r.status is Status.FAIL and r.summary.endswith("0000:e1:00.0")


def test_pure_vm_host_skips_host_driver_checks() -> None:
    from hwb.detect.gpu import CHECKS

    pci = [PciGpu(f"0000:{i:02x}:00.0", "0x2684", "0x030000", "vfio-pci") for i in (0x81, 0xA1)]
    f = GpuFacts(pci=pci, smi_error="couldn't communicate with the NVIDIA driver", secure_boot=True)
    results = {c.__name__: c(f) for c in CHECKS}
    for name in (
        "check_kernel_module",
        "check_nvidia_smi",
        "check_libcuda",
        "check_headers",
        "check_secure_boot",
    ):
        assert results[name].status is Status.SKIP, name
        assert "passed through to VMs" in results[name].summary
    assert results["check_kernel_module"].name == "gpu.kernel_module"
    assert results["check_pci"].status is results["check_bound"].status is Status.PASS
    assert all(r.status is not Status.FAIL for r in results.values())


def test_rtx_pro_6000_blackwell_is_held_to_the_blackwell_rules() -> None:
    # The card in a "2x A6000" report that is really an RTX PRO 6000 Blackwell (compute 12.0).
    rows = [_row(name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition", compute_cap="12.0")]
    ok = GpuFacts(smi_rows=rows, kernel_driver=KernelDriver("580.65.06", True))
    assert check_blackwell(ok).status is Status.PASS
    proprietary = GpuFacts(smi_rows=rows, kernel_driver=KernelDriver("580.65.06", False))
    assert check_blackwell(proprietary).status is Status.FAIL


def test_pcie_generation_lost_to_link_errors_is_not_blamed_on_the_bios() -> None:
    """A Gen5 link that took errors and fell back to Gen4: the target reads low too."""
    rows = [_row(**{"pci.bus_id": "00000000:70:00.0", "pcie.link.gen.max": "4"})]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=[_limit("0000:70:00.0", 5, 5, 4, 913)]))
    assert r.status is Status.WARN
    assert r.summary == (
        "PCIe generation limited: GPU0 Gen4 of Gen5: fell back after link errors; "
        "correctable PCIe errors since boot: GPU0 913"
    )
    assert "BIOS" not in (r.hint or "") and "airflow" in (r.hint or "")
    # errors alone, link still at its best
    rows = [_row(**{"pci.bus_id": "00000000:70:00.0", "pcie.link.gen.max": "5"})]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=[_limit("0000:70:00.0", 5, 5, 5, 3)]))
    assert r.status is Status.WARN and r.summary == "correctable PCIe errors since boot: GPU0 3"


def test_kernel_pcie_errors_warn_with_when_and_where() -> None:
    k = {
        "lines": 4, "corrected": 913, "uncorrected": 0, "error_types": {"RxErr": 905, "BadTLP": 8},
        "devices": {"0000:6f:01.0": 913}, "first_s": 1180.2, "last_s": 4300.0, "sample": [],
    }  # fmt: skip
    debug = [{"gpu": {"addr": "0000:70:00.0"}, "port": {"addr": "0000:6f:01.0"}}]
    r = check_pcie_kernel(GpuFacts(kernel_pcie=k, pcie_debug=debug))
    assert r.status is Status.WARN
    assert r.summary == (
        "kernel logged 913 PCIe error messages this boot (RxErr 905, BadTLP 8) on GPU port "
        "0000:6f:01.0, first 1180 s after boot, last 4300 s"
    )
    clean = {**k, "corrected": 0, "lines": 0, "error_types": {}, "devices": {}}
    assert check_pcie_kernel(GpuFacts(kernel_pcie=clean)).status is Status.PASS
    assert check_pcie_kernel(GpuFacts()).status is Status.SKIP
    # an idle link sits in the top cooling state on its own: no thermal zone, no warning
    idle = {"type": "PCIe_Port_Link_Speed_0000:6f:01.0", "cur_state": 4, "max_state": 4}
    r = check_pcie_kernel(GpuFacts(kernel_pcie=clean, pcie_cooling=[{**idle, "zones": []}]))
    assert r.status is Status.PASS
    assert check_pcie_kernel(GpuFacts(kernel_pcie=clean, pcie_cooling=[idle])).status is Status.PASS
    bound = [{**idle, "cur_state": 1, "zones": ["acpitz"]}]
    r = check_pcie_kernel(GpuFacts(kernel_pcie=clean, pcie_cooling=bound, pcie_debug=debug))
    assert r.status is Status.WARN
    assert (
        r.summary
        == "thermal zone acpitz can slow the PCIe link at GPU port 0000:6f:01.0 (state 1 of 4)"
    )


def test_link_that_trained_low_on_a_full_speed_port_blames_the_path() -> None:
    rows = [_row(**{"pci.bus_id": "00000000:70:00.0", "pcie.link.gen.max": "4"})]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=[_limit("0000:70:00.0", 5, 5, 5, 0)]))
    assert r.status is Status.WARN
    assert r.summary == (
        "PCIe generation limited: GPU0 Gen4 of Gen5: the slot is set to Gen5 but the link trained lower"
    )
    assert "riser" in (r.hint or "") and "run hwb as root" not in (r.hint or "")
    r = check_pcie_link(
        GpuFacts(
            smi_rows=rows, pcie_links=[_limit("0000:70:00.0", 5, 5, 5, 0)],
            kernel_pcie={"aer_native": False},
        )
    )  # fmt: skip
    assert "cannot see errors" in r.summary


def test_bios_cap_is_not_claimed_when_the_firmware_hides_errors() -> None:
    rows = [_row(**{"pci.bus_id": "00000000:70:00.0", "pcie.link.gen.max": "4"})]
    links = [_limit("0000:70:00.0", 5, 5, 4, 0)]
    r = check_pcie_link(
        GpuFacts(smi_rows=rows, pcie_links=links, kernel_pcie={"aer_native": False})
    )
    assert "set in the BIOS" not in r.summary and "firmware handles AER" in r.summary


def test_pcie_link_names_the_card_by_serial() -> None:
    rows = [
        _row(
            **{
                "pci.bus_id": "00000000:70:00.0",
                "pcie.link.gen.max": "4",
                "gpu_serial": "1652724012345",
            }
        )
    ]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=[_limit("0000:70:00.0", 5, 5, 5, 0)]))
    assert r.summary.startswith("PCIe generation limited: GPU0 (SN 1652724012345) Gen4 of Gen5")


def test_link_running_slow_under_traffic_fell_back() -> None:
    rows = [_row(**{"pci.bus_id": "00000000:70:00.0", "pcie.link.gen.max": "5"})]
    busy = {"0000:70:00.0": {"gen_current": 4, "rx_kb_s": 746_000, "tx_kb_s": 70_000}}
    links = [_limit("0000:70:00.0", 5, 5, 5, 34)]
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=links, smi_pci=busy))
    assert r.summary.startswith(
        "PCIe generation limited: GPU0 Gen4 of Gen5: fell back after link errors"
    )
    assert "marginal" in (r.hint or "")
    idle = {"0000:70:00.0": {"gen_current": 1, "rx_kb_s": 500, "tx_kb_s": 700}}
    r = check_pcie_link(GpuFacts(smi_rows=rows, pcie_links=links, smi_pci=idle))
    assert r.summary == "correctable PCIe errors since boot: GPU0 34"
