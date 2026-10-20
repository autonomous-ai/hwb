from pathlib import Path

from hwb.core.result import Status
from hwb.detect.gpu import (
    GpuFacts,
    KernelDriver,
    PciGpu,
    check_blackwell,
    check_bound,
    check_nvidia_smi,
    check_pcie_link,
    parse_driver_version,
    scan_pci,
)

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
