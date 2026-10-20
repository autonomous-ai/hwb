import json
from pathlib import Path

import pytest

from hwb.cli import main
from hwb.core.errors import UnsupportedPlatformError
from hwb.core.platform import OSInfo
from hwb.install.plan import cuda_plan, driver_plan, tools_plan

UBUNTU = OSInfo("ubuntu", ("debian",), "24.04", "Ubuntu 24.04", "6.8.0-45-generic", "x86_64")
DEBIAN12 = OSInfo("debian", (), "12", "Debian 12", "6.1.0-25-amd64", "x86_64")
OMARCHY = OSInfo("arch", (), "", "Omarchy", "6.17.1-arch1-1", "x86_64")


def test_ubuntu_driver_plan_uses_open_server_package(tmp_path: Path) -> None:
    argvs = [s.argv for s in driver_plan(UBUNTU, tmp_path, "580")]
    assert any("nvidia-driver-580-server-open" in a for a in argvs)
    assert any("linux-headers-6.8.0-45-generic" in a for a in argvs)


def test_debian_plans_use_nvidia_repo(tmp_path: Path) -> None:
    steps = cuda_plan(DEBIAN12, tmp_path, "12.8")
    urls = [a for s in steps for a in s.argv if a.startswith("https://")]
    assert urls == [
        "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
    ]
    assert "cuda-toolkit-12-8" in steps[-1].argv


def test_arch_driver_plan(tmp_path: Path) -> None:
    (step,) = driver_plan(OMARCHY, tmp_path)
    assert step.argv[:2] == ("pacman", "-Syu")
    assert "nvidia-open-dkms" in step.argv and "linux-headers" in step.argv


def test_unknown_distro_rejected() -> None:
    with pytest.raises(UnsupportedPlatformError):
        tools_plan(OSInfo("fedora", (), "40", "Fedora", "6.9", "x86_64"))


def test_cli_detect_writes_report_even_off_linux(tmp_path: Path) -> None:
    code = main(["--out", str(tmp_path), "detect"])
    (run_dir,) = (p for p in tmp_path.iterdir() if p.is_dir())
    report = json.loads((run_dir / "report.json").read_text())
    assert report["exit_code"] == code
    assert any(r["name"] == "gpu.pci" for r in report["results"])
    assert (run_dir / "hwb.log.jsonl").stat().st_size > 0
