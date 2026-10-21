from hwb.core.platform import ARCH, DEBIAN, OSInfo, parse_os_release
from hwb.metrics.collect import parse_cpuinfo, parse_dmidecode_memory, parse_meminfo
from hwb.nvsmi import parse_csv, parse_cuda_version


def test_nvsmi_csv_maps_na_to_none() -> None:
    text = "0, NVIDIA RTX A6000, [N/A]\n1, NVIDIA RTX A6000, 300.00\n"
    rows = parse_csv(text, ("index", "name", "power.limit"))
    assert rows[0] == {"index": "0", "name": "NVIDIA RTX A6000", "power.limit": None}
    assert rows[1]["power.limit"] == "300.00"


def test_cuda_version_banner() -> None:
    banner = "| NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 13.0     |"
    assert parse_cuda_version(banner) == "13.0"


def test_os_release_families() -> None:
    ubuntu = parse_os_release('ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\n')
    assert ubuntu == {"ID": "ubuntu", "ID_LIKE": "debian", "VERSION_ID": "24.04"}
    osi = OSInfo("ubuntu", ("debian",), "24.04", "Ubuntu 24.04", "6.8.0-45-generic", "x86_64")
    assert osi.family == DEBIAN
    assert OSInfo("omarchy", ("arch",), "", "Omarchy", "6.17.1-arch1-1", "x86_64").family == ARCH


def test_cpuinfo_dual_socket() -> None:
    block = "model name\t: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz\ncpu cores\t: 32\nflags\t: fpu avx2 avx512f\n"
    text = "\n".join(f"processor\t: {i}\nphysical id\t: {i % 2}\n{block}" for i in range(4))
    info = parse_cpuinfo(text)
    assert info["sockets"] == 2
    assert info["physical_cores"] == 64
    assert info["isa"] == ["avx2", "avx512f"]


def test_meminfo() -> None:
    assert parse_meminfo("MemTotal:       263842716 kB\nSwapTotal: 0 kB\n")["total_gib"] == 251.6


def test_dmidecode_skips_empty_slots() -> None:
    text = (
        "Handle 0x0040, DMI type 17\nMemory Device\n\tSize: 32 GB\n\tLocator: DIMM_A1\n\tSpeed: 3200 MT/s\n"
        "\nHandle 0x0041, DMI type 17\nMemory Device\n\tSize: No Module Installed\n\tLocator: DIMM_A2\n"
    )
    dimms = parse_dmidecode_memory(text)
    assert [d["locator"] for d in dimms] == ["DIMM_A1"]


def test_gpu_serial_gets_its_own_key_and_na_is_none() -> None:
    rows = parse_csv("0, [N/A]\n1, 1652724012345\n", ("index", "serial"))
    assert rows == [
        {"index": "0", "gpu_serial": None},
        {"index": "1", "gpu_serial": "1652724012345"},
    ]
