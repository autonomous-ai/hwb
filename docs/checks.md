# Driver check reference (`hwb info`)

`hwb info` (the default command) first reads the machine (sysfs, procfs, `nvidia-smi`, `ldconfig`, `nvcc`, `journalctl`), then judges what it found. Each check below says what it verifies and what to do when it doesn't pass.

| check | verifies | not PASS → do this |
|---|---|---|
| `gpu.pci` | At least one NVIDIA display/3D controller is on the PCI bus (read from `/sys/bus/pci`, so no driver is needed) | **FAIL:** reseat the card and check its PCIe power cables. In the BIOS, enable *Above 4G Decoding* (and *Resizable BAR*). Try another slot |
| `gpu.nouveau` | The open-source `nouveau` driver is **not** loaded | **FAIL:** install the NVIDIA driver (`hwb install driver`), which blacklists nouveau, then reboot |
| `gpu.kernel_module` | The `nvidia` kernel module is loaded. Reports its version and whether it's the open or proprietary flavor | **FAIL:** `sudo hwb install driver --apply`, then reboot. If it's still missing, look at `dmesg \| grep -i nvrm` and `dkms status` |
| `gpu.bound` | Every NVIDIA GPU on PCI is bound to the `nvidia` driver, or passed through to a VM (`vfio-pci`, `pci-stub`) | **FAIL:** a card is bound to neither, so the driver refused it. Common causes: the driver branch is too old for the GPU, or a hardware fault. Check `dmesg` |
| `gpu.secure_boot` | Secure Boot state. It only matters if it blocks the module | **WARN** (Secure Boot on and module not loaded): enroll the DKMS MOK key (`sudo mokutil --import /var/lib/dkms/mok.pub`, then reboot and confirm in the MOK manager) or disable Secure Boot. Ubuntu's signed packages avoid this |
| `gpu.kernel_headers` | `/lib/modules/$(uname -r)/build` exists | **WARN:** DKMS can't rebuild the driver after a kernel upgrade. Install `linux-headers-$(uname -r)` (Debian/Ubuntu) or `linux-headers` / `linux-lts-headers` (Arch) |
| `gpu.nvidia_smi` | `nvidia-smi` runs and sees every GPU the host owns (PCI GPUs minus those passed through to VMs) | **FAIL** (fails to run): the userspace driver is missing or doesn't match the kernel module, usually because the server hasn't been rebooted since the upgrade. **FAIL** (count mismatch): a GPU failed to initialize. Check `dmesg` for *"fallen off the bus"* / Xid 79, then check power, risers and seating |
| `gpu.blackwell` | For compute capability ≥ 10 (RTX 50xx, RTX PRO 6000 Blackwell, B-series): the open kernel module **and** driver ≥ 570 | **FAIL:** install an `-open` driver package from branch 570 or newer. SKIP when there are no Blackwell GPUs |
| `gpu.pcie_link` | Each GPU's PCIe link **width** equals its maximum, and the highest **generation** the link may reach equals what both the card and its slot support | **WARN, narrower:** the card is running at x8, x4 or x1. Usual causes: bad riser, wrong slot, BIOS bifurcation setting, dirty contacts. **WARN, generation limited** (e.g. a Gen5 card in a Gen5 slot capped at Gen4): as root, hwb reads the slot's *Target Link Speed* and the link's correctable error count since boot. **With errors** the link is marginal and fell back by itself, typically after 15–30 minutes of load once the area around the slot and CPU is hot. A cold power-off brings Gen5 back only until it warms up again; check airflow and the CPU cooler, reseat the card and riser, and record when it drops with `hwb monitor`. **On a clean link** the BIOS set the slot lower, often for a riser: set it back to Auto, then check the link holds without errors under load (`hwb serve`, `hwb bench thermal`). **WARN, errors since boot:** correctable errors on a link that still runs at full speed, the early sign of the same problem. A Gen5 card in a slot that is only Gen4 is fine and only noted. The *current* generation drops at idle, which is normal, so it is judged only under load by the benchmarks and `serve.pcie` |
| `gpu.persistence` | Persistence mode is on (avoids driver re-initialization delays and first-call latency) | **WARN:** `sudo systemctl enable --now nvidia-persistenced` (or `sudo nvidia-smi -pm 1`, which doesn't survive a reboot) |
| `gpu.libcuda` | `libcuda.so.1` is in the linker cache. PyTorch, vLLM and every CUDA app need it | **FAIL:** reinstall the driver package. Run `sudo ldconfig` |
| `gpu.cuda_toolkit` | `nvcc` version, if installed. WARNs when the toolkit is newer than the driver supports | **SKIP** is normal: the toolkit is optional. **WARN:** upgrade the driver or install a matching `cuda-toolkit-X-Y` |
| `gpu.pcie_kernel` | No PCIe errors (AER) in this boot's kernel log, and no kernel cooling device slowing a PCIe link | **SKIP:** run as root to read the log. **WARN:** corrected errors, with the error types (`RxErr` = signal trouble at the receiver), the device and when they started (seconds after boot): on a GPU link they make it fall back a generation, usually once the area is hot. Check airflow, reseat card and riser, try another riser. **FAIL:** uncorrected errors. If the firmware handles AER (`_OSC: platform does not support [AER]`), the OS sees no corrected errors at all, and the check says so |
| `gpu.xid` | No `NVRM: Xid` events in this boot's kernel log | **SKIP:** run as root to read the log. **WARN:** look up the codes. 13/31/43 are usually application faults. 48/63/64/94/95 are memory/ECC. 79 means the GPU fell off the bus (power, thermal, riser). 119/120 are GSP firmware. Reboot, then reproduce under `bench thermal` |

## GPUs passed through to VMs

A GPU bound to `vfio-pci` (or the older `pci-stub`) is handed to a QEMU/KVM guest, and that counts as accepted. The host's `nvidia` driver doesn't own it, so `nvidia-smi` on the host doesn't list it. Its driver stack is checked inside the guest, by running hwb there.

- **Mixed host** (some GPUs on `nvidia`, some on `vfio-pci`): `gpu.bound` PASSes with "5 bound to nvidia, 3 passed through to VMs (vfio-pci)", and `gpu.nvidia_smi` expects only the host's GPUs.
- **Pure VM host** (every GPU on `vfio-pci`): the host driver checks SKIP instead of FAIL, since no host driver is expected. These are kernel module, Secure Boot, headers, nvidia-smi, Blackwell, PCIe width, persistence, libcuda and CUDA toolkit.

A GPU bound to *nothing* still FAILs, and so does a host GPU missing from `nvidia-smi`.

## Known-good results

A healthy 2× A6000 server on Ubuntu 24.04:

```
PASS   gpu.pci            2 NVIDIA GPU(s) on PCI bus
PASS   gpu.nouveau        nouveau not loaded
PASS   gpu.kernel_module  nvidia 580.x (open kernel module)
PASS   gpu.bound          all GPUs bound to nvidia
PASS   gpu.secure_boot    Secure Boot enabled; module signed
PASS   gpu.kernel_headers headers for running kernel present
PASS   gpu.nvidia_smi     2 GPU(s): NVIDIA RTX A6000, NVIDIA RTX A6000 (CUDA 13.0)
SKIP   gpu.blackwell      no Blackwell-class GPUs
PASS   gpu.pcie_link      PCIe link width and generation at max on all GPUs
PASS   gpu.persistence    persistence mode on
PASS   gpu.libcuda        libcuda.so.1 at /lib/x86_64-linux-gnu/libcuda.so.1
SKIP   gpu.cuda_toolkit   nvcc not found (optional: ...)
PASS   gpu.pcie_kernel     no PCIe errors in the kernel log this boot
PASS   gpu.xid            no Xid events this boot
```

## GPU-specific notes

- **RTX A6000** (Ampere, sm_86, 48 GB GDDR6 with ECC, blower cooler, 300 W). Works with either module flavor. ECC state shows up in the inventory.
- **RTX 5090** (Blackwell, sm_120, 32 GB GDDR7, up to 575 W). Requires the open modules, driver ≥ 570, CUDA ≥ 12.8, and PyTorch/vLLM wheels built for cu128 or newer. Check the PSU headroom and the 12V-2x6 connectors before burn-in.
- **RTX PRO 6000 Blackwell** (sm_120, 96 GB GDDR7 with ECC, **PCIe Gen5 x16**). Same driver/CUDA requirements as the RTX 5090. Comes in three editions:
  - **Workstation Edition**, 600 W, flow-through cooler. Two stacked cards breathe each other's exhaust, so watch the upper card in `bench thermal` or `monitor`. Seat the 12V-2x6 connectors fully.
  - **Max-Q**, 300 W, blower cooler.
  - **Server Edition**, passive. It needs chassis airflow.

  Its healthy busy link is **Gen5**: a drop to Gen4 under load is already a downgrade.

**"A6000" is ambiguous.** "RTX A6000" (Ampere, 48 GB, Gen4) and "RTX PRO 6000 Blackwell" (96 GB, Gen5) are often both called "A6000". Check the GPU name and compute capability in the hwb report (or `nvidia-smi --query-gpu=name,compute_cap --format=csv`) before reasoning about expected PCIe generation, memory budget or which model builds will run. hwb itself always compares a link with *its own* maximum, so its PCIe findings are right for either card.

## PCIe debug data

`sudo hwb` also records, in the report's **GPU driver** tab:
- **PCIe links:** for each GPU, both ends of its link. The card's and the slot's maximum, what the slot is set to train to (*Target Link Speed*), what it runs at now, nvidia-smi's host maximum and replay count, and the AER error counters since boot by type for the card and its port. Register values need root.
- **Kernel PCIe log:** this boot's PCIe/AER/Xid kernel lines, counted by error type and device, with the time since boot of the first and last one, plus the first 20 and last 80 lines. GPU UUIDs are removed.
- Kernel PCIe link-speed cooling devices, if any.

`hwb monitor`, `hwb serve` and the benchmarks add the kernel lines about each GPU's link that appeared *during* the run to their PCIe table data, so a drop to a lower generation can be matched with the errors that caused it.
