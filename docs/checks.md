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
| `gpu.blackwell` | For compute capability ≥ 10 (RTX 50xx, B-series): the open kernel module **and** driver ≥ 570 | **FAIL:** install an `-open` driver package from branch 570 or newer. SKIP when there are no Blackwell GPUs |
| `gpu.pcie_link` | Each GPU's current PCIe link **width** equals its maximum | **WARN:** the card is running at x8, x4 or x1. Usual causes: bad riser, wrong slot, BIOS bifurcation setting, dirty contacts. Link *generation* drops at idle, which is normal, so it isn't checked |
| `gpu.persistence` | Persistence mode is on (avoids driver re-initialization delays and first-call latency) | **WARN:** `sudo systemctl enable --now nvidia-persistenced` (or `sudo nvidia-smi -pm 1`, which doesn't survive a reboot) |
| `gpu.libcuda` | `libcuda.so.1` is in the linker cache. PyTorch, vLLM and every CUDA app need it | **FAIL:** reinstall the driver package. Run `sudo ldconfig` |
| `gpu.cuda_toolkit` | `nvcc` version, if installed. WARNs when the toolkit is newer than the driver supports | **SKIP** is normal: the toolkit is optional. **WARN:** upgrade the driver or install a matching `cuda-toolkit-X-Y` |
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
PASS   gpu.pcie_link      PCIe link width at max on all GPUs
PASS   gpu.persistence    persistence mode on
PASS   gpu.libcuda        libcuda.so.1 at /lib/x86_64-linux-gnu/libcuda.so.1
SKIP   gpu.cuda_toolkit   nvcc not found (optional: ...)
PASS   gpu.xid            no Xid events this boot
```

## GPU-specific notes

- **RTX A6000** (Ampere, sm_86, 48 GB GDDR6 with ECC, blower cooler, 300 W). Works with either module flavor. ECC state shows up in the inventory.
- **RTX 5090** (Blackwell, sm_120, 32 GB GDDR7, up to 575 W). Requires the open modules, driver ≥ 570, CUDA ≥ 12.8, and PyTorch/vLLM wheels built for cu128 or newer. Check the PSU headroom and the 12V-2x6 connectors before burn-in.
