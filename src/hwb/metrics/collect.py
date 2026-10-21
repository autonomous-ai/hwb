"""Static hardware inventory: CPU, memory/DIMMs, disks, NICs, board, NUMA, sensors, GPUs.

Each collector is independent; a failing one becomes an ERROR result for that
section only. Serial numbers are deliberately not collected, except each GPU's (from nvidia-smi),
which tracks a card across slots and reports.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import nvsmi
from ..core import proc
from ..core.context import RunContext
from ..core.errors import HwbError
from ..core.platform import is_root, os_info
from ..core.result import CheckResult, Status
from ..serve import weights
from . import workloads

Inventory = dict[str, Any]


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def parse_cpuinfo(text: str) -> Inventory:
    models: set[str] = set()
    sockets: set[str] = set()
    flags: set[str] = set()
    cores_per_socket = 0
    for line in text.splitlines():
        key, _, value = (p.strip() for p in line.partition(":"))
        if key == "model name":
            models.add(value)
        elif key == "physical id":
            sockets.add(value)
        elif key == "cpu cores":
            cores_per_socket = int(value)
        elif key == "flags" and not flags:
            flags = set(value.split())
    n_sockets = max(len(sockets), 1)
    return {
        "models": sorted(models),
        "sockets": n_sockets,
        "physical_cores": cores_per_socket * n_sockets or None,
        "logical_cpus": os.cpu_count(),
        "isa": sorted(flags & {"avx2", "avx512f", "avx512_bf16", "amx_tile", "amx_bf16"}),
    }


def cpu() -> Inventory:
    info = parse_cpuinfo(_read(Path("/proc/cpuinfo")) or "")
    cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    info["governor"] = _read(cpufreq / "scaling_governor")
    max_khz = _read(cpufreq / "cpuinfo_max_freq")
    info["max_mhz"] = int(max_khz) // 1000 if max_khz else None
    return info


def parse_meminfo(text: str) -> Inventory:
    vals: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            vals[key] = int(parts[0])
    return {
        "total_gib": round(vals.get("MemTotal", 0) / 1024**2, 1),
        "swap_gib": round(vals.get("SwapTotal", 0) / 1024**2, 1),
        "hugepages_total": vals.get("HugePages_Total"),
    }


def parse_dmidecode_memory(text: str) -> list[Inventory]:
    """Populated DIMMs from ``dmidecode -t 17``."""
    dimms: list[Inventory] = []
    for block in text.split("\n\n"):
        if "Memory Device" not in block:
            continue
        kv = {}
        for line in block.splitlines():
            key, sep, value = line.strip().partition(":")
            if sep:
                kv[key.strip()] = value.strip()
        size = kv.get("Size", "")
        if not size or size.startswith("No Module"):
            continue
        dimms.append(
            {
                "locator": kv.get("Locator"),
                "size": size,
                "type": kv.get("Type"),
                "speed": kv.get("Speed"),
                "configured_speed": kv.get("Configured Memory Speed"),
                "manufacturer": kv.get("Manufacturer"),
                "part_number": kv.get("Part Number"),
            }
        )
    return dimms


def memory() -> Inventory:
    info = parse_meminfo(_read(Path("/proc/meminfo")) or "")
    if is_root() and proc.have("dmidecode"):
        res = proc.run(["dmidecode", "-t", "17"], timeout=30)
        if res.ok:
            info["dimms"] = parse_dmidecode_memory(res.stdout)
    else:
        info["dimms"] = None  # needs root + dmidecode
    return info


def disks() -> Inventory:
    if proc.have("lsblk"):
        res = proc.run(
            ["lsblk", "-J", "-b", "-d", "-o", "NAME,MODEL,SIZE,ROTA,TRAN,TYPE"], timeout=30
        )
        if res.ok:
            devices = json.loads(res.stdout).get("blockdevices", [])
            return {"devices": [d for d in devices if d.get("type") == "disk"]}
    devs = []
    for dev in sorted(Path("/sys/block").glob("*")):
        if dev.name.startswith(("loop", "ram", "zram")):
            continue
        sectors = _read(dev / "size")
        devs.append(
            {
                "name": dev.name,
                "model": _read(dev / "device/model"),
                "size": int(sectors) * 512 if sectors else None,
                "rota": _read(dev / "queue/rotational") == "1",
            }
        )
    return {"devices": devs}


def network() -> Inventory:
    nics = []
    for nic in sorted(Path("/sys/class/net").glob("*")):
        if nic.name == "lo" or not (nic / "device").exists():
            continue  # skip virtual interfaces
        speed = _read(nic / "speed")
        nics.append(
            {
                "name": nic.name,
                "operstate": _read(nic / "operstate"),
                "speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit() else None,
                "driver": (nic / "device/driver").resolve().name,
            }
        )
    return {"nics": nics}


def board() -> Inventory:
    dmi = Path("/sys/class/dmi/id")
    keys = ("sys_vendor", "product_name", "board_vendor", "board_name", "bios_version", "bios_date")
    return {k: _read(dmi / k) for k in keys}


def numa() -> Inventory:
    nodes = {}
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        nodes[node.name] = _read(node / "cpulist")
    return {"nodes": nodes}


def sensors() -> Inventory:
    """Current temperatures from hwmon, in °C."""
    out: dict[str, dict[str, float]] = {}
    for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        chip = f"{_read(hw / 'name') or 'unknown'}:{hw.name}"
        for temp in sorted(hw.glob("temp*_input")):
            raw = _read(temp)
            if raw is None or not raw.lstrip("-").isdigit():
                continue
            label = _read(hw / temp.name.replace("_input", "_label")) or temp.name
            out.setdefault(chip, {})[label] = int(raw) / 1000
    return {"temperatures_c": out}


def gpus() -> Inventory:
    try:
        return {"gpus": nvsmi.query_gpus()}
    except HwbError as exc:
        return {"gpus": None, "error": str(exc)}


def system() -> Inventory:
    osi = os_info()
    uptime = _read(Path("/proc/uptime"))
    return {
        "os": osi.pretty_name,
        "distro_family": osi.family,
        "kernel": osi.kernel,
        "machine": osi.machine,
        "uptime_s": float(uptime.split()[0]) if uptime else None,
    }


COLLECTORS: dict[str, Callable[[], Inventory]] = {
    "system": system,
    "board": board,
    "cpu": cpu,
    "memory": memory,
    "numa": numa,
    "disks": disks,
    "network": network,
    "sensors": sensors,
    "gpus": gpus,
    "workloads": workloads.gather,
    "model_weights": lambda: {"models": [w.to_dict() for w in weights.find_all()]},
}


def run_info(ctx: RunContext) -> None:
    inventory: Inventory = {}
    ctx.sections["inventory"] = inventory

    for name, collector in COLLECTORS.items():

        def _one(name: str = name, collector: Callable[[], Inventory] = collector) -> CheckResult:
            inventory[name] = collector()
            return CheckResult(f"info.{name}", Status.PASS, _summarize(name, inventory[name]))

        ctx.check(f"info.{name}", _one)


def _summarize(name: str, data: Inventory) -> str:
    if name == "cpu":
        return f"{data['sockets']}x {', '.join(data['models']) or '?'} ({data['logical_cpus']} threads)"
    if name == "memory":
        dimms = data.get("dimms")
        return f"{data['total_gib']} GiB" + (f", {len(dimms)} DIMMs" if dimms else "")
    if name == "gpus":
        rows = data.get("gpus")
        return ", ".join(r.get("name") or "?" for r in rows) if rows else "no nvidia-smi"
    if name == "disks":
        return f"{len(data['devices'])} disk(s)"
    if name == "model_weights":
        return weights.summarize_dicts(data.get("models") or [])
    if name == "workloads":
        return workloads.summarize(data)
    if name == "system":
        return f"{data['os']} / {data['kernel']}"
    return "collected"
