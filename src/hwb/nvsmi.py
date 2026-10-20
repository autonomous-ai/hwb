"""Thin nvidia-smi wrapper. Uses the CLI rather than pynvml to keep the core stdlib-only."""

from __future__ import annotations

import csv
import io
import re

from .core import proc
from .core.errors import HwbError

GPU_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "memory.total",
    "compute_cap",
    "persistence_mode",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "temperature.gpu",
    "power.limit",
)

_NA = {"", "N/A", "[N/A]", "[Not Supported]", "Not Supported"}
_CUDA_RE = re.compile(r"CUDA Version:\s*([\d.]+)")

Row = dict[str, "str | None"]


class NvidiaSmiError(HwbError):
    pass


def parse_csv(text: str, fieldnames: tuple[str, ...]) -> list[Row]:
    rows: list[Row] = []
    for raw in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not raw or all(not c.strip() for c in raw):
            continue
        if len(raw) != len(fieldnames):
            raise NvidiaSmiError(f"unexpected nvidia-smi row ({len(raw)} cols): {raw!r}")
        rows.append(
            {
                k: (None if v.strip() in _NA else v.strip())
                for k, v in zip(fieldnames, raw, strict=True)
            }
        )
    return rows


def query_gpus(fieldnames: tuple[str, ...] = GPU_FIELDS, timeout: float = 30.0) -> list[Row]:
    res = proc.run(
        ["nvidia-smi", f"--query-gpu={','.join(fieldnames)}", "--format=csv,noheader,nounits"],
        timeout=timeout,
    )
    if not res.ok:
        raise NvidiaSmiError((res.stderr or res.stdout).strip() or f"exit code {res.returncode}")
    return parse_csv(res.stdout, fieldnames)


def parse_cuda_version(text: str) -> str | None:
    m = _CUDA_RE.search(text)
    return m.group(1) if m else None


def cuda_driver_version(timeout: float = 30.0) -> str | None:
    """Highest CUDA version the installed driver supports (from the nvidia-smi banner)."""
    res = proc.run(["nvidia-smi"], timeout=timeout)
    return parse_cuda_version(res.stdout) if res.ok else None
