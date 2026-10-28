"""OS / distro identification."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEBIAN = "debian"
ARCH = "arch"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class OSInfo:
    id: str
    id_like: tuple[str, ...]
    version_id: str
    pretty_name: str
    kernel: str
    machine: str

    @property
    def family(self) -> str:
        ids = (self.id, *self.id_like)
        if any(i in ("debian", "ubuntu") for i in ids):
            return DEBIAN
        # Omarchy is Arch-based; depending on version it reports ID=arch or its own ID.
        if any(i in ("arch", "omarchy") for i in ids):
            return ARCH
        return UNKNOWN


def parse_os_release(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key] = value.strip().strip('"').strip("'")
    return out


def os_info(os_release: Path = Path("/etc/os-release")) -> OSInfo:
    try:
        rel = parse_os_release(os_release.read_text(encoding="utf-8"))
    except OSError:
        rel = {}
    return OSInfo(
        id=rel.get("ID", sys.platform).lower(),
        id_like=tuple(rel.get("ID_LIKE", "").lower().split()),
        version_id=rel.get("VERSION_ID", ""),
        pretty_name=rel.get("PRETTY_NAME", platform.platform()),
        kernel=platform.release(),
        machine=platform.machine(),
    )


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def hand_back(base: Path, env: Mapping[str, str] | None = None) -> None:
    """Under `sudo hwb`, give the root-owned files below ``base`` back to the invoking user.

    Run dirs are 0750, so without this a later `hwb submit` without sudo can't read them.
    Only when ``base`` sits in a directory that user owns (their home or project dir), never
    for output the operator put somewhere root-only.
    """
    env = os.environ if env is None else env
    if not is_root():
        return
    try:
        uid, gid = int(env["SUDO_UID"]), int(env["SUDO_GID"])
    except (KeyError, ValueError):
        return
    try:
        if uid == 0 or base.resolve().parent.stat().st_uid != uid:
            return
        paths = [base, *base.rglob("*")]
    except OSError:
        return
    for p in paths:
        try:
            if not p.is_symlink() and p.stat().st_uid == 0:
                os.chown(p, uid, gid)
        except OSError:
            continue  # vanished or not ours to change; the rest still gets handed back
