"""Strip anything identifying from reports before they leave the machine.

Reports may be viewed by anyone holding the link, so a submission carries no host names,
addresses, unit UUIDs, user paths or process names, and no serials except the GPU's own
(`gpu_serial`), which is kept on purpose to track a card across slots and reports. The gateway applies the same
generic rules again (gateway/src/sanitize.js): keep the two in sync. This side also knows
the real host name and home directory, which the server never sees.
"""

from __future__ import annotations

import copy
import hashlib
import re
import socket
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Keys dropped wherever they appear: they identify a unit, a network endpoint or a person.
# "gpu_serial" is deliberately absent: the card's serial is needed to track it across reports.
DROP_KEYS = frozenset(
    {
        "uuid",
        "serial",
        "serial_number",
        "mac",
        "mac_address",
        "ip",
        "ip_address",
        "ipv4",
        "ipv6",
        "hostname",
        "fqdn",
        "machine_id",
        "username",
        "user",
        "email",
    }
)

_Repl = str | Callable[["re.Match[str]"], str]

# (rule, pattern, replacement). Order matters: GPU UUIDs before anything that could eat parts.
PATTERNS: tuple[tuple[str, re.Pattern[str], _Repl], ...] = (
    (
        "gpu_uuid",
        re.compile(r"\b(?:GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I),
        "[gpu-uuid]",
    ),
    ("mac", re.compile(r"\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b", re.I), "[mac]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[email]"),
    (
        "ipv4",
        re.compile(
            r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\w|\.\d)"
        ),
        "[ip]",
    ),
    # Needs "::" or eight groups, so PCI addresses like 0000:41:00.0 are left alone.
    (
        "ipv6",
        re.compile(
            r"(?<![\w:])(?:(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}"
            r"|(?:[0-9a-f]{1,4}:){1,7}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4})*)?"
            r"|::(?:[0-9a-f]{1,4}:)*[0-9a-f]{1,4})(?![\w:])",
            re.I,
        ),
        "[ip]",
    ),
    ("home_path", re.compile(r"(?:/home|/Users)/[^/\s\"':]+"), "~"),
    # Xid lines name the process that hit the error, which can reveal a customer's workload.
    (
        "xid_process",
        re.compile(r"(\bpid=)'?[^,\s]+'?(, name=)[^,)]+"),
        lambda m: f"{m.group(1)}[pid]{m.group(2)}[process]",
    ),
)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_RUN_TS = re.compile(r"^(\d{8}T\d{6}Z)-")
_MACHINE_ID = Path("/etc/machine-id")
_MIN_HOST_LEN = 3
_DISTINCTIVE_HOST_LEN = 6
_MAX_LABEL = 64


def host_pattern(host: str) -> re.Pattern[str]:
    """Where a host name counts as an occurrence.

    Distinctive names (6+ chars, or with digits or hyphens, like "gpu-rig-01") match anywhere
    between non-alphanumerics, including inside paths and run directory names. Short plain
    names ("gpu") must stand alone, so they never rewrite check names such as "gpu.pci".
    """
    if len(host) >= _DISTINCTIVE_HOST_LEN or re.search(r"[\d-]", host):
        return re.compile(rf"(?<![a-z0-9]){re.escape(host)}(?![a-z0-9])", re.I)
    return re.compile(rf"(?<![\w.-]){re.escape(host)}(?![\w.-])", re.I)


def rig_alias(hostname: str, machine_id: str = "") -> str:
    """Stable, one-way name for this rig. The machine id salts it, so the alias cannot be
    reversed by hashing a list of likely host names."""
    digest = hashlib.sha256(f"{machine_id}\0{hostname.lower()}".encode()).hexdigest()
    return f"rig-{digest[:8]}"


@dataclass
class Scrubber:
    hosts: dict[str, str]  # real host name -> alias
    home: str | None = None
    counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        # Longest first, so "gpu-rig-01.lab.example" is replaced before "gpu-rig-01".
        self._hosts = [
            (host_pattern(h), alias)
            for h, alias in sorted(self.hosts.items(), key=lambda kv: -len(kv[0]))
            if len(h) >= _MIN_HOST_LEN
        ]

    def text(self, s: str) -> str:
        out = _CONTROL.sub("", s)
        if self.home and self.home not in ("/", "/root"):
            n = out.count(self.home)
            if n:
                self.counts["home_path"] += n
                out = out.replace(self.home, "~")
        for rule, pattern, repl in PATTERNS:
            out, n = pattern.subn(repl, out)
            self.counts[rule] += n
        for pattern, alias in self._hosts:
            out, n = pattern.subn(alias, out)
            self.counts["hostname"] += n
        return out

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if str(k).lower() in DROP_KEYS:
                    self.counts[f"key:{k}"] += 1
                    continue
                out[self.text(str(k))] = self.value(x)
            return out
        return v


def _compact_ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%Y%m%dT%H%M%SZ")
    except ValueError:
        return "00000000T000000Z"


def local_identity() -> tuple[list[str], str, str | None]:
    """(host names to hide, machine id salt, home directory) for this machine."""
    name = socket.gethostname()
    names = [name, name.split(".")[0]]
    try:
        machine_id = _MACHINE_ID.read_text(encoding="utf-8").strip()
    except OSError:
        machine_id = ""
    try:
        home: str | None = str(Path.home())
    except RuntimeError:
        home = None
    return names, machine_id, home


def sanitize_bundle(
    runs: Iterable[dict[str, Any]],
    *,
    label: str | None = None,
    extra_hosts: Iterable[str] = (),
    machine_id: str = "",
    home: str | None = None,
) -> tuple[dict[str, Any], Counter[str]]:
    """Scrub ``[{id, report}, ...]`` into the upload body. Returns (bundle, what was removed)."""
    runs = [copy.deepcopy(r) for r in runs]
    names = {str(r["report"].get("host", "")) for r in runs} | set(extra_hosts)
    names.discard("")
    # Every name of this rig maps to one alias, keyed by the short host name.
    hosts = {h: rig_alias(h.split(".")[0], machine_id) for h in names}
    scrub = Scrubber(hosts, home=home)

    out_runs = []
    seen: set[str] = set()
    for run in runs:
        report = scrub.value(run["report"])
        report["host"] = hosts.get(str(run["report"].get("host", "")), report.get("host"))
        m = _RUN_TS.match(str(run.get("id", "")))
        ts = m.group(1) if m else _compact_ts(str(run["report"].get("started_at", "")))
        run_id = base = f"{ts}-{report['host']}-{report.get('command', 'run')}"
        n = 2
        while run_id in seen:
            run_id, n = f"{base}-{n}", n + 1
        seen.add(run_id)
        out_runs.append({"id": run_id, "report": report})

    bundle: dict[str, Any] = {"runs": out_runs}
    if label and label.strip():
        bundle["label"] = scrub.text(label.strip())[:_MAX_LABEL]
    return bundle, Counter({k: v for k, v in scrub.counts.items() if v})


def describe_redactions(counts: Counter[str]) -> list[str]:
    names = {
        "hostname": "host name occurrences",
        "gpu_uuid": "GPU UUIDs",
        "mac": "MAC addresses",
        "email": "email addresses",
        "ipv4": "IPv4 addresses",
        "ipv6": "IPv6 addresses",
        "home_path": "home directory paths",
        "xid_process": "process names in Xid events",
    }
    lines = [f"{n} {names[k]}" for k, n in sorted(counts.items()) if k in names]
    dropped = sum(n for k, n in counts.items() if k.startswith("key:"))
    if dropped:
        keys = sorted({k[4:] for k in counts if k.startswith("key:")})
        lines.append(f"{dropped} identifying fields ({', '.join(keys)})")
    return lines
