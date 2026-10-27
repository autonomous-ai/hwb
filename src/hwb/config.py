"""Operator settings that outlive a run: `hwb config`.

A small JSON file, ``$XDG_CONFIG_HOME/hwb/config.json`` (``~/.config/hwb/config.json``), or
``$HWB_CONFIG``. Under ``sudo`` the invoking user's file is read when root has none, so
``hwb serve`` and ``sudo hwb info`` follow the same settings. Environment variables win over
the file.
"""

from __future__ import annotations

import json
import os
import pwd
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .core.errors import HwbError

ENV_CONFIG = "HWB_CONFIG"
ENV_SUBMIT = "HWB_SUBMIT"
FILE = Path("hwb") / "config.json"

# After a run: "ask" offers to submit (interactive runs only), "auto" submits without asking
# (also in tmux/nohup/cron), "off" says nothing.
SUBMIT_MODES = ("ask", "auto", "off")
KEYS = {
    "submit": f"after a run: {' | '.join(SUBMIT_MODES)} (default ask)",
    "label": "label for submitted reports, e.g. a rack position",
    "gateway": "gateway URL for submissions (default: the public hwb gateway)",
}


class ConfigError(HwbError):
    pass


@dataclass(frozen=True)
class Settings:
    submit: str = "ask"
    label: str | None = None
    gateway: str | None = None


def _config_dir(home: Path, env: Mapping[str, str]) -> Path:
    return Path(env.get("XDG_CONFIG_HOME") or home / ".config")


def path(env: Mapping[str, str] | None = None) -> Path:
    """Where `hwb config set` writes."""
    env = os.environ if env is None else env
    if env.get(ENV_CONFIG):
        return Path(env[ENV_CONFIG])
    return _config_dir(Path.home(), env) / FILE


def _sudo_user_path(env: Mapping[str, str]) -> Path | None:
    user = env.get("SUDO_USER")
    if not user or user == "root" or os.geteuid() != 0:
        return None
    try:
        return Path(pwd.getpwnam(user).pw_dir) / ".config" / FILE
    except KeyError:
        return None


def read_file(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if env is None else env
    candidates = [path(env)]
    if not env.get(ENV_CONFIG) and (sudo := _sudo_user_path(env)) is not None:
        candidates.append(sudo)
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read {p}: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{p} is not a JSON object")
        return {k: str(v) for k, v in data.items() if k in KEYS and v is not None}
    return {}


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    values = read_file(env)
    submit = (env.get(ENV_SUBMIT) or values.get("submit") or "ask").strip().lower()
    if submit not in SUBMIT_MODES:
        raise ConfigError(f"submit must be one of {', '.join(SUBMIT_MODES)}, not {submit!r}")
    return Settings(submit=submit, label=values.get("label"), gateway=values.get("gateway"))


def validate(key: str, value: str) -> str:
    if key not in KEYS:
        raise ConfigError(f"unknown setting {key!r}; known: {', '.join(KEYS)}")
    value = value.strip()
    if key == "submit":
        value = value.lower()
        if value not in SUBMIT_MODES:
            raise ConfigError(f"submit must be one of {', '.join(SUBMIT_MODES)}, not {value!r}")
    if not value:
        raise ConfigError(f"empty value for {key}; use `hwb config unset {key}`")
    return value


def write(key: str, value: str | None, env: Mapping[str, str] | None = None) -> Path:
    """Set (or with None remove) one key in the user's own config file."""
    p = path(env)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read {p}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{p} is not a JSON object")
    if value is None:
        data.pop(key, None)
    else:
        data[key] = validate(key, value)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:
        raise ConfigError(f"cannot write {p}: {exc}") from None
    return p
