"""Release update check and self-update for the single-file hwb (hwb.pyz installed as `hwb`).

- ``notice()``: at most once a day (cached under $HWB_HOME), a 2 s look at which release
  GitHub's /releases/latest redirects to. Offline rigs just get no notice. Only a plain
  request to GitHub is made; nothing about the machine is sent.
- ``self_update()``: download the release's hwb.pyz + SHA256SUMS, verify, test-run the new
  file, then atomically replace the running archive. Only with --apply, after confirmation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .core import proc
from .core.errors import HwbError, UserAbortError

log = logging.getLogger(__name__)

REPO = "autonomous-ai/hwb"
RELEASES_URL = f"https://github.com/{REPO}/releases"
# Short link to the latest release's install.sh ({RELEASES_URL}/latest/download/install.sh).
INSTALL_CMD = "curl -fsSL https://autonomous.ai/hwb-install.sh | sudo sh"
ENV_RELEASES = "HWB_RELEASES_URL"  # mirror or test server instead of GitHub
ENV_NO_CHECK = "HWB_NO_UPDATE_CHECK"
ENV_HOME = "HWB_HOME"
CHECK_INTERVAL_S = 24 * 3600
CHECK_TIMEOUT_S = 2.0
DOWNLOAD_TIMEOUT_S = 120.0
_TAG = re.compile(r"/tag/v?(\d+(?:\.\d+){1,2})/?$")
_LOCAL = {"localhost", "127.0.0.1", "::1"}


class UpdateError(HwbError):
    pass


def releases_url() -> str:
    url = (os.environ.get(ENV_RELEASES) or RELEASES_URL).rstrip("/")
    parts = urlsplit(url)
    if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in _LOCAL):
        raise UpdateError(f"{ENV_RELEASES} must be https:// (http only for localhost): {url}")
    return url


def hwb_home() -> Path:
    return Path(os.environ.get(ENV_HOME) or Path.home() / ".cache" / "hwb")


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v))


def is_newer(latest: str | None, current: str = __version__) -> bool:
    return bool(latest) and version_tuple(latest or "") > version_tuple(current)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None  # surface the 30x so we can read its Location without fetching the page


def latest_version(timeout: float = CHECK_TIMEOUT_S) -> str:
    """The newest release's version, from where /releases/latest redirects (no API quota)."""
    req = urllib.request.Request(  # noqa: S310 - releases_url() allows only https (http: localhost)
        f"{releases_url()}/latest", method="HEAD", headers={"User-Agent": f"hwb/{__version__}"}
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as resp:
            location = resp.geturl()
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location", "") if exc.code in (301, 302, 303, 307, 308) else ""
        if not location:
            raise UpdateError(f"release check failed: HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"release check failed: {getattr(exc, 'reason', exc)}") from None
    m = _TAG.search(urlsplit(location).path)
    if not m:
        raise UpdateError(f"no release version in {location!r}")
    return m.group(1)


def notice(now: float | None = None) -> str | None:
    """One line if a newer release exists; checks GitHub at most once per CHECK_INTERVAL_S."""
    if os.environ.get(ENV_NO_CHECK):
        return None
    now = time.time() if now is None else now
    cache = hwb_home() / "update-check.json"
    try:
        state = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    latest = state.get("latest")
    checked_at = state.get("checked_at")
    if not isinstance(checked_at, (int, float)) or now - checked_at >= CHECK_INTERVAL_S:
        try:
            latest = latest_version()
        except UpdateError as exc:
            log.debug("update check: %s", exc)
            latest = None  # offline: say nothing, try again tomorrow
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"checked_at": now, "latest": latest}), encoding="utf-8")
        except OSError:
            pass
    if not isinstance(latest, str) or not is_newer(latest):
        return None
    return f"hwb {latest} is available (this is {__version__}). Update: {update_command()}"


def running_archive() -> Path | None:
    """The hwb.pyz file this process runs from, or None (pip install, source checkout)."""
    for parent in Path(__file__).resolve().parents:
        if parent.is_file():
            return parent if zipfile.is_zipfile(parent) else None
    return None


def update_command() -> str:
    archive = running_archive()
    if archive is None:
        return INSTALL_CMD
    found = shutil.which(archive.name)
    name = archive.name if found and Path(found).resolve() == archive else str(archive)
    sudo = "" if os.access(archive.parent, os.W_OK) else "sudo "
    return f"{sudo}{name} update --apply"


def _download(url: str, dest: Path) -> None:
    # url is always built from releases_url(), which allows only https (http: localhost).
    req = urllib.request.Request(url, headers={"User-Agent": f"hwb/{__version__}"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, dest.open("wb") as fh:  # noqa: S310
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"download failed: {url}: {getattr(exc, 'reason', exc)}") from None


def _confirm(question: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise UserAbortError("refusing to replace hwb non-interactively without --yes")
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        raise UserAbortError("cancelled")


def self_update(*, apply: bool, assume_yes: bool) -> int:
    latest = latest_version(timeout=15)
    print(f"installed: hwb {__version__}\nlatest:    hwb {latest}")
    if not is_newer(latest):
        print("hwb is up to date.")
        return 0
    archive = running_archive()
    if archive is None:
        raise UpdateError(
            "this hwb is not an installed hwb.pyz (pip install or source checkout); "
            f"upgrade it the way it was installed, or switch to the release: {INSTALL_CMD}"
        )
    base = f"{releases_url()}/download/v{latest}"
    print(f"Plan: replace {archive} with {base}/hwb.pyz (SHA256 checked, test-run first).")
    if not apply:
        print(f"Run `{update_command()}` to install it.")
        return 0
    if not (os.access(archive.parent, os.W_OK) and os.access(archive, os.W_OK)):
        raise UpdateError(f"no write access to {archive}; run with sudo")
    _confirm(f"Update hwb {__version__} -> {latest}?", assume_yes)

    tmp = archive.parent / f".{archive.name}.new.{os.getpid()}"
    sums = archive.parent / f".{archive.name}.sums.{os.getpid()}"
    try:
        _download(f"{base}/hwb.pyz", tmp)
        _download(f"{base}/SHA256SUMS", sums)
        want = next(
            (
                line.split()[0]
                for line in sums.read_text(encoding="utf-8").splitlines()
                if line.split()[1:] in (["hwb.pyz"], ["*hwb.pyz"])
            ),
            None,
        )
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if want != got:
            raise UpdateError(f"checksum mismatch for hwb.pyz ({got} != {want}); nothing replaced")
        res = proc.run([sys.executable, str(tmp), "--version"], timeout=60)
        if res.stdout.strip() != f"hwb {latest}":
            raise UpdateError(
                f"downloaded hwb does not run as expected: {res.stdout.strip() or res.stderr.strip()}"
            )
        tmp.chmod(archive.stat().st_mode & 0o7777)
        tmp.replace(archive)  # atomic: the old file stays intact until this moment
    finally:
        for p in (tmp, sums):
            p.unlink(missing_ok=True)
    print(f"Updated {archive} to hwb {latest}.")
    return 0
