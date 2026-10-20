"""install.sh, the daily update check and `hwb update`, against a local release server."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import zipapp
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hwb import __version__, update

REPO_ROOT = Path(__file__).parent.parent
NEW = "9.9.9"


def _build_pyz(dest: Path, version: str, work: Path) -> Path:
    """A real hwb.pyz from this source tree, with its version set to ``version``."""
    src = work / f"src-{version}"
    shutil.copytree(
        REPO_ROOT / "src" / "hwb", src / "hwb", ignore=shutil.ignore_patterns("__pycache__")
    )
    init = src / "hwb" / "__init__.py"
    init.write_text(init.read_text().replace(f'"{__version__}"', f'"{version}"'))
    zipapp.create_archive(src, dest, interpreter="/usr/bin/env python3", main="hwb.cli:entry")
    dest.chmod(0o755)
    return dest


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Release directory: hwb.pyz (version 9.9.9) + SHA256SUMS, as `make dist` lays it out."""
    d = tmp_path_factory.mktemp("release")
    pyz = _build_pyz(d / "hwb.pyz", NEW, tmp_path_factory.mktemp("work"))
    (d / "SHA256SUMS").write_text(f"{hashlib.sha256(pyz.read_bytes()).hexdigest()}  hwb.pyz\n")
    return d


class _Releases(BaseHTTPRequestHandler):
    """GitHub's layout: /releases/latest -> 302 /releases/tag/vX; /releases/download/vX/<file>."""

    files: Path
    hits: list[str] = []  # noqa: RUF012 - shared with the test on purpose

    def log_message(self, *args: object) -> None:
        pass

    def _serve(self, body: bool) -> None:
        self.hits.append(self.path)
        if self.path == "/releases/latest":
            self.send_response(302)
            self.send_header("Location", f"https://example.invalid/releases/tag/v{NEW}")
            self.end_headers()
            return
        prefix = f"/releases/download/v{NEW}/"
        f = self.files / self.path.removeprefix(prefix) if self.path.startswith(prefix) else None
        if f is None or not f.is_file():
            self.send_response(404)
            self.end_headers()
            return
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(data)

    def do_GET(self) -> None:
        self._serve(True)

    def do_HEAD(self) -> None:
        self._serve(False)


@pytest.fixture
def server(release: Path, tmp_path: Path) -> Iterator[str]:
    files = tmp_path / "served"
    shutil.copytree(release, files)
    _Releases.files = files
    _Releases.hits = []
    srv = HTTPServer(("127.0.0.1", 0), _Releases)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/releases"
    finally:
        srv.shutdown()
        srv.server_close()


def _run(pyz: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(pyz), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )


# ---------------------------------------------------------------- version check


def test_latest_version_from_the_redirect(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(update.ENV_RELEASES, server)
    assert update.latest_version() == NEW
    assert _Releases.hits == ["/releases/latest"]  # HEAD only, the tag page is never fetched


def test_only_https_or_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(update.ENV_RELEASES, "http://mirror.example.com/releases")
    with pytest.raises(update.UpdateError):
        update.releases_url()


def test_notice_checks_at_most_daily_and_stays_quiet_offline(
    server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(update.ENV_RELEASES, server)
    monkeypatch.setenv(update.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.delenv(update.ENV_NO_CHECK, raising=False)
    line = update.notice(now=1000.0)
    assert line and line.startswith(f"hwb {NEW} is available (this is {__version__}). Update: ")
    assert update.notice(now=1000.0 + 3600) == line and len(_Releases.hits) == 1  # cached
    update.notice(now=1000.0 + update.CHECK_INTERVAL_S)
    assert len(_Releases.hits) == 2  # a day later: checked again

    monkeypatch.setenv(update.ENV_RELEASES, "http://127.0.0.1:9/releases")  # nothing listens
    monkeypatch.setenv(update.ENV_HOME, str(tmp_path / "offline"))
    assert update.notice(now=1.0) is None
    monkeypatch.setenv(update.ENV_NO_CHECK, "1")
    assert update.notice(now=1.0) is None


def test_version_ordering() -> None:
    assert update.is_newer("0.10.0", "0.9.9") and not update.is_newer("0.4.0", "0.4.0")
    assert not update.is_newer(None, "0.4.0")


# ---------------------------------------------------------------- hwb update


@pytest.fixture
def installed(tmp_path: Path) -> Path:
    """The current source as an installed `hwb` (a pyz without extension, like install.sh does)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    return _build_pyz(bin_dir / "hwb", __version__, tmp_path)


def test_update_without_apply_only_reports(installed: Path, server: str, tmp_path: Path) -> None:
    env = {update.ENV_RELEASES: server, update.ENV_HOME: str(tmp_path / "h")}
    before = installed.read_bytes()
    r = _run(installed, "update", env=env)
    assert r.returncode == 0, r.stderr
    assert f"latest:    hwb {NEW}" in r.stdout and "update --apply" in r.stdout
    assert installed.read_bytes() == before


def test_update_apply_replaces_the_running_hwb(
    installed: Path, server: str, tmp_path: Path
) -> None:
    env = {update.ENV_RELEASES: server, update.ENV_HOME: str(tmp_path / "h")}
    r = _run(installed, "update", "--apply", "--yes", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"Updated {installed} to hwb {NEW}." in r.stdout
    assert _run(installed, "--version", env=env).stdout.strip() == f"hwb {NEW}"
    assert os.access(installed, os.X_OK)
    assert sorted(p.name for p in installed.parent.iterdir()) == ["hwb"]  # no temp files left


def test_update_refuses_a_tampered_download(installed: Path, server: str, tmp_path: Path) -> None:
    served = Path(_Releases.files)
    (served / "hwb.pyz").write_bytes((served / "hwb.pyz").read_bytes() + b"tampered")
    before = installed.read_bytes()
    env = {update.ENV_RELEASES: server, update.ENV_HOME: str(tmp_path / "h")}
    r = _run(installed, "update", "--apply", "--yes", env=env)
    assert r.returncode == 2 and "checksum mismatch" in r.stderr
    assert installed.read_bytes() == before


def test_update_needs_confirmation_without_a_terminal(
    installed: Path, server: str, tmp_path: Path
) -> None:
    env = {update.ENV_RELEASES: server, update.ENV_HOME: str(tmp_path / "h")}
    r = _run(installed, "update", "--apply", env=env)
    assert r.returncode == 2 and "without --yes" in r.stderr


# ---------------------------------------------------------------- install.sh


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_install_script_verifies_and_installs(release: Path, tmp_path: Path) -> None:
    target = tmp_path / "bin"
    env = {**os.environ, "HWB_BASE_URL": f"file://{release}", "HWB_INSTALL_DIR": str(target)}
    r = subprocess.run(
        ["sh", str(REPO_ROOT / "install.sh")], env=env, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    assert f"installed hwb {NEW} at {target}/hwb" in r.stderr
    assert (
        subprocess.run(
            [str(target / "hwb"), "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
        == f"hwb {NEW}"
    )

    bad = tmp_path / "bad"
    shutil.copytree(release, bad)
    (bad / "SHA256SUMS").write_text(f"{'0' * 64}  hwb.pyz\n")
    env |= {"HWB_BASE_URL": f"file://{bad}", "HWB_INSTALL_DIR": str(tmp_path / "bin2")}
    r = subprocess.run(
        ["sh", str(REPO_ROOT / "install.sh")], env=env, capture_output=True, text=True, check=False
    )
    assert r.returncode != 0 and "checksum mismatch" in r.stderr
    assert not (tmp_path / "bin2" / "hwb").exists()


def test_release_assets_include_the_installer() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "install.sh" in makefile and "dist/install.sh" in workflow
    assert json.dumps(update.INSTALL_CMD).count("install.sh") == 1
