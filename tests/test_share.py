"""hwb submit: sanitizing, the confirmation gate and the upload, against a fake gateway."""

from __future__ import annotations

import json
import re
import stat
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from hwb import cli
from hwb.share import client
from hwb.share.sanitize import rig_alias, sanitize_bundle

FIXTURES = Path(__file__).parent.parent / "ui" / "src" / "fixtures"
REPORT_ID = "3f0c1d1e-8a2b-4c3d-9e4f-0123456789ab"
QR = ["1110", "1001", "0101", "1011"]


def _report(name: str = "healthy", host: str = "gpu-rig-01") -> dict[str, Any]:
    r: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    r["host"] = host
    return r


def _leaky() -> dict[str, Any]:
    r = _report()
    r["results"][0]["summary"] = (
        "owner ops@corp.example.com at 10.20.30.40 / fe80::1ff:fe23:4567:890a"
    )
    r["results"][1]["hint"] = (
        "see /home/alice/hwb-runs/20261020T061003Z-gpu-rig-01-all/hwb.log.jsonl"
    )
    r["results"][2]["data"] = {
        "nic": "00:1a:2b:3c:4d:5e",
        "serial": "PHX123",
        "gpu": "GPU-12345678-1234-1234-1234-123456789abc",
    }
    r["sections"]["gpu_facts"]["smi_rows"][0]["uuid"] = "GPU-00000000-aaaa"
    return r


# ---------------------------------------------------------------- sanitize


def test_sanitize_removes_identifiers_and_keeps_hardware_facts() -> None:
    runs = [{"id": "20261020T061003Z-gpu-rig-01-all", "report": _leaky()}]
    bundle, counts = sanitize_bundle(
        runs, machine_id="m1", home="/home/alice", label="rack 7 @ 10.0.0.9"
    )
    text = json.dumps(bundle)

    for secret in (
        "gpu-rig-01",
        "ops@corp.example.com",
        "10.20.30.40",
        "fe80::1ff:fe23:4567:890a",
        "00:1a:2b:3c:4d:5e",
        "PHX123",
        "GPU-12345678",
        "GPU-00000000-aaaa",
        "/home/alice",
        "python3",
        "10.0.0.9",
    ):
        assert secret not in text, secret
    for kept in (
        "0000:41:00.0",
        "00000000:41:00.0",
        "580.65.06",
        "NVIDIA GeForce RTX 5090",
        "gpu.pci",
    ):
        assert kept in text, kept

    alias = rig_alias("gpu-rig-01", "m1")
    assert bundle["runs"][0]["report"]["host"] == alias
    assert bundle["runs"][0]["id"] == f"20261020T061003Z-{alias}-all"
    assert bundle["label"] == "rack 7 @ [ip]"
    assert "~/hwb-runs/" in text
    assert (
        counts["gpu_uuid"] == 1 and counts["key:uuid"] == 6
    )  # 2 GPUs x 3 places and counts["key:serial"] == 1
    assert counts["xid_process"] >= 1 and counts["hostname"] >= 1


def test_alias_is_stable_per_rig_and_salted() -> None:
    assert rig_alias("gpu-rig-01", "m1") == rig_alias("GPU-RIG-01", "m1")
    assert rig_alias("gpu-rig-01", "m1") != rig_alias("gpu-rig-01", "m2")


def test_short_host_name_never_rewrites_check_names() -> None:
    bundle, _ = sanitize_bundle([{"id": "x", "report": _report(host="gpu")}])
    names = [c["name"] for c in bundle["runs"][0]["report"]["results"]]
    assert "gpu.pci" in names and "gpu.nvidia_smi" in names


# ---------------------------------------------------------------- client pieces


def test_gateway_url_must_be_https_except_localhost() -> None:
    assert client.gateway_base("https://hwb.example.com/") == "https://hwb.example.com"
    assert client.gateway_base("http://127.0.0.1:8787") == "http://127.0.0.1:8787"
    for bad in ("http://hwb.example.com", "ftp://x", "", None):
        with pytest.raises(client.GatewayError):
            client.gateway_base(bad)


def test_qr_rendering_has_quiet_zone_and_two_rows_per_line() -> None:
    plain = client.render_qr(QR, color=False).splitlines()
    assert len(plain) == 6  # 4 rows + 2x4 quiet zone = 12 rows, 2 per line
    assert all(len(line) == 4 + 8 for line in plain)
    assert plain[0] == "█" * 12  # quiet zone is light
    colored = client.render_qr(QR, color=True)
    assert colored.splitlines()[0] == "\033[48;5;231m " * 12 + "\033[0m"  # equal halves: no glyph
    assert "▀" in colored and colored.endswith("\033[0m")


def test_full_qr_uses_colored_spaces_only_and_keeps_every_module() -> None:
    """The style that scans everywhere: no Unicode, one line per module row."""
    full = client.render_qr(QR, color=True, style="full")
    assert full.isascii() and len(full.splitlines()) == 4 + 8
    decoded = [
        "".join("1" if c == "40" else "0" for c in re.findall(r"\033\[(4[07])m  ", line))
        for line in full.splitlines()
    ]
    assert decoded[4:8] == ["0000" + r + "0000" for r in QR]
    assert set(decoded[0]) == {"0"}  # quiet zone
    assert client.render_qr(QR, color=False, style="ascii").splitlines()[0] == "##" * 12


def test_qr_style_follows_the_terminal() -> None:
    utf8 = {"LANG": "en_US.UTF-8", "TERM": "xterm-256color"}
    tall, short = 80, 24
    assert client.qr_style(utf8, "utf-8", color=True, size=33, lines=tall) == "full"
    assert client.qr_style(utf8, "utf-8", color=True, size=33, lines=short) == "half"
    # ssh into a box with a C locale (Python coerced it), the Linux console: no Unicode
    coerced = {"LC_CTYPE": "C.UTF-8", "TERM": "xterm"}
    assert not client.unicode_terminal(coerced, "utf-8")
    assert client.qr_style(coerced, "utf-8", color=True, size=33, lines=short) == "full"
    assert (
        client.qr_style({**utf8, "TERM": "linux"}, "utf-8", color=True, size=33, lines=short)
        == "full"
    )
    assert client.qr_style({}, "ascii", color=False, size=33, lines=tall) == "ascii"
    assert (
        client.qr_style({**utf8, "HWB_QR": "off"}, "utf-8", color=True, size=33, lines=tall)
        == "off"
    )


# ---------------------------------------------------------------- end to end


class _Gateway(BaseHTTPRequestHandler):
    received: ClassVar[list[tuple[str, str, dict[str, str], bytes]]] = []
    status = 201

    def log_message(self, *args: Any) -> None:
        pass

    def _reply(self, code: int, body: Any = None) -> None:
        raw = json.dumps(body).encode() if body is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.received.append(("POST", self.path, dict(self.headers), body))
        if self.status != 201:
            self.send_response(self.status)
            raw = json.dumps({"error": "refused"}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if self.status == 429:
                self.send_header("Retry-After", "405")
            self.end_headers()
            self.wfile.write(raw)
            return
        port = self.server.server_address[1]
        self._reply(
            201,
            {
                "id": REPORT_ID,
                "url": f"http://127.0.0.1:{port}/r/{REPORT_ID}",
                "qr": QR,
                "delete_token": "d" * 64,
                "created_at": "2026-10-20T07:00:00Z",
                "retention_days": 90,
            },
        )

    def do_DELETE(self) -> None:
        self.received.append(("DELETE", self.path, dict(self.headers), b""))
        self._reply(204)


@pytest.fixture
def gateway() -> Iterator[str]:
    _Gateway.received = []
    _Gateway.status = 201
    server = HTTPServer(("127.0.0.1", 0), _Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def out(tmp_path: Path) -> Path:
    for name, run_id in (
        ("healthy", "20261020T061003Z-gpu-rig-01-all"),
        ("broken", "20261020T061004Z-gpu-rig-01-detect"),
    ):
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "report.json").write_text(
            json.dumps(_leaky() if name == "healthy" else _report(name))
        )
    return tmp_path


def test_submit_uploads_sanitized_runs_and_prints_link(
    out: Path, gateway: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWB_GATEWAY_TOKEN", "upload-secret")
    code = cli.main(
        ["--out", str(out), "submit", "--gateway", gateway, "--yes", "--label", "rack 7"]
    )
    assert code == 0

    ((method, path, headers, body),) = _Gateway.received
    assert (method, path) == ("POST", "/api/reports")
    assert headers["Authorization"] == "Bearer upload-secret"
    sent = json.loads(body)
    assert [r["report"]["command"] for r in sent["runs"]] == ["all", "detect"]
    assert sent["label"] == "rack 7"
    assert "gpu-rig-01" not in body.decode() and "10.20.30.40" not in body.decode()

    printed = capsys.readouterr().out
    assert f"/r/{REPORT_ID}" in printed
    assert "█" in printed  # the QR code
    assert "Removed before upload:" in printed and "GPU UUIDs" in printed

    saved = out / client.SUBMISSIONS
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert json.loads(saved.read_text())["delete_token"] == "d" * 64
    # The submission bookkeeping is not mistaken for a run.
    assert not list(out.glob("*-submit"))


def test_submit_refuses_without_confirmation_when_not_interactive(out: Path, gateway: str) -> None:
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway]) == cli.EXIT_ERROR
    assert _Gateway.received == []


def test_dry_run_writes_the_exact_upload_and_sends_nothing(out: Path, gateway: str) -> None:
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--dry-run"]) == 0
    assert _Gateway.received == []
    preview = json.loads((out / "submission-preview.json").read_text())
    assert len(preview["runs"]) == 2 and "gpu-rig-01" not in json.dumps(preview)


def test_gateway_errors_are_reported(
    out: Path, gateway: str, caplog: pytest.LogCaptureFixture
) -> None:
    _Gateway.status = 401
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes"]) == cli.EXIT_ERROR
    assert not (out / client.SUBMISSIONS).exists()


def test_delete_uses_the_saved_token(out: Path, gateway: str) -> None:
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes"]) == 0
    assert cli.main(["--out", str(out), "submit", "--delete", REPORT_ID, "--yes"]) == 0
    method, path, headers, _ = _Gateway.received[-1]
    assert (method, path) == ("DELETE", f"/api/reports/{REPORT_ID}")
    assert headers["Authorization"] == "Bearer " + "d" * 64


def test_run_selection(out: Path, gateway: str) -> None:
    assert (
        cli.main(
            [
                "--out",
                str(out),
                "submit",
                "--gateway",
                gateway,
                "--yes",
                "--run",
                "20261020T061004Z-gpu-rig-01-detect",
            ]
        )
        == 0
    )
    sent = json.loads(_Gateway.received[0][3])
    assert [r["report"]["command"] for r in sent["runs"]] == ["detect"]
    assert (
        cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes", "--run", "nope"])
        == cli.EXIT_ERROR
    )


def test_gateway_precedence_flag_then_env_then_built_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(client.ENV_GATEWAY, raising=False)
    assert client.resolve_gateway() == client.DEFAULT_GATEWAY
    assert client.gateway_base(client.DEFAULT_GATEWAY).startswith("https://")
    monkeypatch.setenv(client.ENV_GATEWAY, "https://env.example.com")
    assert client.resolve_gateway() == "https://env.example.com"
    assert client.resolve_gateway("https://flag.example.com") == "https://flag.example.com"


def test_submit_uses_the_built_in_gateway_by_default(
    out: Path, gateway: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(client.ENV_GATEWAY, raising=False)
    monkeypatch.setattr(client, "DEFAULT_GATEWAY", gateway)  # never the live one in tests
    assert cli.main(["--out", str(out), "submit", "--yes"]) == 0
    assert _Gateway.received[0][:2] == ("POST", "/api/reports")
    assert json.loads((out / client.SUBMISSIONS).read_text())["gateway"] == gateway


def test_missing_token_gets_a_hint(
    out: Path, gateway: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _Gateway.status = 401
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes"]) == cli.EXIT_ERROR
    assert "set HWB_GATEWAY_TOKEN" in capsys.readouterr().err


def test_oversized_report_is_refused_before_sending(
    out: Path, gateway: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(client, "MAX_UPLOAD_BYTES", 1000)
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes"]) == cli.EXIT_ERROR
    assert _Gateway.received == []
    assert "over the gateway's 1000-byte limit" in capsys.readouterr().err


def test_public_gateway_needs_no_token(
    out: Path, gateway: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(client.ENV_TOKEN, raising=False)
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes"]) == 0
    assert "Authorization" not in _Gateway.received[0][2]


def test_sharing_can_be_switched_off(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(client.ENV_GATEWAY, "off")
    assert not client.sharing_enabled()
    assert cli.main(["--out", str(out), "submit", "--yes"]) == cli.EXIT_ERROR


def test_rate_limit_says_when_to_retry(
    out: Path, gateway: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _Gateway.status = 429
    assert cli.main(["--out", str(out), "submit", "--gateway", gateway, "--yes"]) == cli.EXIT_ERROR
    assert "(retry in 405 s)" in capsys.readouterr().err
