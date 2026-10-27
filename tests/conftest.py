from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _own_hwb_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's ~/.config/hwb settings out of every test."""
    monkeypatch.setenv("HWB_CONFIG", str(tmp_path / "hwb-config.json"))
    monkeypatch.delenv("HWB_SUBMIT", raising=False)
