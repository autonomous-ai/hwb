import re
from pathlib import Path

import hwb


def test_pyproject_version_matches_package() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert m is not None
    assert m.group(1) == hwb.__version__
