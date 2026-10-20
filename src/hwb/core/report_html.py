"""report.html: the prebuilt single-file React viewer (ui/) with every run's report embedded.

There is ONE report.html per output directory (``--out``, default hwb-runs/). After each run
it is regenerated from all ``*/report.json`` beside it, so operators open a single file and
switch between an overview and the individual runs with tabs.

The viewer is built at development time (`make ui`, needs Node) into hwb/assets/index.html,
with all JS and CSS inlined, and ships as package data. At run time we only drop the
reports into its JSON slot, so the rig needs nothing beyond Python and the page needs
no network.
"""

from __future__ import annotations

import functools
import json
import logging
from importlib import resources
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TEMPLATE = "index.html"
INDEX_HTML = "report.html"
SLOT = '<script id="hwb-report" type="application/json">__HWB_REPORT__</script>'


@functools.lru_cache(maxsize=1)
def template() -> str:
    text = (resources.files("hwb") / "assets" / TEMPLATE).read_text(encoding="utf-8")
    if text.count(SLOT) != 1:
        raise ValueError(
            f"report template hwb/assets/{TEMPLATE} has no data slot; rebuild with make ui"
        )
    return text


def embed_json(payload: dict[str, Any]) -> str:
    # Inside <script> only "</script" or "<!--" could end the element early; escaping every
    # "<" rules both out, and JSON.parse turns \u003c back into "<".
    return json.dumps(payload, default=str, ensure_ascii=False).replace("<", "\\u003c")


def render_html(payload: dict[str, Any]) -> str:
    return template().replace(SLOT, SLOT.replace("__HWB_REPORT__", embed_json(payload)), 1)


def collect_runs(base: Path) -> list[dict[str, Any]]:
    """Every readable ``<base>/<run>/report.json``, oldest first; unreadable ones are skipped."""
    found: list[tuple[str, str, dict[str, Any]]] = []
    for path in base.glob("*/report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("skipping %s in %s: %s", path, INDEX_HTML, exc)
            continue
        if isinstance(report, dict):
            found.append((str(report.get("started_at", "")), path.parent.name, report))
    return [
        {"id": run_id, "report": report} for _, run_id, report in sorted(found, key=lambda f: f[:2])
    ]


def write_index(base: Path) -> str:
    """The combined report.html for every run under ``base``."""
    return render_html({"runs": collect_runs(base)})
