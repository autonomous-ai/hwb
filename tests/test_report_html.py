import json
from html.parser import HTMLParser

from hwb.core.report_html import SLOT, embed_json, render_html, template


class _Slot(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.data: list[str] = []
        self._on = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._on = tag == "script" and ("id", "hwb-report") in attrs

    def handle_endtag(self, tag: str) -> None:
        self._on = False

    def handle_data(self, data: str) -> None:
        if self._on:
            self.data.append(data)


def _embedded(page: str) -> object:
    p = _Slot()
    p.feed(page)
    return json.loads("".join(p.data))


def test_template_is_packaged_with_one_slot() -> None:
    page = template()
    assert page.count(SLOT) == 1
    assert page.lstrip().lower().startswith("<!doctype html>")


def test_payload_cannot_break_out_of_the_script_element() -> None:
    evil = "</script><script>alert(1)</script><!--"
    payload = {"host": evil, "results": [{"summary": evil, "hint": "ünïcode ✓"}]}
    page = render_html(payload)
    assert "</script><script>alert(1)" not in page
    assert "<" not in embed_json(payload)
    assert _embedded(page) == payload


def test_non_json_values_are_stringified() -> None:
    from pathlib import Path

    assert _embedded(render_html({"path": Path("/tmp/x")})) == {"path": "/tmp/x"}
