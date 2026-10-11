"""R2 (plan 2026-10-09): the SearXNG engine list carries only engines that load.

SearXNG logged "Cannot load engine" for ecosia, presearch and reddit at every
start (they sat in the list and answered nothing), wikidata added latency for
entity stubs, and default_lang "en" let engines pick regional editions.
DuckDuckGo is re-enabled as a 48 h trial under rule R2 (R1's counters decide).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "searxng" / "settings.yml"
EXAMPLE = ROOT / "searxng" / "settings.yml.example"
NEVER = {"ecosia", "presearch", "reddit"}
DISABLED = {"wikidata"}   # a DEFAULT engine: only an explicit disabled entry stops it loading


def _strip_secret(text: str) -> str:
    return re.sub(r'secret_key: ".*"', 'secret_key: "X"', text)


def test_live_and_example_differ_only_on_the_secret_key():
    assert _strip_secret(LIVE.read_text(encoding="utf-8")) == _strip_secret(EXAMPLE.read_text(encoding="utf-8"))


def test_no_engine_is_listed_twice_and_none_that_fail_to_load():
    doc = yaml.safe_load(LIVE.read_text(encoding="utf-8"))
    names = [e["name"] for e in doc["engines"]]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
    assert not (set(names) & NEVER), set(names) & NEVER
    for e in doc["engines"]:
        if e["name"] in DISABLED:
            assert e.get("disabled") is True, e["name"]
    assert DISABLED <= set(names)


def test_language_and_the_ddg_trial():
    doc = yaml.safe_load(LIVE.read_text(encoding="utf-8"))
    assert doc["search"]["default_lang"] == "en-US"
    ddg = next(e for e in doc["engines"] if e["name"] == "duckduckgo")
    assert ddg.get("disabled") is False
