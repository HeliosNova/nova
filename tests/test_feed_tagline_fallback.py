"""Feed taglines are a fallback now, not a reason to skip enrichment.

Owner report 2026-09-04: Product Hunt, SEC, GitHub Security, FDA, Hacker News
and Research Frontiers "show hyperlinks instead of report". For Product Hunt the
cause was a threshold: title feeds were let through at >=30 characters because
their tagline is item-specific, and clearing it before a fetch that might flake
would leave a bare link. Marketing taglines clear 30 comfortably - "Your Slack
org chart, built by everyone in it." is 45 - so 14 of 15 items were never
enriched and the row was a title, a link and a slogan.

The threshold protected the wrong thing. Every feed now needs 60 characters of
prose, and the item-specific text is SET ASIDE and restored if enrichment comes
back empty. Boilerplate, echoed titles and URL fragments are not restored -
killing those is what the clear was written for.

Every exit after the clear must restore, which is the part worth testing: three
of the four returns in that function are error paths.
"""
from __future__ import annotations

import pytest

import app.monitors.deep_research as dr
import app.monitors.domain_study_runner as dsr


class _Item:
    def __init__(self, title, url, summary):
        self.title = title
        self.url = url
        self.summary = summary
        self.meta = {}
        self.source_host = "producthunt.com"



@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Seal every fetch path.

    _enrich_thin_native_items falls back to deep_research._fetch_body with
    allow_bypass=True, which it imports INSIDE the function - so patching the
    runner's namespace misses it and the test reaches the live internet, twice,
    with a backoff. That is what these tests were doing: 17 seconds of real
    requests, and enough leftover client state to fail an unrelated HTTP test
    later in the suite.
    """
    async def _no_page(url, body_chars=3000):
        return None, ""

    async def _no_body(url, **kw):
        return ""

    monkeypatch.setattr(dsr, "_fetch_page_date", _no_page)
    monkeypatch.setattr(dr, "_fetch_body", _no_body)


TAGLINE = "Your Slack org chart, built by everyone in it."
BOILER = "Read more about this launch on Product Hunt today."


def _items():
    return [_Item("Snitch", "https://www.producthunt.com/products/snitch-4", TAGLINE)]


@pytest.mark.asyncio
async def test_a_tagline_no_longer_exempts_an_item_from_enrichment(monkeypatch):
    """45 characters cleared the old 30-char bar and skipped enrichment."""
    seen = {}

    async def _record(url, body_chars=3000):
        seen["fetched"] = url
        return None, ""          # no body -> the "0 usable bodies" exit

    monkeypatch.setattr(dsr, "_fetch_page_date", _record)
    items = _items()
    await dsr._enrich_thin_native_items("Product Hunt Trending", "product hunt", items)
    assert seen.get("fetched"), "the item was never treated as thin"


@pytest.mark.asyncio
async def test_the_tagline_comes_back_when_enrichment_finds_nothing(monkeypatch):
    """The failure the old threshold was guarding against, now handled."""
    items = _items()
    await dsr._enrich_thin_native_items("Product Hunt Trending", "product hunt", items)
    assert items[0].summary == TAGLINE, "a bare link is worse than a thin tagline"


@pytest.mark.asyncio
async def test_boilerplate_is_still_killed_not_restored(monkeypatch):
    """The clear exists to stop one survivor of a repeated line rendering
    alongside real summaries; restoring it would undo that."""
    items = [_Item(f"Launch {i}", f"https://example.com/{i}", BOILER) for i in range(4)]
    await dsr._enrich_thin_native_items("Product Hunt Trending", "product hunt", items)
    assert all(not it.summary for it in items), "boilerplate came back"


@pytest.mark.asyncio
async def test_a_summary_echoing_its_title_is_not_restored(monkeypatch):
    items = [_Item("Snitch", "https://example.com/x", "Snitch")]
    await dsr._enrich_thin_native_items("Product Hunt Trending", "product hunt", items)
    assert not items[0].summary


@pytest.mark.asyncio
async def test_sec_is_still_left_alone(monkeypatch):
    """EDGAR index pages carry no prose; Form 4 XML is where its signal is."""
    async def _boom(*a, **kw):
        raise AssertionError("SEC must not be fetched for enrichment")

    monkeypatch.setattr(dsr, "_fetch_page_date", _boom)
    items = [_Item("VALVOLINE INC", "https://www.sec.gov/x", "")]
    await dsr._enrich_thin_native_items("SEC Insider Trading", "sec", items)
    assert items[0].summary == ""
