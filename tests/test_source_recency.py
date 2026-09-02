"""Recency and source class as first-class read signals (audit 2026-09-01).

Measured: wikipedia (0.834), investopedia (0.926), cnet (0.952), howtogeek
(0.910) ranked as 'quality' sources; any .edu host — academia.edu included —
ranked with the wires as primary; no read article carried a publish date; a
September digest led with a July event and carried a 2020 bullet.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.monitors import deep_research as dr


def test_url_dates_are_extracted():
    assert dr._url_date("https://www.reuters.com/world/2026/08/31/foo-bar/") == "2026-08-31"
    assert dr._url_date("https://example.com/news/2026-08-30-foo") == "2026-08-30"
    assert dr._url_date("https://example.com/2026/08/foo") is None
    assert dr._url_date("https://example.com/foo") is None


def test_reference_sites_are_background_not_quality():
    assert dr._is_reference_host("en.wikipedia.org")
    assert dr._is_reference_host("investopedia.com")
    assert not dr._is_reference_host("reuters.com")
    assert dr._source_quality("https://en.wikipedia.org/wiki/Apache_HTTP_Server") <= 1.0
    assert dr._source_quality("https://www.reuters.com/x") == 3.0


def test_blanket_edu_is_not_primary():
    assert dr._source_quality("https://www.academia.edu/12345/some-paper") < 3.0
    assert dr._source_quality("https://www.nasa.gov/press") == 3.0


def test_lead_on_a_reference_site_is_flagged_not_anchored():
    text = ("**Lead Development: Something big**\nA claim about the world (en.wikipedia.org).\n\n"
            "**Secondary Developments**\n* Other (reuters.com).")
    out, gated = dr._gate_lead_credibility(text)
    assert gated and "Sourcing note" in out
    text2 = text.replace("(en.wikipedia.org)", "(reuters.com)")
    out2, gated2 = dr._gate_lead_credibility(text2)
    assert not gated2


def test_stale_lead_gets_a_freshness_note(monkeypatch):
    monkeypatch.setattr(dr, "_NOW", lambda: datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc))
    dr._PUB_DATES.clear()
    dr._PUB_DATES["https://reuters.com/a"] = "2026-07-20"
    arts = [("t", "https://reuters.com/a", "body")]
    text = ("**Lead Development: Old news**\nSomething happened (reuters.com).\n\n"
            "**Secondary Developments**\n* Other (bbc.com).")
    out, _ = dr._gate_lead_credibility(text, articles=arts)
    assert "Freshness note" in out
    dr._PUB_DATES["https://reuters.com/a"] = "2026-08-31"
    out2, _ = dr._gate_lead_credibility(text, articles=arts)
    assert "Freshness note" not in out2


def test_read_bodies_records_publish_dates():
    from types import SimpleNamespace
    dr._PUB_DATES.clear()
    r = SimpleNamespace(url="https://www.reuters.com/world/2026/08/31/foo/", title="t", published_date="")
    dr._record_pub_date(r)
    assert dr._PUB_DATES[r.url] == "2026-08-31"
    r2 = SimpleNamespace(url="https://bbc.com/x", title="t", published_date="2026-08-30T10:00:00Z")
    dr._record_pub_date(r2)
    assert dr._PUB_DATES[r2.url].startswith("2026-08-30")
