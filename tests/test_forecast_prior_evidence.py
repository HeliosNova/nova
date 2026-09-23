"""Evidence for minting is what was knowable at the time (2026-09-23).

The bake-off on 144 resolved claims showed the stated number, written with
the digest in context, at Brier 0.229 against 0.31-0.33 for every blind
re-estimate. Context is the lever. Two candidate levers for the mint, both
measured leak-free on the resolved record before either ships:

  * gather_prior_evidence — the mirror of the grader's evidence search; it
    keeps only results published on or before the claim, and in strict mode
    drops undated ones, so a replay cannot read the outcome.
  * reference_class — one call naming the outside view (class + base rate)
    that the confidence samples are anchored on.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import forecasts as fc


def _r(url, date="", title="t", snippet="s"):
    return SimpleNamespace(url=url, title=title, snippet=snippet, engine="x", published_date=date)


RESULTS = [
    _r("https://reuters.com/before", "2026-06-01", title="before"),
    _r("https://apnews.com/after", "2026-08-15", title="after"),
    _r("https://bbc.com/undated", "", title="undated"),
    _r("https://junk.example/x", "2026-06-02", title="junk"),
    _r("https://reuters.com/before", "2026-06-01", title="duplicate"),
]


def _authority(host):
    return 0.1 if "junk" in host else 0.9


@pytest.mark.asyncio
async def test_strict_replay_keeps_only_dated_prior_evidence():
    with patch("app.tools.native_search.search", AsyncMock(return_value=RESULTS)), \
         patch("app.core.source_authority.authority", _authority):
        block = await fc.gather_prior_evidence("claim", as_of="2026-06-23 18:03:27", strict_dates=True)
    assert "before" in block
    assert "after" not in block and "undated" not in block and "junk" not in block
    assert block.count("reuters.com") == 1          # the duplicate url is folded


@pytest.mark.asyncio
async def test_lenient_mode_keeps_undated_but_never_later_evidence():
    with patch("app.tools.native_search.search", AsyncMock(return_value=RESULTS)), \
         patch("app.core.source_authority.authority", _authority):
        block = await fc.gather_prior_evidence("claim", as_of="2026-06-23 18:03:27")
    assert "before" in block and "undated" in block
    assert "after" not in block and "junk" not in block


@pytest.mark.asyncio
async def test_live_minting_has_no_cutoff_only_the_credibility_filter():
    with patch("app.tools.native_search.search", AsyncMock(return_value=RESULTS)), \
         patch("app.core.source_authority.authority", _authority):
        block = await fc.gather_prior_evidence("claim")
    for t in ("before", "after", "undated"):
        assert t in block
    assert "junk" not in block


@pytest.mark.asyncio
async def test_a_failed_search_yields_nothing_rather_than_a_guess():
    with patch("app.tools.native_search.search", AsyncMock(side_effect=RuntimeError("down"))):
        assert await fc.gather_prior_evidence("claim", as_of="2026-06-23") == ""


@pytest.mark.asyncio
async def test_reference_class_returns_the_outside_view():
    with patch.object(fc.llm, "invoke_nothink",
                      AsyncMock(return_value='{"reference_class": "bills passing one chamber within a month", "base_rate": 0.15}')):
        ref = await fc.reference_class("The House passes the bill by October 1", "2026-10-01")
    assert ref == ("bills passing one chamber within a month", 0.15)
    block = fc.outside_view_block(ref)
    assert block.startswith("REFERENCE CLASS: bills passing") and "BASE RATE for that class: 0.15" in block


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ['{"reference_class": "x", "base_rate": 1.7}', '{"base_rate": 0.3}', "not json", ""])
async def test_an_unusable_reference_answer_is_none(reply):
    with patch.object(fc.llm, "invoke_nothink", AsyncMock(return_value=reply)):
        assert await fc.reference_class("claim") is None
    assert fc.outside_view_block(None) == ""
