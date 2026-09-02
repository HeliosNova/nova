"""A judge that can move (audit 2026-09-01).

Measured: output_eval graded output[:3000] of ~8k-char digests on plausibility,
floored any sub-8 score whose critique it could not verify back to 8, and sat at
mean 9.59 / stdev 0.20 over 40 grades. A digest carrying a 2020 bullet, an
off-topic bullet and 'the launched a …' past char 3,000 scored 9.5 'none'.
Now: the whole digest is graded, unverifiable critiques drop the row instead of
inflating it, deterministic canaries cap the score no matter what the judge
says, and novelty against the domain dossier is recorded.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core import output_eval as oe


def _digest(tail: str = "") -> str:
    head = "## 🌐 open source — domain overview\n_read 14 sources: cnbc.com, bbc.com · September 01, 2026_\n"
    body = ("**Lead Development: Real news**\nThe consortium shipped its charter on August 30, 2026 (cnbc.com). " * 60)
    return head + body + tail


@pytest.mark.asyncio
async def test_judge_sees_the_whole_digest():
    seen = {}

    async def _fake(messages, **kwargs):
        seen["prompt"] = messages[0]["content"]
        return '{"relevance": 9, "facts": 9, "freshness": 9, "format": 9, "one_line_critique": "none"}'

    with patch("app.core.llm.invoke_nothink", _fake):
        out = await oe._grade_one("Domain Study: Open Source and GitHub", _digest("ZZ_TAIL_MARKER (bbc.com)."))
    assert out is not None
    assert "ZZ_TAIL_MARKER" in seen["prompt"], "the judge must read past the first 3,000 chars"


@pytest.mark.asyncio
async def test_unverifiable_critique_drops_the_row_instead_of_flooring():
    async def _fake(messages, **kwargs):
        return ('{"relevance": 5, "facts": 4, "freshness": 9, "format": 9, '
                '"one_line_critique": "mentions \\"Quantum Gravity Corp\\" which is fabricated"}')

    with patch("app.core.llm.invoke_nothink", _fake):
        out = await oe._grade_one("Domain Study: X", _digest())
    assert out is None


def test_canaries_catch_stale_year_artifact_and_pseudo_citation():
    text = _digest(
        "* **Apache httpd share:** Apache httpd held 24% of the server market in April 2020 (wikipedia.org).\n"
        "* The launched a **$10 million initiative** (cnbc.com).\n"
        "* More than one-sixth of the population was displaced (deep analysis: Historical Context).\n")
    flags = oe._canaries(text, year=2026)
    assert {"stale-year", "artifact", "pseudo-citation"} <= set(flags)
    assert oe._canaries(_digest(), year=2026) == []


def test_context_years_do_not_trip_the_stale_canary():
    text = _digest("* Since 2024 the foundation has grown; on August 30, 2026 it shipped (cnbc.com).\n")
    assert "stale-year" not in oe._canaries(text, year=2026)


@pytest.mark.asyncio
async def test_canary_caps_the_score_whatever_the_judge_says():
    async def _fake(messages, **kwargs):
        return '{"relevance": 10, "facts": 10, "freshness": 10, "format": 10, "one_line_critique": "none"}'

    text = _digest("* Apache httpd held 24% of the server market in April 2020 (wikipedia.org).\n")
    with patch("app.core.llm.invoke_nothink", _fake):
        out = await oe._grade_one("Domain Study: X", text)
    assert out["facts"] <= 6 and "canary" in out["critique"]


def test_novelty_against_dossier():
    digest = "Airbus delivered 63 aircraft in August. Boeing halted the 737 line. Safran raised guidance."
    same = "Airbus delivered 63 aircraft in August. Boeing halted the 737 line. Safran raised guidance."
    assert oe._novelty(digest, same) is not None and oe._novelty(digest, same) < 0.2
    assert oe._novelty(digest, "Unrelated prior understanding about lithium mining in Chile.") > 0.8
    assert oe._novelty(digest, "") is None
