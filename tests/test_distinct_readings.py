"""X1 (plan 2026-10-09): list distinct readings before reasoning.

The synthesis prompt tags each finding with a corroboration count, and the
count is what the model reasons from — three articles that are one
syndicated wire story read as three confirmations. `distinct_readings` lists
ONE line per independent cluster per distinctive figure (host · figure ·
date) so the model sees the readings, not the article count. Wired behind
ENABLE_DISTINCT_READINGS (default off) until the frozen-topic replay passes
rule X1 (support within 0.02 of baseline AND cited-copy share falls).
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

from app.monitors import deep_research as dr

WIRE = ("Acme reported revenue of $1,250,000 for the quarter, the company said on Monday. "
        "Shares rose after the report, which beat analyst expectations by a wide margin. "
        "The chief executive said demand remained strong across every region it serves.")
ORIGINAL = ("In its own analysis the Gazette notes Acme's $1,250,000 quarter and a separate "
            "contract worth 3,400,000 dollars signed with the state, which the wire did not carry.")
ARTICLES = [
    ("Acme beats", "https://reuters.com/acme", WIRE),
    ("Acme beats (syndicated)", "https://mirror.example/acme", WIRE),
    ("Acme analysis", "https://gazette.example/acme", ORIGINAL),
]
FINDINGS = [(t, u, b[:120]) for t, u, b in ARTICLES]


def test_copies_collapse_to_one_reading_per_independent_cluster():
    with patch.dict(dr._PUB_DATES, {"https://reuters.com/acme": "2026-10-06",
                                    "https://gazette.example/acme": "2026-10-07"}, clear=True):
        block = dr.distinct_readings(FINDINGS, ARTICLES)
    lines = [l for l in block.splitlines() if " \u00b7 " in l]
    rev = [l for l in lines if "$1,250,000" in l]
    assert len(rev) == 2, lines
    assert any(l.startswith("reuters.com") and l.endswith("2026-10-06") for l in rev)
    assert any(l.startswith("gazette.example") and l.endswith("2026-10-07") for l in rev)
    assert not any("mirror.example" in l for l in lines)
    contract = [l for l in lines if "3,400,000" in l]
    assert len(contract) == 1 and contract[0].startswith("gazette.example")
    assert "Reason from the readings, not from the count of articles" in block


def test_no_distinctive_figure_yields_no_block():
    arts = [("a", "https://a.example/1", "Prices rose 8% and the market liked it.")]
    assert dr.distinct_readings([("a", "https://a.example/1", "x")], arts) == ""


def test_unknown_date_is_marked_not_invented():
    with patch.dict(dr._PUB_DATES, {}, clear=True):
        block = dr.distinct_readings(FINDINGS, ARTICLES)
    assert "n/d" in block


def test_wired_into_the_synthesis_prompt_behind_the_flag():
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "ENABLE_DISTINCT_READINGS" in src and "distinct_readings(" in src
    from app.config import config
    assert hasattr(config, "ENABLE_DISTINCT_READINGS")
