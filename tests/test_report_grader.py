"""Deterministic coverage for the FACT (citation-support) half of the RACE/FACT
report grader. RACE is LLM-judged and covered by the live harness, not unit tests."""
from __future__ import annotations

from app.monitors import report_grader as rg


def test_hosts_parsed_from_report_header():
    r = ("## 🌐 finance — domain overview\n"
         "_read 3 sources: apnews.com, www.cnbc.com, fool.com · 6 facts · July 01_\n\ntext")
    assert rg._hosts_from_report(r) == ["apnews.com", "cnbc.com", "fool.com"]


def test_fact_score_counts_valid_and_fabricated():
    report = (
        "_read 2 sources: reuters.com, cnbc.com · July 01_\n\n"
        "The Federal Reserve held rates at 3.5% to 3.75% this week (reuters.com).\n"    # valid
        "Oracle shares fell 11 percent after earnings missed expectations (cnbc.com).\n"  # valid
        "A secret merger worth 5 billion dollars was agreed in private (bloomberg.com).\n"  # fabricated host
        "Analysts remain broadly cautious about the second half outlook overall.\n"      # no cite
    )
    f = rg.fact_score(report)
    assert f["n_factual"] >= 3
    assert 0.0 < f["support"] <= 1.0            # some claims cite a READ host
    assert f["fabricated_rate"] > 0.0           # bloomberg.com wasn't read → flagged
    # explicit read_hosts override works too
    f2 = rg.fact_score(report, read_hosts=["reuters.com", "cnbc.com", "bloomberg.com"])
    assert f2["fabricated_rate"] == 0.0         # now every cited host counts as read


def test_cites_in_handles_reliability_tags_and_multihost():
    # the real engine cites as (host · reliability-tag), sometimes multi-host — must parse
    assert rg._cites_in("The Fed held rates (reuters.com).") == ["reuters.com"]
    assert rg._cites_in("Dot plot shows two hikes (theglobeandmail.com · primary-doc).") == \
        ["theglobeandmail.com"]
    assert rg._cites_in("Both confirm (fool.com · single/unverified; ap.org · wire).") == \
        ["fool.com", "ap.org"]
    assert rg._cites_in("A ratio (33%) and a note (e.g. later) with no host.") == []


def test_fact_score_reads_tagged_citations():
    report = ("_read 2 sources: theglobeandmail.com, fool.com · July 01_\n\n"
              "The Federal Reserve Dot Plot shows one-third of members expect two hikes "
              "(theglobeandmail.com · primary-doc).\n"
              "This is the fifth rate cycle since 1999 (fool.com · single/unverified).\n")
    f = rg.fact_score(report)
    assert f["citation_rate"] == 1.0        # both factual sentences ARE cited (tag form)
    assert f["support"] == 1.0 and f["fabricated_rate"] == 0.0


def test_fact_score_empty_report_is_safe():
    f = rg.fact_score("")
    assert f["n_factual"] == 0 and f["support"] == 0.0
