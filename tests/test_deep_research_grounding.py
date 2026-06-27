"""Deterministic regression coverage for the deep-research grounding spine.

These are the anti-hallucination / citation-grounding helpers in
`app/monitors/deep_research.py` — the engine's core honesty guarantee. They are
pure (or deterministic-async, no LLM), so they are unit-testable red-on-regression
without a model. This file closes the largest test blind spot found in the
2026-06-24 audit (the 1527-line engine had zero unit coverage).
"""
from __future__ import annotations

import pytest

from app.monitors import deep_research as dr


# --- _strip_fake_citations: drop lines that cite only UNREAD outlets ----------

def test_strip_fake_citations_drops_unread_outlet():
    hosts = ["reuters.com", "cnbc.com"]  # what we actually read
    text = (
        "Lead: the deal closed today (reuters.com).\n"
        "Fabricated attribution: a secret clause (bloomberg.com).\n"
        "Connective prose with no citation at all."
    )
    out = dr._strip_fake_citations(text, hosts)
    assert "reuters.com" in out                 # cited a read host → kept
    assert "bloomberg.com" not in out           # cited only an unread host → dropped
    assert "Connective prose" in out            # uncited line → kept


def test_strip_fake_citations_matches_subdomain_of_read_host():
    # A '(slashdot.org)' citation must match a read host 'tech.slashdot.org'.
    out = dr._strip_fake_citations("Item (slashdot.org).", ["tech.slashdot.org"])
    assert "slashdot.org" in out


# --- _unverified_numbers: flag magnitudes absent from the source corpus -------

def test_unverified_numbers_flags_absent_magnitude_keeps_present():
    corpus = "the breach affected 86,644 firewalls in total".lower()
    corpus_nc = corpus.replace(",", "")
    text = "It hit 86,644 firewalls but a report inflated it to 110 million devices."
    bad = dr._unverified_numbers(text, corpus, corpus_nc)
    assert any("110" in b for b in bad)          # absent from sources → flagged
    assert not any("86,644" in b for b in bad)   # present in sources → not flagged


def test_unverified_numbers_year_vs_count_collision():
    # '2030 employees' must be flagged even though '2030' appears as a YEAR in the
    # sources (the bare-value check passes by matching the year mention).
    corpus = "the roadmap runs to 2030 and the firm had 20 to 30 employees".lower()
    bad = dr._unverified_numbers(text="It now employs 2030 employees (x.com).",
                                 corpus=corpus, corpus_nc=corpus.replace(",", ""))
    assert any("2030 employees" in b for b in bad)


def test_unverified_numbers_ignores_small_bare_integers():
    corpus = "nothing relevant here".lower()
    bad = dr._unverified_numbers("There were 7 incidents and 3 reports.",
                                 corpus, corpus)
    assert bad == []   # small bare integers are not magnitude figures


# --- _orphan_terms: distinctive terms absent from every source ----------------

def test_orphan_terms_flags_absent_acronym():
    corpus = "abbvie acquired the biotech for its depression pipeline".lower()
    orphans = dr._orphan_terms("AbbVie bought the firm for its LSD depression therapy.", corpus)
    assert "LSD" in orphans


def test_orphan_terms_spares_common_acronyms_and_paraphrase():
    corpus = "the federal reserve held rates as the us and eu agreed on ai rules".lower()
    orphans = dr._orphan_terms("The Federal Reserve held rates; the US and EU agreed on AI rules.", corpus)
    assert orphans == []   # 'Federal Reserve' present; US/EU/AI are common acronyms


# --- _tidy_citations: cosmetic citation cleanup -------------------------------

def test_tidy_citations_strips_dollar_wrapped_citation():
    assert dr._tidy_citations("Gains ($quiverquant.com$).") == "Gains (quiverquant.com)."


def test_tidy_citations_drops_leaked_title_fragment():
    # A domain-less parenthetical ending in '...' is a leaked source title → dropped.
    assert dr._tidy_citations("News here (some title fragment...).") == "News here."


def test_tidy_citations_leaves_real_money_untouched():
    out = dr._tidy_citations("Revenue was $9.2m this quarter.")
    assert "$9.2m" in out


# --- _corroborate_numbers: deterministic cross-source ✓ badge (async, no LLM) -

@pytest.mark.asyncio
async def test_corroborate_numbers_badges_two_source_agreement():
    articles = [
        ("t1", "http://a.com/1", "The deal was valued at $4,200,000,000 in total."),
        ("t2", "http://b.com/2", "Analysts pegged the acquisition at $4,200,000,000."),
    ]
    text = "The acquisition was valued at $4,200,000,000 by multiple outlets."
    out, n = await dr._corroborate_numbers(text, articles)
    assert n >= 1 and "✓" in out


@pytest.mark.asyncio
async def test_corroborate_numbers_no_badge_for_single_source():
    articles = [
        ("t1", "http://a.com/1", "Only this outlet says $4,200,000,000."),
        ("t2", "http://b.com/2", "An unrelated story about weather."),
    ]
    out, n = await dr._corroborate_numbers("Valued at $4,200,000,000.", articles)
    assert n == 0 and "✓" not in out
