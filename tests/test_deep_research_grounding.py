"""Deterministic regression coverage for the deep-research grounding spine.

These are the anti-hallucination / citation-grounding helpers in
`app/monitors/deep_research.py` — the engine's core honesty guarantee. They are
pure (or deterministic-async, no LLM), so they are unit-testable red-on-regression
without a model. This file closes the largest test blind spot found in the
2026-06-24 audit (the 1527-line engine had zero unit coverage).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

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
async def test_corroborate_numbers_returns_set_without_mutating_text():
    # The ✓ inline badge was removed (read as cruft + rubber-stamped misframed figures);
    # the function now returns the confirmed-figure SET and leaves the prose untouched.
    articles = [
        ("t1", "http://a.com/1", "The deal was valued at $4,200,000,000 in total."),
        ("t2", "http://b.com/2", "Analysts pegged the acquisition at $4,200,000,000."),
    ]
    text = "The acquisition was valued at $4,200,000,000 by multiple outlets."
    out, confirmed = await dr._corroborate_numbers(text, articles)
    assert out == text and "✓" not in out                     # prose NOT mutated
    assert any("4,200,000,000" in f for f in confirmed)        # figure in the set (for Lever A)


@pytest.mark.asyncio
async def test_corroborate_numbers_empty_set_for_single_source():
    articles = [
        ("t1", "http://a.com/1", "Only this outlet says $4,200,000,000."),
        ("t2", "http://b.com/2", "An unrelated story about weather."),
    ]
    out, confirmed = await dr._corroborate_numbers("Valued at $4,200,000,000.", articles)
    assert confirmed == set() and "✓" not in out


# --- _orphan_terms: single CamelCase compound coinages (2026-06-27 polish) --------

def test_orphan_terms_flags_camelcase_coinage():
    # The "ParyonUSD" class: a coined CamelCase token absent from every source
    # (sources say the real "pUSD"). Slips the acronym + multi-word-phrase patterns.
    corpus = "polymarket lost funds held in pusd on polygon, swapped for ether".lower()
    orphans = dr._orphan_terms("Funds initially held in ParyonUSD on the Polygon network.", corpus)
    assert "ParyonUSD" in orphans


def test_orphan_terms_spares_common_camelcase():
    corpus = "the company shipped a new release".lower()  # 'GitHub' absent on purpose
    orphans = dr._orphan_terms("The repo is hosted on GitHub.", corpus)
    assert "GitHub" not in orphans   # whitelisted brand, never flagged


def test_orphan_terms_compound_reprieve_when_parts_present():
    # 'frontier' + 'tech' both in sources -> reformat of a real term, not a coinage.
    corpus = "the new frontier in tech, and the tech frontier keeps expanding".lower()
    orphans = dr._orphan_terms("Announced by FrontierTech this week.", corpus)
    assert "FrontierTech" not in orphans


# --- _unverified_numbers: measurement figures in the hundreds (185-miles class) ----

def test_unverified_numbers_flags_absent_measurement():
    corpus = "the satellite will be boosted to about 370 miles to avoid reentry".lower()
    bad = dr._unverified_numbers("It will raise altitude to 185 miles.", corpus, corpus.replace(",", ""))
    assert any("185 miles" in b for b in bad)


def test_unverified_numbers_keeps_present_measurement():
    corpus = "boosted to 370 miles to avoid reentry".lower()
    bad = dr._unverified_numbers("Raised to 370 miles altitude.", corpus, corpus.replace(",", ""))
    assert not any(b.startswith("370") for b in bad)


def test_unverified_numbers_ignores_small_measurement():
    # 1-2 digit measurements are too collision-prone to flag; only hundreds+.
    corpus = "no distances mentioned at all here".lower()
    bad = dr._unverified_numbers("The storm was 8 km tall.", corpus, corpus)
    assert bad == []


# --- source blocklist: weak syndication farms (2026-06-27 audit) ------------------

def test_blocked_hosts_cover_syndication_farm():
    assert dr._blocked("https://kenyastar.com/pax-silica-article")
    assert dr._blocked("http://philippinetimes.com/x")
    assert dr._blocked("https://www.vietnamtribune.com/y")
    # legitimate outlets must still pass
    assert not dr._blocked("https://www.reuters.com/world/")
    assert not dr._blocked("https://www.cnbc.com/2026/06/27/x.html")


# --- _numeric_grafts: cited-source-anchored numeric mis-attribution (Lever B) -----

def test_numeric_grafts_flags_misattributed_figure():
    # '$107.8B' is cited to a.com (which says $104.3B) but 107.8 lives in b.com (an
    # earlier peak) — present-elsewhere, absent-from-cited => mis-attributed.
    articles = [
        ("ETF flows", "http://a.com/1",
         "Spot bitcoin ETF assets fell from $104.3 billion to $82.8 billion over two weeks."),
        ("Market recap", "http://b.com/2",
         "At their March peak the ETFs had held $107.8 billion before the long decline began."),
    ]
    text = "ETF assets dropped from $107.8 billion to $82.8 billion over two weeks (a.com)."
    grafts = dr._numeric_grafts(text, articles)
    assert any("107.8" in g for g in grafts)          # wrong, from another source
    assert not any("82.8" in g for g in grafts)       # correct, in the cited source


def test_numeric_grafts_ignores_figure_in_cited_source():
    articles = [
        ("A", "http://a.com/1", "The company reported $4.2 billion in quarterly revenue."),
        ("B", "http://b.com/2", "An unrelated index rose three percent on the day."),
    ]
    text = "Revenue hit $4.2 billion in the quarter (a.com)."
    assert dr._numeric_grafts(text, articles) == []   # 4.2B is in the cited source


def test_numeric_grafts_skips_multi_cited_sentence():
    # Can't attribute a figure to one source when two are cited — skip (no false graft).
    articles = [
        ("A", "http://a.com/1", "Revenue was $104.3 billion last year."),
        ("B", "http://b.com/2", "Analysts had modeled $107.8 billion."),
    ]
    text = "The figure landed at $107.8 billion (a.com; b.com)."
    assert dr._numeric_grafts(text, articles) == []


# --- Lever A: bounded fresh-evidence verification of the lead's claims ------------

class _Res:
    def __init__(self, title, snippet):
        self.title, self.snippet, self.url = title, snippet, "http://x.com/1"


def test_lead_claims_extracts_numeric_lead_sentences():
    text = (
        "## crypto — domain overview\n_read 9 sources_\n\n"
        "**Lead Development**\n"
        "Bitcoin slid below $60,000 amid heavy outflows this week. Spot ETF assets dropped from "
        "$107.8 billion to $82.8 billion over two weeks (a.com). A qualitative aside with no figure.\n\n"
        "**Secondary Developments**\n* Something about $5 billion elsewhere (b.com)."
    )
    claims = dr._lead_claims(text, max_claims=3)
    assert any("107.8" in c for c in claims)            # distinctive lead figure captured
    assert all("Secondary" not in c and "elsewhere" not in c for c in claims)  # never past the lead
    assert all("(a.com)" not in c for c in claims)      # inline citations stripped


def test_lead_claims_empty_without_figures():
    text = "**Lead Development**\nA purely qualitative update with no specific numbers at all today.\n"
    assert dr._lead_claims(text) == []


@pytest.mark.asyncio
async def test_verify_lead_claims_appends_caveat_on_contradiction(monkeypatch):
    text = ("**Lead Development**\nSpot ETF assets dropped from $107.8 billion to $82.8 billion "
            "over two weeks (a.com).\n")
    from app.tools import native_search as _ns
    monkeypatch.setattr(_ns, "search", AsyncMock(return_value=[
        _Res("ETF assets near $104.3B", "Total spot bitcoin ETF assets stand around $104.3 billion."),
        _Res("Weekly crypto recap", "Markets declined over the week."),
    ]))
    monkeypatch.setattr(dr.llm, "invoke_nothink", AsyncMock(
        return_value='{"verdict":"contradicted","note":"Current sources put the figure near $104.3 billion."}'))
    monkeypatch.setattr(dr.llm, "extract_json_object",
                        lambda r: {"verdict": "contradicted", "note": "Current sources put the figure near $104.3 billion."})
    out, n = await dr._verify_lead_claims(text, "crypto")
    assert n >= 1
    assert "Fresh-check" in out and "104.3" in out
    assert out.startswith(text)        # body untouched; caveat only appended


def test_lead_claims_skips_corroborated_figures():
    # A figure already corroborated (≥2 sources, passed in via the set) is NOT a
    # fresh-check candidate; drift hides in single-source figures, so only the
    # un-corroborated one is extracted.
    text = ("**Lead Development**\n"
            "Bitcoin held above $60,000 on confirmed exchange data this week today. "
            "Gold futures meanwhile traded near $4,713.3 per ounce on a single data feed today.")
    claims = dr._lead_claims(text, corroborated={"$60,000"}, max_claims=3)
    assert any("4,713" in c for c in claims)           # single-source figure → checked
    assert not any("60,000" in c for c in claims)      # corroborated → skipped


# --- _deep_analyze: cluster findings into stories + analyze each in depth -----------

@pytest.mark.asyncio
async def test_deep_analyze_clusters_and_analyzes(monkeypatch):
    findings = [
        ("BTC drops", "http://a.com/1", "Bitcoin fell below 60000 this week."),
        ("ETF outflows", "http://b.com/2", "Spot bitcoin ETFs bled billions."),
        ("Gold rises", "http://c.com/3", "Gold hit a fresh record high."),
    ]
    calls = {"n": 0}

    async def fake(msgs, **k):
        calls["n"] += 1
        if "Group these" in msgs[0]["content"]:
            return '[{"title":"Crypto selloff","items":[0,1]},{"title":"Gold rally","items":[2]}]'
        return "Deep analysis: key facts, why it matters, what to watch."

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake)
    out = await dr._deep_analyze(findings, "Finance", "June 29, 2026")
    assert "### Crypto selloff" in out and "### Gold rally" in out
    assert calls["n"] == 3   # 1 clustering call + 2 per-story analysis calls


@pytest.mark.asyncio
async def test_deep_analyze_routes_to_synthesis_model(monkeypatch):
    # Lever C: a configured bigger model must be threaded to every analysis call.
    findings = [("a", "http://a.com/1", "x" * 60), ("b", "http://b.com/2", "y" * 60),
                ("c", "http://c.com/3", "z" * 60)]
    seen = {"models": []}

    async def fake(msgs, **k):
        seen["models"].append(k.get("model"))
        if "Group these" in msgs[0]["content"]:
            return '[{"title":"S","items":[0,1,2]}]'
        return "deep analysis text"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake)
    await dr._deep_analyze(findings, "X", "today", model="qwen3.6:27b")
    assert seen["models"] and all(m == "qwen3.6:27b" for m in seen["models"])


@pytest.mark.asyncio
async def test_deep_analyze_skips_when_few_findings(monkeypatch):
    called = {"n": 0}

    async def fake(*a, **k):
        called["n"] += 1
        return "x"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake)
    out = await dr._deep_analyze(
        [("t", "http://a.com/1", "body"), ("t2", "http://b.com/2", "body2")], "X", "today")
    assert out == "" and called["n"] == 0   # <3 findings → no deep analysis, no calls


@pytest.mark.asyncio
async def test_deep_analyze_reads_full_bodies_when_provided(monkeypatch):
    # Fix #1 (input starvation): the per-story DEEP analysis must reason over the
    # full article BODY, not the 240-token finding stub. When bodies are supplied,
    # the analysis prompt must carry the body text — that is where depth comes from.
    findings = [
        ("BTC drops", "http://a.com/1", "Bitcoin fell."),       # short finding stub
        ("ETF outflows", "http://b.com/2", "ETFs bled."),
        ("Gold rises", "http://c.com/3", "Gold record."),
    ]
    bodies = {
        "http://a.com/1": "FULLBODYA " + ("bitcoin selloff detail " * 50),
        "http://b.com/2": "FULLBODYB " + ("etf outflow specifics " * 50),
        "http://c.com/3": "FULLBODYC " + ("gold rally context " * 50),
    }
    seen = {"analysis": []}

    async def fake(msgs, **k):
        content = msgs[0]["content"]
        if "Group these" in content:
            return '[{"title":"Crypto","items":[0,1]},{"title":"Gold","items":[2]}]'
        seen["analysis"].append(content)
        return "analysis text"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake)
    out = await dr._deep_analyze(findings, "Finance", "today", bodies=bodies)
    joined = "\n".join(seen["analysis"])
    # the analyst layer saw the FULL bodies, not just the finding stubs
    assert "FULLBODYA" in joined and "FULLBODYB" in joined and "FULLBODYC" in joined
    assert "### Crypto" in out and "### Gold" in out


@pytest.mark.asyncio
async def test_deep_analyze_falls_back_to_finding_without_body(monkeypatch):
    # When a body is unavailable for a url, the analysis falls back to the finding
    # text — never empty (backward-compatible with the no-bodies callers).
    findings = [("A", "http://a.com/1", "FINDING-A only"),
                ("B", "http://b.com/2", "FINDING-B only"),
                ("C", "http://c.com/3", "FINDING-C only")]
    seen = {"analysis": []}

    async def fake(msgs, **k):
        content = msgs[0]["content"]
        if "Group these" in content:
            return '[{"title":"S","items":[0,1,2]}]'
        seen["analysis"].append(content)
        return "ok"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake)
    await dr._deep_analyze(findings, "X", "today", bodies={})   # no bodies at all
    joined = "\n".join(seen["analysis"])
    assert "FINDING-A only" in joined and "FINDING-C only" in joined


# --- #3: strip-pass erosion guards (orphan headers + length floor + 27B rewrite) --

def test_drop_orphan_headers_removes_empty_section():
    text = ("**Lead Development**\nBig thing happened (reuters.com).\n\n"
            "**Secondary Developments**\n\n"      # body stripped by a pass → orphan
            "**Connections & bottom line**\nThe throughline (ap.com).")
    out = dr._drop_orphan_headers(text)
    assert "**Secondary Developments**" not in out          # dangling header removed
    assert "Big thing happened" in out                      # real sections kept
    assert "**Connections & bottom line**" in out and "throughline" in out


def test_drop_orphan_headers_keeps_inline_label():
    # A bold label with content on the SAME line is not a bare header — keep it.
    text = "**Lead Development** — the big story unfolded today (bbc.com)."
    assert dr._drop_orphan_headers(text) == text


def test_accept_correction_rejects_gutted_output():
    original = "**Lead**\n" + ("Substantive sentence with real detail. " * 30)  # >400 chars
    gutted = "**Lead**\nShort."                                                  # <50% length
    assert dr._accept_correction(original, gutted) is False                      # over-stripped
    faithful = original.replace("detail", "specifics")                           # same size
    assert dr._accept_correction(original, faithful) is True


def test_accept_correction_rejects_structureless_or_empty():
    assert dr._accept_correction("**X**\nbody", "plain prose, no structure") is False
    assert dr._accept_correction("**X**\nbody", "") is False


# --- broken-sentence repair (dropped-noun corruption from the strip passes) --------

def test_repair_drops_dropped_noun_fragments():
    text = ("**Lead Development**\n"
            "The summit collapsed after accusations (reuters.com). It accused NATO of letting "
            "down the by failing to back the campaign. China adopted the and on March 12 (ap.com).")
    out = dr._repair_broken_sentences(text)
    assert "the by failing" not in out          # article-with-deleted-noun fragment dropped
    assert "the and on March 12" not in out
    assert "The summit collapsed" in out         # the clean sentence survives
    assert "**Lead Development**" in out          # header preserved


def test_repair_drops_missing_number_and_dropped_title():
    text = ("Refineries may sell fuel with sulfur up to times the legal limit (x.com). "
            "The US of State confirmed the deal (y.com). The deal closed cleanly (z.com).")
    out = dr._repair_broken_sentences(text)
    assert "up to times" not in out              # missing number before "times"
    assert "US of State confirmed" not in out    # dropped "Secretary"
    assert "The deal closed cleanly" in out


def test_repair_leaves_clean_text_untouched():
    # No false positives on good prose — incl. legitimate "Department of State".
    text = ("**Lead Development**\nCisco acquired Splunk for $28 billion (reuters.com). The "
            "Department of State confirmed the deal closed in March (cnbc.com).")
    assert dr._repair_broken_sentences(text) == text


# --- #5: promo/shill + evergreen body filter + low-cred host blocks ----------------

def test_promo_filter_drops_memecoin_shill():
    body = ("Pepeto presale crossed $10 million ahead of rumored listing on Binance. "
            "Analysts call it the next 100x potential gem — don't miss out, join the "
            "presale and get in early before the token sale ends. " * 3)
    assert dr._is_promo_or_filler(body) is True
    assert dr._junk_body(body) is True


def test_filler_filter_drops_evergreen_contact_page():
    body = ("The FDA is responsible for protecting the public health by ensuring drug "
            "safety. For questions call 1-888-463-6332 (1-888-INFO-FDA). " * 4)
    assert dr._is_promo_or_filler(body) is True


def test_promo_filter_keeps_real_crypto_news():
    body = ("Aave deployed its V4 protocol on Ethereum mainnet today, introducing a unified "
            "liquidity layer and a new risk module. The DAO approved the upgrade after a "
            "three-week governance vote, citing improved capital efficiency. Total value "
            "locked across Aave markets stands near $18 billion per DefiLlama data. " * 3)
    assert dr._is_promo_or_filler(body) is False
    assert dr._junk_body(body) is False


def test_low_cred_hosts_blocked():
    assert dr._blocked("https://www.zerohedge.com/markets/some-story")
    assert dr._blocked("https://coinalertnews.com/pepeto-presale")
    assert not dr._blocked("https://www.reuters.com/markets/x")


def test_currency_mislabel_restamps_cny_as_usd():
    # #4 (currency class): the value traces to the source but in CNY, not USD → re-stamp.
    body = ("Alibaba Group reported results today. The company posted trailing twelve-month "
            "revenue of 1.185 trillion yuan for the period, up year over year on cloud and "
            "commerce strength. Management cited robust demand; analysts had modeled slightly "
            "lower. The Hong Kong listing rose as investors digested the report. " * 2)
    articles = [("Alibaba earnings", "http://x.com/1", body)]
    text = "**Lead**\nAlibaba revenue reached $1.185 trillion over the trailing year (x.com)."
    out = dr._correct_currency_mislabels(text, articles)
    assert "CNY 1.185 trillion" in out and "$1.185 trillion" not in out


def test_currency_mislabel_leaves_real_usd_untouched():
    body = ("The all-cash acquisition was valued at $1.185 trillion in USD terms, the parties "
            "confirmed in a joint statement. The deal, among the largest on record, is expected "
            "to close next year pending regulatory approval across several jurisdictions and a "
            "shareholder vote scheduled for the spring. Advisers were named on both sides. " * 2)
    articles = [("Deal", "http://x.com/1", body)]
    text = "**Lead**\nThe deal was worth $1.185 trillion (x.com)."
    assert dr._correct_currency_mislabels(text, articles) == text   # source confirms USD → unchanged


def test_currency_mislabel_skips_when_value_absent():
    # A FABRICATED figure (not in sources) is not a currency call — left for the
    # numeric-grounding / fresh-check layers, never mis-restamped.
    articles = [("Unrelated", "http://x.com/1",
                 "A story about weather patterns and regional sports with no such figure here. " * 8)]
    text = "**Lead**\nRevenue hit $1.185 trillion (x.com)."
    assert dr._correct_currency_mislabels(text, articles) == text


def test_currency_mislabel_ignores_already_qualified_dollar():
    # Regression (caught live 2026-07-01): a figure already prefixed (HK$/US$/A$) must
    # NOT be double-stamped into "HKHK$".
    body = ("Momenta priced its Hong Kong IPO to raise approximately HK$5.89 billion, or about "
            "$751 million, offering shares to cornerstone investors including GIC and BlackRock in "
            "a closely watched listing on the exchange this week amid strong institutional demand. " * 2)
    articles = [("Momenta IPO", "http://x.com/1", body)]
    text = "**Lead**\nMomenta raised HK$5.89 billion ($751 million) in its IPO (x.com)."
    out = dr._correct_currency_mislabels(text, articles)
    assert "HKHK$" not in out and out == text


def test_bottomline_drops_novel_currency_figure():
    # Audit: the bottom line invented "$67B" not in the body ($87.6B/$21B) → drop it.
    text = ("**Lead Development**\nThe OMB requested $87.6 billion, allocating $21 billion (ap.com).\n\n"
            "**Connections & bottom line** — The throughline is fiscal expansion (ap.com). "
            "Watch whether Congress approves the requested $67 billion for the program (ap.com).")
    out = dr._strip_novel_bottomline_figures(text)
    assert "$67 billion" not in out                       # invented in the conclusion → dropped
    assert "throughline is fiscal expansion" in out       # the clean sentence survives
    assert "$87.6 billion" in out and "$21 billion" in out


def test_bottomline_keeps_restated_figure():
    text = ("**Lead**\nLosses reached $708 billion under the scenario (fed.gov).\n\n"
            "**Connections & bottom line** — With $708 billion in modeled losses, watch how "
            "buffers hold (fed.gov).")
    assert dr._strip_novel_bottomline_figures(text) == text   # figure restated from above → kept


def test_bottomline_exempts_percentage_threshold():
    text = ("**Lead**\nInflation was 4.2% in May (bls.gov).\n\n**Connections & bottom line** — "
            "Watch whether inflation breaks 5% next quarter (bls.gov).")
    assert dr._strip_novel_bottomline_figures(text) == text   # bare % threshold exempt → kept


@pytest.mark.asyncio
async def test_jina_bypass_extracts_paywalled_article(monkeypatch):
    # Paywall bypass: Jina Reader returns clean article text; preamble + link URLs stripped.
    import httpx as _httpx

    class _Resp:
        status_code = 200
        text = ("Title: US stocks gain\nURL Source: https://ft.com/x\nPublished Time: 2026-06-30\n"
                "Markdown Content: US equities [rose sharply](https://ft.com/y) in 2026. "
                + "Markets rallied on strong earnings and easing rate expectations this quarter. " * 9)

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp()

    monkeypatch.setattr(_httpx, "AsyncClient", _Client)
    out = await dr._fetch_via_jina("https://www.ft.com/content/x")
    assert out and "US equities rose sharply" in out         # markdown link → its text
    assert "Markdown Content:" not in out and "https://ft.com/y" not in out


@pytest.mark.asyncio
async def test_jina_bypass_returns_none_on_block(monkeypatch):
    import httpx as _httpx

    class _Resp:
        status_code = 403
        text = "blocked"

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp()

    monkeypatch.setattr(_httpx, "AsyncClient", _Client)
    assert await dr._fetch_via_jina("https://www.ft.com/content/x") is None


def test_tier_label_reflects_source_quality():
    assert dr._tier_label("https://www.reuters.com/x") == "wire/primary"
    assert dr._tier_label("https://www.bbc.com/news/x") == "quality"
    assert dr._tier_label("https://randomblog.example/x") == "single/unverified"


def test_annotated_evidence_tags_reliability_and_corroboration():
    findings = [
        ("Oracle EBS RCE actively exploited", "https://www.reuters.com/1", "finding a"),
        ("Oracle EBS RCE under active attack", "https://apnews.com/2", "finding b"),   # same story, 2nd wire
        ("Obscure memecoin token launches", "https://randomblog.example/3", "finding c"),  # lone weak src
    ]
    out = dr._annotated_evidence(findings)
    assert "reuters.com · wire/primary" in out
    assert "2 sources" in out                              # Oracle story corroborated across wires
    assert "randomblog.example · single/unverified · 1 source" in out   # lone weak source flagged


def test_generic_feed_titles_rejected_real_stories_kept():
    # Vague-monitor fix: outlet-homepage "headlines" that describe the OUTLET, not an
    # event, must be rejected from the coverage-ranking pool; real stories must pass.
    g = dr._is_generic_feed_title
    assert g("Associated Press News publishes breaking headlines on June 30, 2026")
    assert g("BBC Home reports world news and business updates")
    assert g("BBC News latest updates June 30 2026")          # no-verb outlet-meta title
    assert g("Turkiye Today covers latest developments in Turkiye region")
    assert g("AP News - Breaking News, Latest Headlines, Photos & Videos")
    assert g("World News - The New York Times International")   # section-homepage, all-meta core
    assert g("World | Latest News & Updates")
    assert not g("Starmer resigns as UK Prime Minister amid Ukraine policy split")
    assert not g("Oracle E-Business Suite RCE CVE-2026-46817 under active exploitation")
    assert not g("Colombia elects Abelardo de la Espriella in presidential runoff")
    assert not g("Breaking: Venezuela earthquake death toll surpasses 1,700")   # colon-prefixed real story kept
    assert not g("In pictures: Pakistan-administered Kashmir returns to daily life")


@pytest.mark.asyncio
async def test_correction_pass_routes_to_synthesis_model(monkeypatch):
    # #3c: the correction REWRITE uses the synthesis model when set, so the 9B no
    # longer edits the 27B's output (and it avoids a mid-pipeline swap back to 9B).
    seen = {"model": "UNSET"}

    async def fake(msgs, **k):
        seen["model"] = k.get("model")
        return "**Lead**\nThe revised figure is modest (x.com)."

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake)
    text = "**Lead**\nRevenue hit $4,200,000,000 last quarter (x.com)."
    bodies = ["an unrelated source body with no such number " * 40]   # figure not traceable
    await dr._ground_numbers(text, bodies, model="qwen3.6:27b")
    assert seen["model"] == "qwen3.6:27b"


# --- _ensure_citations: deterministic backstop for the uncited-digest misfire -----

@pytest.mark.asyncio
async def test_ensure_citations_recites_uncited_digest(monkeypatch):
    text = (
        "## world — overview\n_read 3 sources_\n\n"
        "**Lead Development**\n"
        "The death toll from the disaster rose to over 1,400 people with tens of thousands more "
        "reported unaccounted for by their families across the affected region. Rescue teams from "
        "multiple nations deployed within a day to search collapsed structures and reach isolated "
        "zones, while the interim government pledged a major emergency relief fund and vowed to save "
        "as many trapped survivors as possible amid the extensive structural damage reported. "
        "International aid organizations warned that the window for rescuing survivors was narrowing "
        "rapidly as conditions on the ground deteriorated and communication lines remained down.\n"
    )
    findings = [("Quake toll rises", "http://apnews.com/1",
                 "The death toll rose to over 1,400 as rescue teams from many nations deployed.")]
    recited = ("**Lead Development**\nThe death toll rose to over 1,400 (apnews.com). Rescue "
               "teams from multiple nations deployed (apnews.com).")
    monkeypatch.setattr(dr.llm, "invoke_nothink", AsyncMock(return_value=recited))
    out, n = await dr._ensure_citations(text, findings)
    assert n >= 1 and "apnews.com" in out


@pytest.mark.asyncio
async def test_ensure_citations_noop_when_already_cited(monkeypatch):
    text = ("**Lead**\nBitcoin fell hard this week amid outflows (cnbc.com). Spot ETFs bled "
            "billions (coindesk.com). Equities also dropped on the news (theblock.co).\n")
    called = {"llm": False}

    async def _should_not_run(*a, **k):
        called["llm"] = True
        return "x"

    monkeypatch.setattr(dr.llm, "invoke_nothink", _should_not_run)
    out, n = await dr._ensure_citations(text, [("t", "http://cnbc.com/1", "body")])
    assert n == 0 and out == text and called["llm"] is False   # deterministic skip, no LLM call


@pytest.mark.asyncio
async def test_verify_lead_claims_no_caveat_when_supported(monkeypatch):
    text = "**Lead Development**\nRevenue reached $4,200,000,000 in the quarter (a.com).\n"
    from app.tools import native_search as _ns
    monkeypatch.setattr(_ns, "search", AsyncMock(return_value=[
        _Res("Revenue $4.2B", "The company posted $4.2 billion in revenue."),
        _Res("Recap", "A solid quarter overall."),
    ]))
    monkeypatch.setattr(dr.llm, "invoke_nothink", AsyncMock(return_value='{"verdict":"supported","note":""}'))
    monkeypatch.setattr(dr.llm, "extract_json_object", lambda r: {"verdict": "supported", "note": ""})
    out, n = await dr._verify_lead_claims(text, "x")
    assert n == 0 and "Fresh-check" not in out and out == text


# --- Stealth browser paywall routing (residential-IP free bypass) -------------

class _FR:
    """Minimal stand-in for a ToolResult."""
    def __init__(self, output="", success=False):
        self.output = output
        self.success = success


def test_paywall_host_split_invariant():
    # Hard (server-side, unreadable without auth) and metered (client-side, stealth-
    # readable) must be disjoint, and their union is the full paywall set the gather knows.
    assert dr._HARD_PAYWALL_HOSTS.isdisjoint(dr._METERED_HOSTS)
    assert dr._PAYWALL_HOSTS == dr._HARD_PAYWALL_HOSTS | dr._METERED_HOSTS
    assert {"ft.com", "bloomberg.com", "wsj.com"} <= dr._HARD_PAYWALL_HOSTS
    assert {"nytimes.com", "businessinsider.com"} <= dr._METERED_HOSTS


def test_browser_stealth_hardening_present():
    # The webdriver mask + challenge-wait are what let a residential-IP browser clear
    # Cloudflare; guard against a silent revert to the detectable default.
    import app.tools.browser as br
    assert "navigator, 'webdriver'" in br._STEALTH_INIT_JS
    assert "() => undefined" in br._STEALTH_INIT_JS
    assert any("security verification" in m for m in br._CF_CHALLENGE_MARKERS)
    assert hasattr(br.BrowserTool, "_await_challenge")
    assert hasattr(br.BrowserTool, "_safe_inner_text")


@pytest.mark.asyncio
async def test_hard_paywall_routes_to_bypass_not_browser(monkeypatch):
    # ft.com: http misses → straight to the reader bypass; the browser (which would
    # only render the "Subscribe to read" wall) is NEVER launched.
    import app.tools.http_fetch as hf
    import app.tools.browser as br

    class _Http:
        async def execute(self, **kw):
            return _FR(output="", success=False)

    class _Browser:
        launched = False
        async def execute(self, **kw):
            _Browser.launched = True
            return _FR(output="Subscribe to read", success=True)

    monkeypatch.setattr(hf, "HttpFetchTool", _Http)
    monkeypatch.setattr(br, "BrowserTool", _Browser)
    jina = {"called": False}

    async def _fake_jina(url):
        jina["called"] = True
        return "BYPASS ARTICLE BODY"

    monkeypatch.setattr(dr, "_fetch_via_jina", _fake_jina)
    out = await dr._fetch_body("https://www.ft.com/content/abc", browser_budget=[3])
    assert out == "BYPASS ARTICLE BODY"
    assert jina["called"] is True
    assert _Browser.launched is False       # hard paywall must not burn a render


@pytest.mark.asyncio
async def test_metered_paywall_escalates_to_stealth_browser(monkeypatch):
    # businessinsider.com: http misses → the STEALTH browser is tried (fresh cookieless
    # context resets the meter) and its article is returned; the bypass is not reached.
    import app.tools.http_fetch as hf
    import app.tools.browser as br

    good = ("Microsoft plans to cut thousands of jobs in the coming days as it restructures "
            "its cloud and devices divisions. The reductions follow a period of heavy "
            "investment in artificial intelligence infrastructure and data centers. "
            "Executives said the changes would help the company focus resources on its "
            "highest-growth products and services. Affected employees will be notified this "
            "week, according to people familiar with the matter. The move reflects broader "
            "cost pressures across the technology sector, where several large firms have "
            "announced similar reductions in recent months as they reprioritize spending. ")

    class _Http:
        async def execute(self, **kw):
            return _FR(output="", success=False)

    class _Browser:
        launched = False
        async def execute(self, **kw):
            _Browser.launched = True
            return _FR(output=good, success=True)

    monkeypatch.setattr(hf, "HttpFetchTool", _Http)
    monkeypatch.setattr(br, "BrowserTool", _Browser)
    jina = {"called": False}

    async def _fake_jina(url):
        jina["called"] = True
        return None

    monkeypatch.setattr(dr, "_fetch_via_jina", _fake_jina)
    out = await dr._fetch_body("https://www.businessinsider.com/microsoft-layoffs", browser_budget=[3])
    assert out and "Microsoft plans to cut thousands" in out
    assert _Browser.launched is True        # metered paywall IS handed to the browser
    assert jina["called"] is False          # browser succeeded → bypass not needed


# --- Subject-selection quality: promo / listicle / meta filters + coverage weight ---

def test_promo_pump_headlines_filtered():
    # Pump-and-dump / advertorial copy must be dropped so it can't become a subject.
    for t in [
        "Get in on the ground floor: An emerging giant off Wall Street's radar",
        "The Next Nvidia: This Chip Stock Could Explode 10x",
        "This Millionaire-Maker Stock Is a Screaming Buy",
        "Why You Need to Buy This Under-the-Radar AI Stock",
        "Bitcoin Price Prediction: BTC to Hit 200k",
    ]:
        assert dr._is_seo_headline(t) is True, t


def test_financial_listicles_filtered_but_count_news_survives():
    for t in ["8 Commodity ETFs for Diversification", "5 Dividend Stocks to Buy and Hold Forever",
              "3 Stocks to Buy Now Before They Soar"]:
        assert dr._is_seo_headline(t) is True, t
    # Real news that merely STARTS with a number (a count, not a list size) must survive.
    for t in ["2 China funds halt redemptions amid liquidity crunch",
              "3 killed as banks halt trading in emerging markets",
              "SpaceX to join the Nasdaq-100, driving huge ETF buying demand"]:
        assert dr._is_seo_headline(t) is False, t


def test_generic_meta_subjects_filtered():
    for s in ["Reuters publishes latest finance news headlines today July 01 2026",
              "ft.com automobiles section", "Bloomberg Commodities Markets update July 1 2026",
              "CNBC Futures and Commodities market movements", "FedEx shipment tracking services active"]:
        assert dr._is_generic_subject(s) is True, s
    # Concrete stories must survive.
    for s in ["Uber-backed Lime secures $167mn IPO for bike and scooter group",
              "US stocks record biggest quarterly gain in six years",
              "New UK prime minister faces unchanged bond market conditions"]:
        assert dr._is_generic_subject(s) is False, s


def test_strip_prompt_leak_removes_echoed_synthesis_instructions():
    # Seen live (Physics digest): the 27B echoed the section instructions verbatim after
    # the header instead of replacing them with content. Strip the instructions, keep the
    # header + the real content that follows.
    leaked = ("**Lead Development** — THE single most consequential physics development: a full "
              "paragraph (5-8 sentences) with EXACT numbers and the second-order implications.\n\n"
              "The 2026 New Horizons Prize went to four theorists (news.uchicago.edu).\n\n"
              "**Secondary Developments** — each OTHER genuinely consequential physics development "
              "as its own bullet. EXCLUDE sports results. Better five sharp on-mission bullets than "
              "ten padded with filler.\n\n"
              "*   **Riemann result:** a Nature manuscript establishes it (nature.com).")
    out = dr._strip_prompt_leak(leaked)
    assert "single most consequential" not in out and "padded with filler" not in out
    assert "New Horizons Prize" in out and "Riemann result" in out       # real content kept
    assert "**Lead Development**" in out and "**Secondary Developments**" in out  # headers kept
    # A clean digest (model followed instructions) is left byte-for-byte untouched.
    clean = "**Lead Development** — NVIDIA canceled its Rubin Ultra GPU (arstechnica.com)."
    assert dr._strip_prompt_leak(clean) == clean


def test_coverage_score_discounts_pr_wire_syndication():
    # A story on 3 independent quality desks must outrank the SAME release on 5 PR wires.
    real = {"toks": set(), "titles": ["a"], "hosts": {"reuters.com", "bloomberg.com", "cnbc.com"}}
    wire = {"toks": set(), "titles": ["b", "c", "d", "e", "f"],
            "hosts": {"prnewswire.com", "businesswire.com", "globenewswire.com",
                      "newswire.com", "accesswire.com"}}
    assert dr._coverage_score(real) > dr._coverage_score(wire)
    # even a SINGLE wire-service desk collapses to +1, so its 5 hosts don't beat 1 reuters
    solo = {"toks": set(), "titles": ["g"], "hosts": {"reuters.com"}}
    assert dr._coverage_score(solo) >= dr._coverage_score(wire)
