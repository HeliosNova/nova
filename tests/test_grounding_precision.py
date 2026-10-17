"""Grounding precision (2026-10-17): the orphan-term passes stop excising
ordinary vocabulary. Every reprieve below is a term the live log showed the
contamination guard or the claim-grounding pass removing from a delivered
digest; every surviving orphan is a shape the passes exist to catch."""

from __future__ import annotations

import pytest

from app.monitors import deep_research as dr

CORPUS_FED = ("the fed held rates steady on wednesday as chair powell said inflation remains "
              "elevated. markets in the u.s. and europe rose after trump praised the decision. "
              "assets under management at the fund reached a record, with liquefied natural gas "
              "exports and anti-money laundering rules also in focus for the quarter. " * 3)


# --- initialisms ---------------------------------------------------------------
@pytest.mark.parametrize("acr,corpus", [
    ("AUM", "total assets under management rose"),
    ("KKIA", "flights into king khalid international airport were halted"),
    ("OFAC", "the office of foreign assets control sanctioned the bank"),
    ("LNG", "liquefied natural gas exports"),
    ("AML", "anti-money laundering rules"),
    ("PHEV", "plug-in hybrid electric vehicle sales"),
    ("THAAD", "the terminal high altitude area defense battery"),
    ("ROI", "a strong return on investment"),
])
def test_initialism_spelled_out_in_the_sources_is_not_an_orphan(acr, corpus):
    assert dr._initialism_in_corpus(acr, corpus)
    assert acr not in dr._orphan_terms(f"The {acr} figure rose sharply.", corpus * 20)


@pytest.mark.parametrize("acr,corpus", [
    ("IBIT", "flows into the bitcoin fund continued as traders bought"),
    ("TSMC", "the semiconductor market could tighten further this year"),   # 'the' is no initial
    ("PURL", "the company launched a new product line in march"),
])
def test_a_coincidental_word_run_does_not_spell_an_acronym(acr, corpus):
    assert not dr._initialism_in_corpus(acr, corpus)


def test_reverse_initials_hbm_and_ofac():
    assert dr._phrase_initials_in_corpus("High Bandwidth Memory", "demand for hbm chips")
    assert dr._phrase_initials_in_corpus("Foreign Assets Control", "ofac added the firm")
    assert not dr._phrase_initials_in_corpus("Muse Glimmer", "the mg ratio")   # two words: no


@pytest.mark.parametrize("tok", ["II", "III", "IV", "VII"])
def test_roman_numerals_are_never_orphans(tok):
    assert tok not in dr._orphan_terms(f"Phase {tok} trials began.", "trials began " * 100)


def test_fiscal_year_with_the_year_in_the_sources_is_not_an_orphan():
    corpus = "the fiscal 2027 budget request " * 40
    assert "FY2027" not in dr._orphan_terms("The FY2027 request grew.", corpus)
    assert "FY2031" in dr._orphan_terms("The FY2031 request grew.", corpus)


# --- titled names ----------------------------------------------------------------
@pytest.mark.parametrize("phrase,corpus", [
    ("President Donald Trump", "trump praised the decision"),
    ("Treasury Secretary Scott Bessent", "bessent told reporters"),
    ("Ukrainian President Volodymyr Zelenskyy", "zelenskyy arrived in kyiv"),
    ("Managing Director Kristalina Georgieva", "georgieva warned"),
    ("State Marco Rubio", "rubio met the delegation"),
])
def test_a_title_led_name_the_sources_carry_is_not_an_orphan(phrase, corpus):
    assert dr._titled_name_in_corpus(phrase, corpus)
    assert phrase not in dr._orphan_terms(f"{phrase} spoke on Monday.", corpus * 30)


@pytest.mark.parametrize("phrase", ["Muse Glimmer", "Imphal Valley", "Mirum Pharmaceuticals"])
def test_a_real_graft_still_fails(phrase):
    corpus = "the acquisition closed in april and the shares rose on the news " * 20
    assert phrase in dr._orphan_terms(f"The deal involved {phrase} directly.", corpus)


# --- common phrases and lead words ---------------------------------------------
@pytest.mark.parametrize("phrase", ["United States", "Federal Reserve", "Middle East", "Red Sea",
                                    "White House", "Global South", "Supreme Court"])
def test_common_phrases_are_never_orphans(phrase):
    corpus = "oil prices rose on tuesday after the attack " * 30
    assert phrase not in dr._orphan_terms(f"Across the {phrase}, prices rose.", corpus)


@pytest.mark.parametrize("text,glued", [
    ("The September jobs report showed gains.", "The September"),
    ("While Anthropic reported growth, rivals slowed.", "While Anthropic"),
    ("Between October 1 and 9, exports fell.", "Between October"),
    ("After Mr. Chen resigned, shares fell.", "After Mr"),
])
def test_a_lead_word_glued_to_one_capital_is_not_a_phrase(text, glued):
    corpus = "exports fell and shares fell on the report " * 30
    assert glued not in dr._orphan_terms(text, corpus)


def test_a_lead_word_is_stripped_and_the_rest_still_checked():
    corpus = "the quarter ended with growth " * 30
    out = dr._orphan_terms("While Anthropic Labs reported growth, rivals slowed.", corpus)
    assert out == ["Anthropic Labs"]      # the excision then removes the name, not "While"


# --- contamination: two other sources carrying a term make it vocabulary --------
@pytest.mark.asyncio
async def test_a_term_two_other_sources_carry_is_not_a_graft(monkeypatch):
    async def _no_fix(*_a, **_k):
        return ""
    monkeypatch.setattr(dr, "_invoke_bg", _no_fix)
    filler = "the market moved on the news and traders repositioned across sectors. " * 12
    arts = [
        ("A", "https://alpha.com/x", filler + "alpha reported record quarterly revenue."),
        ("B", "https://beta.com/y", filler + "quantedge global raised a new fund."),
        ("C", "https://gamma.com/z", filler + "quantedge global also expanded in singapore."),
    ]
    text = "Alpha's revenue beat estimates as Quantedge Global expanded (alpha.com)."
    out, n = await dr._check_contamination(text, arts)
    assert n == 0 and out == text


@pytest.mark.asyncio
async def test_a_term_only_one_other_source_carries_is_still_a_graft(monkeypatch):
    async def _no_fix(*_a, **_k):
        return ""
    monkeypatch.setattr(dr, "_invoke_bg", _no_fix)
    filler = "the market moved on the news and traders repositioned across sectors. " * 12
    arts = [
        ("A", "https://alpha.com/x", filler + "alpha reported record quarterly revenue."),
        ("B", "https://beta.com/y", filler + "quantedge global raised a new fund."),
        ("C", "https://gamma.com/z", filler + "a third outlet covered the earnings call."),
    ]
    text = "Alpha's revenue beat estimates as Quantedge Global expanded (alpha.com)."
    out, n = await dr._check_contamination(text, arts)
    assert n == 1 and "Quantedge Global" not in out
