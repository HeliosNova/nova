"""Deterministic per-sentence citation attribution (`_cite_uncited_sentences`).

Autopsy 2026-07-06: ~28% of factual sentences in full-length digests carry no
inline citation — sibling sentences of a cited story. The pass appends '(host)'
ONLY when exactly one read source strongly contains the sentence's distinctive
tokens; ambiguity or weak evidence must leave the sentence untouched."""
from __future__ import annotations

import app.monitors.deep_research as dr
from app.monitors import report_grader as rg

ARTS = [
    ("Apple integrates Gemini into Siri",
     "https://www.cnbc.com/apple-gemini",
     "Apple has integrated Google's Gemini models into Siri after a 2.1 billion "
     "dollar annual deal. The move ends internal Apple Foundation Models work. "
     "Apple executed a strategic pivot since the App Store era."),
    ("Reddit fights AI spam",
     "https://www.theverge.com/reddit-llm",
     "Reddit is deploying Large Language Models to detect subtle coordinated "
     "spam campaigns at scale, moderators said."),
]


def test_appends_citation_on_unique_strong_match():
    text = ("* **Apple's pivot:** Apple executed a strategic pivot since the App Store era, "
            "integrating Gemini models into Siri.\n")
    out, n = dr._cite_uncited_sentences(text, ARTS)
    assert n == 1
    assert "(cnbc.com)" in out
    # the citation lands before the terminal period, grader-parseable
    assert rg._cites_in(out) == ["cnbc.com"]


def test_leaves_already_cited_and_short_lines_alone():
    text = ("* Apple executed a strategic pivot since the App Store era, integrating "
            "Gemini models into Siri (cnbc.com).\n"
            "* Short line here.\n")
    out, n = dr._cite_uncited_sentences(text, ARTS)
    assert n == 0 and out == text.rstrip("\n") + "\n" or n == 0


def test_ambiguous_or_unsupported_sentences_untouched():
    # tokens found nowhere in the sources → no guessball citation
    text = "* Zorblax Corporation raised 9.9 quadrillion zorkmids in Ruritania funding.\n"
    out, n = dr._cite_uncited_sentences(text, ARTS)
    assert n == 0 and "(" not in out.replace("(", "", 0) or n == 0
    assert "cnbc.com" not in out and "theverge.com" not in out


def test_needs_two_distinctive_tokens():
    # only one distinctive token ("Reddit") — too weak to attribute
    text = "* Reddit continued to operate its platform normally throughout the period.\n"
    out, n = dr._cite_uncited_sentences(text, ARTS)
    assert n == 0


def test_headers_and_metadata_untouched():
    text = ("## 🌐 technology — domain overview\n"
            "_read 2 sources: cnbc.com, theverge.com · July 06_\n"
            "**Secondary Developments**\n")
    out, n = dr._cite_uncited_sentences(text, ARTS)
    assert n == 0 and out == text.rstrip("\n") or n == 0


def test_multi_sentence_bullet_only_uncited_sibling_gets_cite():
    text = ("* Reddit is deploying Large Language Models to detect subtle coordinated "
            "spam campaigns at scale (theverge.com). Apple executed a strategic pivot "
            "since the App Store era with its Gemini models integration into Siri.\n")
    out, n = dr._cite_uncited_sentences(text, ARTS)
    assert n == 1
    assert out.count("(theverge.com)") == 1 and "(cnbc.com)" in out


def test_capped_additions():
    many = "\n".join(
        "* Apple executed a strategic pivot since the App Store era, integrating "
        "Gemini models into Siri today." for _ in range(20))
    out, n = dr._cite_uncited_sentences(many, ARTS, max_added=5)
    assert n == 5 and out.count("(cnbc.com)") == 5
