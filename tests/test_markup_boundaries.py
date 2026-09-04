"""Markdown must not leak into fields that get searched, spoken or re-prompted.

Digest prose is markdown and every field extracted from it inherits the
emphasis marks. Measured 2026-09-04 on the live store: 1,327 of 1,959 storyline
events, 20 of 259 open questions and 4 open forecast claims carried `**bold**`
or backticks. Those are not cosmetic — a storyline event is re-read by the next
digest, by dossier consolidation and by entity timelines; a curiosity topic and
a forecast claim are web-searched verbatim.

The morning's fix sanitised curiosity topics only, which treated the symptom
where it was noticed instead of the class. One helper now guards every
boundary. Digest BODIES are deliberately excluded: those are meant to be
markdown.
"""
from __future__ import annotations

import pytest

from app.core.text_utils import strip_markup

DIRTY = "**Nvidia** ships `Rubin` to __three__ Azure regions"
CLEAN = "Nvidia ships Rubin to three Azure regions"


@pytest.mark.parametrize("raw,expected", [
    (DIRTY, CLEAN),
    ("## Heading then text", "Heading then text"),
    ("- bullet item", "bullet item"),
    ("1. numbered item", "numbered item"),
    ("> quoted line", "quoted line"),
    ("[Reuters](https://reuters.com) reported it", "Reuters reported it"),
    ("spaces    collapse\n\nacross lines", "spaces collapse across lines"),
    ("plain text survives", "plain text survives"),
    ("", ""),
])
def test_strip_markup(raw, expected):
    assert strip_markup(raw) == expected


def test_figures_and_citations_survive():
    out = strip_markup("Bitcoin tests the **$77,000** support level (coindesk.com)")
    assert "$77,000" in out and "(coindesk.com)" in out and "*" not in out


def test_storyline_events_are_clean_at_the_write_boundary():
    from app.core.storylines import _event_summary
    out = _event_summary(DIRTY)
    assert "**" not in out and "`" not in out
    assert out.startswith("Nvidia ships Rubin")


def test_open_questions_are_clean_before_they_become_search_queries(db):
    from app.core.questions import extract_questions
    body = ("## Current understanding\nx\n## Open questions\n"
            "- How much of the **$7.5 billion** package is allocated to steel?\n")
    qs = extract_questions(body)
    assert qs and "**" not in qs[0]
    assert "$7.5 billion" in qs[0]


def test_forecast_claims_are_clean(db):
    from app.core.forecasts import create_forecast
    fid = create_forecast(db, "Bitcoin tests the **$77,000** support level by 2026-12-31",
                          days=60, confidence=0.6)
    claim = db.fetchone("SELECT claim FROM forecasts WHERE id = ?", (fid,))["claim"]
    assert "**" not in claim and "$77,000" in claim


def test_curiosity_topics_use_the_same_helper(db):
    from app.core.curiosity import CuriosityQueue
    q = CuriosityQueue(db)
    qid = q.add("What is the current status of the **$2.48 billion** facility?",
                source="dossier_open_question")
    topic = db.fetchone("SELECT topic FROM curiosity_queue WHERE id = ?", (qid,))["topic"]
    assert "**" not in topic and "$2.48 billion" in topic


def test_digest_bodies_are_not_sanitised():
    """The briefing itself is markdown; only extracted fields are stripped."""
    import inspect

    from app.monitors import deep_research as dr
    assert "strip_markup" not in inspect.getsource(dr._synthesize_from_evidence)
