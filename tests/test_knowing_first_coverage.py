"""A dossier that covers the question answers it first, tools second (2026-09-09).

The knowing-first gate was written narrow on purpose (2026-09-01): a
knowledge-interrogation phrasing, a dossier title fully named in the query, or
a lesson whose topic the query names. The nightly `knowing_dossier_paraphrase`
eval showed the gap twice on 2026-09-09: the seeded "Fusion Energy Pilots"
dossier was retrieved and injected for "Where do things stand with Iceland's
experimental fusion project?" (the retriever scored it: a title token plus two
body overlaps), and the 9B still ran five web rounds to the circuit breaker and
answered with tool narration. The dossier covered the question; the gate could
not see that, because it only ever saw the rendered prompt text.

So the retriever now reports its overlap score, the think context keeps the
retrieved dossiers, and the gate treats a dossier that matched on a title token
AND at least two more tokens as covering the query. Tool-implying queries stay
tool-first, and an empty tool-less answer still falls back to tools.
"""
from __future__ import annotations

import pytest

import app.core.brain as brain
from app.core import dossiers as d

QUERY = "Where do things stand with Iceland's experimental fusion project?"
FUSION_BODY = (
    "## Current understanding\n"
    "The Aurora-7 fusion pilot plant in Iceland is the field's furthest-along compact "
    "tokamak: in July 2026 it sustained Q=2.1 for 43 seconds (fusionreview.com).\n"
    "## Key facts & figures\n- Fusion gain: Q=2.1 sustained for 43 seconds (fusionreview.com)\n"
    "## Open questions\n- Whether Q>3 is reachable without a divertor redesign\n"
)
RENDERED = ("Standing knowledge dossiers (Nova's accumulated understanding). Answer from them "
            "when they cover the question:\n### Fusion Energy Pilots\nThe Aurora-7 fusion pilot "
            "plant in Iceland is the field's furthest-along compact tokamak.")


def test_the_retriever_reports_how_well_a_dossier_matched(db):
    db.execute("INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
               "VALUES ('domain', 'fusion-energy-pilots', 'Fusion Energy Pilots', ?, 'seed', 1)",
               (FUSION_BODY,))
    hits = d.get_relevant_dossiers(db, QUERY)
    assert hits and hits[0]["title"] == "Fusion Energy Pilots"
    assert hits[0]["score"] >= 4, hits[0]


def test_a_covering_dossier_makes_the_first_generation_tool_less():
    covering = [{"title": "Fusion Energy Pilots", "excerpt": "…", "score": 4}]
    assert brain._knowing_answers_query(QUERY, RENDERED, "", dossiers=covering)


def test_a_weak_overlap_keeps_the_narrow_gate():
    weak = [{"title": "Energy and Climate", "excerpt": "…", "score": 2}]
    assert not brain._knowing_answers_query(QUERY, RENDERED, "", dossiers=weak)


def test_tool_verbs_still_win_over_coverage():
    covering = [{"title": "Fusion Energy Pilots", "excerpt": "…", "score": 9}]
    assert not brain._knowing_answers_query(
        "Search the web for the latest on Iceland's fusion project", RENDERED, "", dossiers=covering)


def test_the_old_rules_are_unchanged_without_dossier_objects():
    assert brain._knowing_answers_query("what's the latest on Microsoft?",
                                        "Standing knowledge dossiers:\n### Microsoft\nCopilot changes.", "")
    assert not brain._knowing_answers_query(QUERY, RENDERED, "")
