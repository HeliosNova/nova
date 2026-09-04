"""Entity timelines — the dated spine (2026-09-03, Phase 4.4).

The KG has been bitemporal since 2026-05-16: every fact carries when it was
recorded and, once replaced, when it was superseded and by what. Nothing read
that trail. Meanwhile the entity-dossier prompt demanded "5-10 dated bullets of
the major shifts, oldest→newest" from `_entity_sources`, which supplied undated
live facts with every supersession filtered out — the model had to invent the
history it was graded on.

Timelines are deliberately kept OUT of digest synthesis: a paired A/B measured
prior context there as a cost to fact grounding.
"""
from __future__ import annotations

import inspect

from app.core.timelines import (
    BELIEF,
    LEARNED,
    REVISED,
    STORYLINE,
    entity_timeline,
    format_timeline,
    timeline_block,
)


import pytest


@pytest.fixture(autouse=True)
def _kg_columns(db):
    """superseded_at / trust / quarantined are added lazily when the KG is first
    constructed, not by init_schema — production builds one at startup."""
    from app.core.kg import KnowledgeGraph
    KnowledgeGraph(db)


def _fact(db, subject, predicate, obj, *, days_ago, superseded_days_ago=None, by=None):
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, source, created_at, "
        "superseded_at, superseded_by) VALUES (?, ?, ?, 0.9, 'extracted', datetime('now', ?), "
        + ("datetime('now', ?)" if superseded_days_ago is not None else "NULL") + ", ?)",
        ((subject, predicate, obj, f"-{days_ago} days", f"-{superseded_days_ago} days", by)
         if superseded_days_ago is not None
         else (subject, predicate, obj, f"-{days_ago} days", by)),
    )
    return db.fetchone("SELECT MAX(id) AS id FROM kg_facts")["id"]


def test_a_superseded_fact_becomes_a_dated_change(db):
    new_id = _fact(db, "nvidia", "leads", "the accelerator market", days_ago=3)
    _fact(db, "nvidia", "leads", "the training market", days_ago=40,
          superseded_days_ago=3, by=new_id)

    events = entity_timeline(db, "nvidia")
    kinds = {e["kind"] for e in events}
    assert LEARNED in kinds and REVISED in kinds
    revised = next(e for e in events if e["kind"] == REVISED)
    assert "the training market → the accelerator market" in revised["text"]


def test_a_retired_fact_with_no_replacement_still_reads(db):
    _fact(db, "acme", "leads", "the widget market", days_ago=20, superseded_days_ago=2)
    revised = next(e for e in entity_timeline(db, "acme") if e["kind"] == REVISED)
    assert "no longer holds" in revised["text"]


def test_events_are_oldest_first_and_dated(db):
    _fact(db, "tsmc", "located_in", "hsinchu", days_ago=30)
    _fact(db, "tsmc", "produces", "2nm wafers", days_ago=2)
    events = entity_timeline(db, "tsmc")
    assert [e["when"] for e in events] == sorted(e["when"] for e in events)
    assert all(len(e["when"]) == 10 and e["when"][4] == "-" for e in events)


def test_storyline_events_and_belief_revisions_join_the_timeline(db):
    db.execute("INSERT INTO storylines (story_key, title, summary, monitors_csv, status, "
               "update_count, last_updated) VALUES ('rubin', 'Rubin rollout', 's', 'm', "
               "'active', 1, datetime('now'))")
    sid = db.fetchone("SELECT id FROM storylines")["id"]
    db.execute("INSERT INTO storyline_events (storyline_id, summary, source_monitor, is_new, "
               "created_at) VALUES (?, 'Nvidia ships Rubin to Azure', 'm', 1, datetime('now','-1 day'))",
               (sid,))
    db.execute("INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
               "VALUES ('entity', 'nvidia', 'Nvidia', 'b', 'c', 1)")
    did = db.fetchone("SELECT id FROM dossiers")["id"]
    db.execute("INSERT INTO belief_revisions (dossier_id, dkey, revised, created_at) "
               "VALUES (?, 'nvidia', 'Nvidia was believed to be sampling only → now shipping', "
               "datetime('now','-2 days'))", (did,))

    kinds = {e["kind"] for e in entity_timeline(db, "nvidia")}
    assert STORYLINE in kinds and BELIEF in kinds


def test_the_window_excludes_ancient_history(db):
    _fact(db, "oldcorp", "located_in", "nowhere", days_ago=400)
    assert entity_timeline(db, "oldcorp", days=180) == []
    assert entity_timeline(db, "oldcorp", days=500)


def test_unknown_subject_and_blank_are_empty(db):
    assert entity_timeline(db, "nobody at all") == []
    assert entity_timeline(db, "") == []
    assert timeline_block(db, "") == ""


def test_rendering_marks_changes_and_respects_the_cap():
    events = [
        {"when": "2026-08-01", "kind": LEARNED, "text": "x leads y"},
        {"when": "2026-09-01", "kind": REVISED, "text": "x leads: y → z"},
    ]
    out = format_timeline(events)
    assert out.index("2026-08-01") < out.index("2026-09-01"), "oldest first"
    assert "changed: x leads: y → z" in out
    assert "changed: x leads y" not in out
    assert format_timeline([]) == ""

    big = [{"when": "2026-09-0%d" % (i % 9 + 1), "kind": LEARNED, "text": "y" * 200}
           for i in range(50)]
    assert len(format_timeline(big, cap=500)) <= 500


def test_entity_dossiers_are_fed_the_timeline(db):
    """The consolidation prompt demands dated bullets; the material must carry dates."""
    from app.core.dossiers import _entity_sources
    _fact(db, "nvidia", "leads", "accelerators", days_ago=5)
    src = _entity_sources(db, "nvidia", None)
    assert "TIMELINE" in src and "2026-" in src

    prompt_src = inspect.getsource(_entity_sources)
    assert "timeline_block" in prompt_src


def test_digest_synthesis_is_deliberately_left_alone():
    """The A/B verdict must not be quietly undone by a timeline injection."""
    from app.monitors import deep_research as dr
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "timeline" not in src.lower()


def test_the_api_route_exists_and_is_read_only():
    from app.api.system import router
    routes = {r.path: tuple(sorted(r.methods or [])) for r in router.routes}
    assert routes.get("/kg/timeline") == ("GET",)
