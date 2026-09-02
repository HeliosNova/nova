"""Open-questions ledger + belief revisions (2026-09-02, Phase 4.4)."""
from __future__ import annotations

import pytest

from app.core import questions as Q

BODY_V1 = (
    "## Current understanding\nRubin is shipping.\n"
    "## Open questions\n"
    "- What is Nvidia's current Rubin production capacity per quarter?\n"
    "- How much of TSMC's 2nm capacity is committed to Nvidia today?\n"
    "- Will Nvidia announce Rubin Ultra in 2027?\n"          # future-shaped: not a question
    "Watch for: Rubin revenue disclosure\n"
)
BODY_V2 = (
    "## Current understanding\nRubin is shipping at 40k units a quarter.\n"
    "REVISED: Rubin was believed to be sampling only → it is now in volume shipment → Micron HBM4 filings (micron.com)\n"
    "## Open questions\n"
    "- How much of TSMC's 2nm capacity is committed to Nvidia today?\n"
    "- Who is Nvidia's second packaging supplier for Rubin?\n"
)


@pytest.fixture
def dossier(db):
    db.execute("INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
               "VALUES ('domain', 'alpha-lab', 'Alpha Lab', ?, 'initial', 1)", (BODY_V1,))
    return db.fetchone("SELECT id FROM dossiers WHERE dkey = 'alpha-lab'")["id"]


def _statuses(db):
    return {r["question"]: r["status"] for r in db.fetchall("SELECT question, status FROM dossier_questions")}


def test_tables_exist_in_fresh_schema(db):
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(dossier_questions)")}
    assert {"dkey", "question", "qkey", "status", "curiosity_id", "resolution", "last_seen_at"} <= cols
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(belief_revisions)")}
    assert {"dkey", "revised", "created_at"} <= cols


def test_future_shaped_lines_are_not_questions():
    qs = Q.extract_questions(BODY_V1)
    assert qs == [
        "What is Nvidia's current Rubin production capacity per quarter?",
        "How much of TSMC's 2nm capacity is committed to Nvidia today?",
    ]


def test_question_key_normalizes_punctuation_and_case():
    assert Q.question_key("What is  Nvidia's capacity?") == Q.question_key("what is nvidia s capacity")


def test_sync_opens_keeps_and_retires_across_consolidations(db, dossier):
    stats = Q.sync_questions(db, dossier, "alpha-lab", BODY_V1)
    assert stats == {"new": 2, "kept": 0, "retired": 0}
    assert set(_statuses(db).values()) == {"open"}

    stats = Q.sync_questions(db, dossier, "alpha-lab", BODY_V2)
    assert stats == {"new": 1, "kept": 1, "retired": 1}
    st = _statuses(db)
    assert st["What is Nvidia's current Rubin production capacity per quarter?"] == "retired"
    assert st["How much of TSMC's 2nm capacity is committed to Nvidia today?"] == "open"
    assert st["Who is Nvidia's second packaging supplier for Rubin?"] == "open"

    # A retired question the body asks again reopens.
    Q.sync_questions(db, dossier, "alpha-lab", BODY_V1)
    assert _statuses(db)["What is Nvidia's current Rubin production capacity per quarter?"] == "open"
    assert Q.frontier(db, "alpha-lab")["retired"] == 1  # the packaging question dropped out


def test_queued_then_researched_lifecycle(db, dossier):
    Q.sync_questions(db, dossier, "alpha-lab", BODY_V1)
    q = "What is Nvidia's current Rubin production capacity per quarter?"
    assert Q.mark_queued(db, "alpha-lab", q, 77)
    assert not Q.mark_queued(db, "alpha-lab", q, None)
    assert _statuses(db)[q] == "queued"
    assert Q.mark_researched(db, 77, "About 40k units per quarter (micron.com)")
    assert not Q.mark_researched(db, 77, "again")          # idempotent
    row = db.fetchone("SELECT status, resolution, resolved_at FROM dossier_questions WHERE curiosity_id = 77")
    assert row["status"] == "researched" and "40k" in row["resolution"] and row["resolved_at"]
    fr = Q.frontier(db, "alpha-lab")
    assert fr == {"open": 1, "queued": 0, "researched": 1, "retired": 0, "total": 2}


def test_researched_survives_a_body_that_dropped_it(db, dossier):
    Q.sync_questions(db, dossier, "alpha-lab", BODY_V1)
    q = "What is Nvidia's current Rubin production capacity per quarter?"
    Q.mark_queued(db, "alpha-lab", q, 5)
    Q.sync_questions(db, dossier, "alpha-lab", BODY_V2)   # body no longer asks it → retired
    assert _statuses(db)[q] == "retired"
    assert Q.mark_researched(db, 5, "answered late")      # the answer still lands
    assert _statuses(db)[q] == "researched"


def test_revised_lines_are_recorded_once(db, dossier):
    assert Q.record_revisions(db, dossier, "alpha-lab", BODY_V2) == 1
    assert Q.record_revisions(db, dossier, "alpha-lab", BODY_V2) == 0
    rows = Q.list_revisions(db, dkey="alpha-lab")
    assert len(rows) == 1 and rows[0]["revised"].startswith("Rubin was believed")


def test_sync_after_consolidation_reads_the_stored_body(db, dossier):
    db.execute("UPDATE dossiers SET body = ? WHERE id = ?", (BODY_V2, dossier))
    stats = Q.sync_after_consolidation(db, "domain", "alpha-lab")
    assert stats == {"new": 2, "kept": 0, "retired": 0, "revisions": 1}
    assert Q.sync_after_consolidation(db, "domain", "nope") == {}


def test_list_questions_filters(db, dossier):
    Q.sync_questions(db, dossier, "alpha-lab", BODY_V1)
    assert len(Q.list_questions(db, status="open")) == 2
    assert Q.list_questions(db, status="researched") == []
    assert len(Q.list_questions(db, dkey="alpha-lab", limit=1)) == 1


def test_closed_storyline_questions_are_retired(db):
    for key, status in (("live-thread", "active"), ("dead-thread", "closed")):
        db.execute("INSERT INTO storylines (story_key, title, summary, monitors_csv, status, update_count, last_updated) "
                   "VALUES (?, ?, 's', 'm', ?, 3, datetime('now'))", (key, key, status))
        db.execute("INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
                   "VALUES ('storyline', ?, ?, ?, 'x', 1)", (key, key, BODY_V1))
        did = db.fetchone("SELECT id FROM dossiers WHERE dkey = ?", (key,))["id"]
        Q.sync_questions(db, did, key, BODY_V1)
    assert Q.retire_orphaned(db) == 2
    assert Q.retire_orphaned(db) == 0
    assert Q.frontier(db, "dead-thread")["retired"] == 2
    assert Q.frontier(db, "live-thread")["open"] == 2
    row = db.fetchone("SELECT resolution FROM dossier_questions WHERE dkey = 'dead-thread'")
    assert row["resolution"] == "storyline closed"


def test_api_routes_literal_before_parametrized():
    from app.api.dossiers import router
    paths = [r.path for r in router.routes]
    assert paths.index("/dossiers/questions") < paths.index("/dossiers/{dossier_id}")
    assert paths.index("/dossiers/beliefs") < paths.index("/dossiers/{dossier_id}")
    assert "/dossiers/{dossier_id}/questions" in paths


def test_pathway_registered():
    from app.monitors.pathways import get_pathway
    p = get_pathway("question_ledger")
    assert p and p.table == "dossier_questions" and p.monitor == "Knowledge Consolidation"
