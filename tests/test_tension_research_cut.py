"""A source that never once succeeded, and the slot that fed it (2026-09-06).

Curiosity resolves about one question a day against ~50 pending. Broken down
by where the question came from, over the queue's whole lifetime:

    source                 resolved  failed  pending  dismissed   rate
    dossier_open_question        21       7       39         32    75%
    agent_failure                 5       0        0          0   100%
    dossier_tension               0       6        5         15     0%

Not a low rate. ZERO, across 26 items. A tension asks which of Nova's OWN two
stored digests is right — "Current Events says 209 billion but Latin America
says 65 billion" — and the research pass behind this queue searches the web, so
it structurally cannot answer. The judge said exactly that, eight times in one
log: "the response fails to resolve the contradiction".

At MAX_CURIOSITY_ATTEMPTS=3, the six failures alone consumed eighteen research
passes, and the five pending would have taken fifteen more.

The 2026-09-01 reading was that these were STARVED behind 0.6-0.7 open
questions, and they were given a reserved every-third pick in `get_next` to fix
it. That was a hypothesis; five days and a third of the loop's throughput later
the count was still zero. It is now tested and refuted, so both the mint and
the slot are gone.

What SURVIVES is the detection. `_numeric_tensions` still runs and the ⚡
Tension line is still written into the dossier body — its docstring always said
"surfaces the tension for investigation; never auto-resolves", which was the
right contract. Answering one needs to read Nova's own records, a different
capability from web research that does not exist yet.
"""
from __future__ import annotations

import os
import tempfile

from app.database import SafeDB


def _db(path):
    d = SafeDB(path)
    d.init_schema()
    return d


def _queue(db):
    db.execute(
        "CREATE TABLE IF NOT EXISTS curiosity_queue ("
        "id INTEGER PRIMARY KEY, topic TEXT, source TEXT, urgency REAL, "
        "status TEXT, attempts INT DEFAULT 0, resolution TEXT, "
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP, resolved_at TEXT)")


def test_migration_37_retires_only_the_pending_tensions(tmp_path):
    p = str(tmp_path / "m.db")
    db = _db(p)
    _queue(db)
    for topic, src, st in (
        ("Resolve contradiction: A says 1 but B says 2", "dossier_tension", "pending"),
        ("Resolve contradiction: already done", "dossier_tension", "failed"),
        ("Resolve contradiction: resolved somehow", "dossier_tension", "resolved"),
        ("India: what reforms are proposed", "dossier_open_question", "pending"),
    ):
        db.execute("INSERT INTO curiosity_queue (topic, source, status) VALUES (?,?,?)",
                   (topic, src, st))
    db.execute("DELETE FROM schema_version WHERE version = 37")
    db.close()

    db2 = _db(p)
    got = {(r["source"], r["status"]) for r in
           db2.fetchall("SELECT source, status FROM curiosity_queue")}
    db2.close()
    assert ("dossier_tension", "pending") not in got, "the pending tension must be retired"
    assert ("dossier_tension", "failed") in got, "history is a record, not a target"
    assert ("dossier_tension", "resolved") in got
    assert ("dossier_open_question", "pending") in got, "the 75% source is untouched"


def test_the_retired_row_says_why(tmp_path):
    """A dismissed row with no reason is indistinguishable from queue churn."""
    p = str(tmp_path / "m2.db")
    db = _db(p)
    _queue(db)
    db.execute("INSERT INTO curiosity_queue (topic, source, status) VALUES (?,?,?)",
               ("Resolve contradiction: x", "dossier_tension", "pending"))
    db.execute("DELETE FROM schema_version WHERE version = 37")
    db.close()
    db2 = _db(p)
    res = db2.fetchone("SELECT resolution FROM curiosity_queue")["resolution"]
    db2.close()
    assert "web pass cannot settle" in res
    assert "own digests" in res


def test_a_fresh_install_does_not_crash_on_the_missing_table():
    """curiosity_queue is created LAZILY, so it is absent on a new install and
    an unguarded UPDATE raises OperationalError straight out of startup. This
    only shows up when the migration is run against an EMPTY database."""
    p = os.path.join(tempfile.mkdtemp(), "fresh.db")
    db = _db(p)                      # would raise before the sqlite_master guard
    assert db.fetchone("SELECT 1 FROM schema_version WHERE version = 37")
    db.close()


def test_the_migration_is_idempotent(tmp_path):
    p = str(tmp_path / "m3.db")
    db = _db(p)
    _queue(db)
    db.execute("INSERT INTO curiosity_queue (topic, source, status) VALUES (?,?,?)",
               ("Resolve contradiction: x", "dossier_tension", "pending"))
    db.execute("DELETE FROM schema_version WHERE version = 37")
    db.close()
    for _ in range(3):
        d = _db(p)
        d.close()
    d = _db(p)
    n = d.fetchone("SELECT COUNT(*) c FROM curiosity_queue")["c"]
    d.close()
    assert n == 1, "re-running must not duplicate or delete rows"
