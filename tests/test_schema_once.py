"""Per-DB schema-ensure memo (audit 2026-08-23).

Every store re-ran its idempotent __init__ DDL (CREATE TABLE IF NOT EXISTS +
ALTER-catch migrations) on EVERY construction. Stores are constructed
repeatedly in async contexts (live: kg.py DDL x8 in 2h on the event-loop
thread), so write-lock DDL kept landing on the loop — the lock-convoy bug
class. SafeDB.schema_ensured/mark_schema_ensured memoize per (SafeDB instance,
tag): first construction runs DDL, later ones skip it entirely (no write-lock
acquire). Fresh SafeDB instances (tests, new processes) still run DDL.
"""

from app.database import SafeDB


def test_memo_starts_false_and_sticks(tmp_path):
    db = SafeDB(str(tmp_path / "a.db"))
    assert not db.schema_ensured("kg")
    db.mark_schema_ensured("kg")
    assert db.schema_ensured("kg")
    assert not db.schema_ensured("reflexions")  # per-tag
    db.close()


def test_memo_is_per_instance(tmp_path):
    db1 = SafeDB(str(tmp_path / "a.db"))
    db2 = SafeDB(str(tmp_path / "b.db"))
    db1.mark_schema_ensured("kg")
    assert not db2.schema_ensured("kg")
    db1.close()
    db2.close()


def test_second_store_construction_runs_no_ddl(tmp_path):
    from app.core.kg import KnowledgeGraph

    db = SafeDB(str(tmp_path / "kg.db"))
    db.init_schema()
    KnowledgeGraph(db)  # first construction: runs DDL

    executed = []
    real_execute = db.execute

    def counting_execute(sql, params=()):
        executed.append(sql)
        return real_execute(sql, params)

    db.execute = counting_execute
    KnowledgeGraph(db)  # second construction: memo must skip ALL DDL
    ddl = [s for s in executed if s.strip().upper().startswith(("CREATE", "ALTER"))]
    assert ddl == [], f"second construction still ran DDL: {ddl[:3]}"
    db.close()


def test_fresh_db_instance_still_gets_schema(tmp_path):
    from app.core.kg import KnowledgeGraph

    db = SafeDB(str(tmp_path / "fresh.db"))
    db.init_schema()
    KnowledgeGraph(db)
    row = db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_facts'"
    )
    assert row is not None, "first construction must still create the schema"
    db.close()


def test_startup_grace_gates_loop_warning(tmp_path, caplog):
    """The sync-DB-on-loop tripwire is silent during startup grace (one-shot
    init calls before traffic can't block anything; 18 warnings/boot bred
    alarm fatigue) and loud after end_startup_grace — so any steady-state
    warning is a genuine offender."""
    import asyncio
    import logging

    db = SafeDB(str(tmp_path / "grace.db"))
    db.init_schema()
    prior = SafeDB._startup_grace
    try:
        async def probe():
            db.fetchone("SELECT 1")

        SafeDB._startup_grace = True
        SafeDB._loop_thread_warned.discard("SELECT 1")
        with caplog.at_level(logging.WARNING, logger="app.database"):
            asyncio.run(probe())
        assert "Sync DB call on event-loop thread" not in caplog.text

        SafeDB.end_startup_grace()
        SafeDB._loop_thread_warned.discard("SELECT 1")
        with caplog.at_level(logging.WARNING, logger="app.database"):
            asyncio.run(probe())
        assert "Sync DB call on event-loop thread" in caplog.text
    finally:
        SafeDB._startup_grace = prior
        SafeDB._loop_thread_warned.discard("SELECT 1")
        db.close()
