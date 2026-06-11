"""Bitemporal KG tests — task #29, Memento-style as-of queries.

Covers:
- Migration adds superseded_at column + backfills from legacy valid_to rows
- Supersession path now sets superseded_at alongside valid_to
- query_as_of(entity, valid_at=, recorded_at=) returns the right snapshot
  under each timeline filter combination
"""
from __future__ import annotations

import pytest

from app.core import kg as kg_module
from app.core.kg import KnowledgeGraph


# ---------------------------------------------------------------------------
# Time helpers — patch _now_iso so we can control transaction timestamps
# ---------------------------------------------------------------------------

def _patch_now(monkeypatch, ts: str) -> None:
    """Pin kg._now_iso to return ts. Use to script transaction times in tests."""
    monkeypatch.setattr(kg_module, "_now_iso", lambda: ts)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

class TestMigration:
    def test_superseded_at_column_exists(self, db):
        KnowledgeGraph(db)
        cols = {r["name"] for r in db.fetchall("PRAGMA table_info(kg_facts)")}
        assert "superseded_at" in cols, "task #29 migration must add superseded_at"

    def test_backfill_from_valid_to(self, db):
        """Rows that have valid_to set (logically superseded before migration)
        must get superseded_at = valid_to on first KG init."""
        # Pre-populate a row that looks like it was superseded pre-migration:
        # KG hasn't been instantiated yet, so the table doesn't exist. Init once
        # to create it, then manually wipe the new column and re-init to trigger
        # the backfill path.
        KnowledgeGraph(db)
        db.execute(
            "INSERT INTO kg_facts (subject, predicate, object, confidence, source, "
            "valid_from, valid_to, superseded_at) "
            "VALUES ('alice', 'lives_in', 'paris', 0.9, 'user', "
            "'2026-01-01 00:00:00', '2026-02-01 00:00:00', NULL)"
        )
        # Re-init to trigger the backfill UPDATE
        KnowledgeGraph(db)
        row = db.fetchone("SELECT superseded_at FROM kg_facts WHERE subject='alice'")
        assert row["superseded_at"] == "2026-02-01 00:00:00"

    def test_index_created(self, db):
        KnowledgeGraph(db)
        # Index existence query
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_kg_superseded_at'"
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Supersession writes superseded_at
# ---------------------------------------------------------------------------

class TestSupersessionRecordsTransactionTime:
    @pytest.mark.asyncio
    async def test_superseded_at_set_when_conflict_resolved(self, db, monkeypatch):
        kg = KnowledgeGraph(db)

        # Insert at T1
        _patch_now(monkeypatch, "2026-03-01 10:00:00")
        await kg.add_fact("alice", "lives_in", "paris", source="user")

        # Conflict at T2 — same subject+predicate, different object
        _patch_now(monkeypatch, "2026-04-01 10:00:00")
        await kg.add_fact("alice", "lives_in", "berlin", source="user")

        rows = db.fetchall(
            "SELECT object, valid_to, superseded_at, superseded_by "
            "FROM kg_facts WHERE LOWER(subject)='alice' ORDER BY id"
        )
        # First row (paris) must be superseded with BOTH valid_to AND superseded_at set
        paris = next(r for r in rows if r["object"].lower() == "paris")
        assert paris["valid_to"] == "2026-04-01 10:00:00"
        assert paris["superseded_at"] == "2026-04-01 10:00:00"
        assert paris["superseded_by"] is not None
        # Second row (berlin) is the current fact — neither set
        berlin = next(r for r in rows if r["object"].lower() == "berlin")
        assert berlin["valid_to"] is None
        assert berlin["superseded_at"] is None


# ---------------------------------------------------------------------------
# query_as_of — bitemporal semantics
# ---------------------------------------------------------------------------

class TestQueryAsOf:
    @pytest.fixture
    async def history_kg(self, db, monkeypatch):
        """Three-version history of alice/lives_in:
          T1 = 2026-01-15 → paris (inserted T1, superseded T2)
          T2 = 2026-02-15 → berlin (inserted T2, superseded T3)
          T3 = 2026-03-15 → tokyo (inserted T3, current)
        """
        kg = KnowledgeGraph(db)

        _patch_now(monkeypatch, "2026-01-15 12:00:00")
        await kg.add_fact("alice", "lives_in", "paris", source="user")

        _patch_now(monkeypatch, "2026-02-15 12:00:00")
        await kg.add_fact("alice", "lives_in", "berlin", source="user")

        _patch_now(monkeypatch, "2026-03-15 12:00:00")
        await kg.add_fact("alice", "lives_in", "tokyo", source="user")
        return kg

    @pytest.mark.asyncio
    async def test_no_filters_returns_current_only(self, history_kg):
        rows = history_kg.query_as_of("alice")
        objs = [r["object"].lower() for r in rows]
        assert objs == ["tokyo"]

    @pytest.mark.asyncio
    async def test_recorded_at_before_first_insert_returns_nothing(self, history_kg):
        rows = history_kg.query_as_of("alice", recorded_at="2026-01-01 00:00:00")
        assert rows == []

    @pytest.mark.asyncio
    async def test_recorded_at_after_t1_before_t2_returns_paris(self, history_kg):
        rows = history_kg.query_as_of("alice", recorded_at="2026-02-01 00:00:00")
        objs = [r["object"].lower() for r in rows]
        assert objs == ["paris"]

    @pytest.mark.asyncio
    async def test_recorded_at_after_t2_before_t3_returns_berlin(self, history_kg):
        rows = history_kg.query_as_of("alice", recorded_at="2026-03-01 00:00:00")
        objs = [r["object"].lower() for r in rows]
        assert objs == ["berlin"]

    @pytest.mark.asyncio
    async def test_recorded_at_after_t3_returns_tokyo(self, history_kg):
        rows = history_kg.query_as_of("alice", recorded_at="2026-04-01 00:00:00")
        objs = [r["object"].lower() for r in rows]
        assert objs == ["tokyo"]

    @pytest.mark.asyncio
    async def test_recorded_at_exactly_at_supersession_excludes_old(self, history_kg):
        """Boundary: at T2 exactly, paris is just-superseded (superseded_at = T2).
        The strict-greater check (superseded_at > recorded_at) means paris is hidden
        at recorded_at == T2. The new fact berlin was created_at = T2 so it IS visible.
        """
        rows = history_kg.query_as_of("alice", recorded_at="2026-02-15 12:00:00")
        objs = [r["object"].lower() for r in rows]
        assert objs == ["berlin"]

    @pytest.mark.asyncio
    async def test_valid_at_alone_uses_world_time_only(self, history_kg):
        """valid_at without recorded_at: filter by world-validity only. Since
        all three facts share the same valid_from = created_at pattern in our
        usage, valid_at == T2 should yield berlin (paris ended T2 via valid_to)."""
        rows = history_kg.query_as_of("alice", valid_at="2026-02-20 00:00:00")
        # paris ended valid_to = T2 = 2026-02-15, so excluded at 02-20
        # berlin valid from 02-15, valid_to set when superseded by tokyo at T3 = 2026-03-15
        # tokyo valid from 03-15 — not yet valid at 02-20
        objs = [r["object"].lower() for r in rows]
        assert objs == ["berlin"]

    @pytest.mark.asyncio
    async def test_valid_at_and_recorded_at_combined(self, history_kg):
        """Bitemporal combined: 'what did we believe on 2026-04-01 about
        what was true on 2026-02-20?' We had all 3 records by 04-01; only
        berlin was world-valid on 02-20. So both filters together → berlin.
        """
        rows = history_kg.query_as_of(
            "alice",
            valid_at="2026-02-20 00:00:00",
            recorded_at="2026-04-01 00:00:00",
        )
        objs = [r["object"].lower() for r in rows]
        assert objs == ["berlin"]

    @pytest.mark.asyncio
    async def test_valid_at_late_with_early_recorded_at_returns_nothing(self, history_kg):
        """'What did we believe on 2026-02-01 about what was true on 2026-04-01?'
        On 02-01 we only knew paris. paris's valid window ended 02-15. So our
        2026-02-01-belief-set says nothing about 2026-04-01.
        """
        rows = history_kg.query_as_of(
            "alice",
            valid_at="2026-04-01 00:00:00",
            recorded_at="2026-02-01 00:00:00",
        )
        assert rows == []

    def test_unknown_entity_returns_empty(self, db):
        kg = KnowledgeGraph(db)
        assert kg.query_as_of("nobody") == []

    def test_normalizes_entity_input(self, db):
        kg = KnowledgeGraph(db)
        # Empty/whitespace entity must short-circuit, not error
        assert kg.query_as_of("") == []
        assert kg.query_as_of("   ") == []
