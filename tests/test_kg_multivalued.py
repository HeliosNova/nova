"""Multi-valued predicates + revival bitemporal integrity (audit 2026-06-12).

Two bugs this pins:
1. add_fact mechanically superseded EVERY same-subject+predicate fact, so
   genuinely multi-valued knowledge (contains / has_property / member_of /
   co-authors) silently collapsed to a single object. Now only functional
   predicates supersede; multi-valued ones coexist.
2. Reviving a previously-superseded triple did not reset superseded_at, so the
   live fact stayed invisible to every bitemporal query (query_as_of()).
"""
from __future__ import annotations

import pytest

from app.core import kg as kg_module
from app.core.kg import KnowledgeGraph


def _patch_now(monkeypatch, ts: str) -> None:
    monkeypatch.setattr(kg_module, "_now_iso", lambda: ts)


def _objects(kg, subject, predicate=None):
    rows = kg.query(subject)
    objs = {r["object"].lower() for r in rows if predicate is None or r["predicate"] == predicate}
    return objs


# ---------------------------------------------------------------------------
# Multi-valued predicates coexist (no collapse)
# ---------------------------------------------------------------------------

class TestMultiValuedCoexist:
    @pytest.mark.asyncio
    async def test_has_property_keeps_all_values(self, db):
        kg = KnowledgeGraph(db)
        for trait in ("fast", "local", "private", "extensible"):
            await kg.add_fact("nova", "has_property", trait, source="user")
        objs = _objects(kg, "nova", "has_property")
        assert objs == {"fast", "local", "private", "extensible"}, objs
        assert kg.get_stats()["superseded_facts"] == 0

    @pytest.mark.asyncio
    async def test_contains_keeps_all_values(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("python", "contains", "lists")
        await kg.add_fact("python", "contains", "dicts")
        await kg.add_fact("python", "contains", "sets")
        assert _objects(kg, "python", "contains") == {"lists", "dicts", "sets"}

    @pytest.mark.asyncio
    async def test_member_of_and_coauthors_coexist(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("alice", "member_of", "team red")
        await kg.add_fact("alice", "member_of", "team blue")
        await kg.add_fact("paper", "written_by", "alice")
        await kg.add_fact("paper", "written_by", "bob")
        assert _objects(kg, "alice", "member_of") == {"team red", "team blue"}
        assert _objects(kg, "paper", "written_by") == {"alice", "bob"}

    @pytest.mark.asyncio
    async def test_related_to_degrade_target_does_not_collapse(self, db):
        # Non-canonical predicates degrade to related_to; those must coexist too,
        # or the degrade-don't-orphan design would merge unrelated facts.
        kg = KnowledgeGraph(db)
        await kg.add_fact("google", "acquired", "youtube")   # -> related_to
        await kg.add_fact("google", "acquired", "android")   # -> related_to
        assert _objects(kg, "google", "related_to") == {"youtube", "android"}

    @pytest.mark.asyncio
    async def test_born_in_multiscale_coexists(self, db):
        # Ulm AND Germany are both true at different granularities — superseding
        # one was the classic collapse.
        kg = KnowledgeGraph(db)
        await kg.add_fact("einstein", "born_in", "ulm")
        await kg.add_fact("einstein", "born_in", "germany")
        assert _objects(kg, "einstein", "born_in") == {"ulm", "germany"}


# ---------------------------------------------------------------------------
# Functional predicates still supersede
# ---------------------------------------------------------------------------

class TestFunctionalStillSupersedes:
    @pytest.mark.asyncio
    async def test_lives_in_supersedes(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("alice", "lives_in", "paris", source="user")
        await kg.add_fact("alice", "lives_in", "berlin", source="user")
        assert _objects(kg, "alice", "lives_in") == {"berlin"}
        assert kg.get_stats()["superseded_facts"] == 1

    @pytest.mark.asyncio
    async def test_price_of_supersedes(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("bitcoin", "price_of", "50000", source="user")
        await kg.add_fact("bitcoin", "price_of", "60000", source="user")
        assert _objects(kg, "bitcoin", "price_of") == {"60000"}

    @pytest.mark.asyncio
    async def test_inverse_functional_leads_supersedes_prior_holder(self, db):
        # One leader per org: a new subject for the same object retires the old.
        kg = KnowledgeGraph(db)
        await kg.add_fact("steve jobs", "leads", "apple", source="user")
        await kg.add_fact("tim cook", "leads", "apple", source="user")
        leaders = {r["subject"].lower() for r in kg.query("apple")
                   if r["predicate"] == "leads"}
        assert leaders == {"tim cook"}, leaders


# ---------------------------------------------------------------------------
# Revival resets superseded_at -> fact reappears in bitemporal queries
# ---------------------------------------------------------------------------

class TestRevivalBitemporalIntegrity:
    @pytest.mark.asyncio
    async def test_revived_fact_visible_in_query_as_of(self, db, monkeypatch):
        kg = KnowledgeGraph(db)
        # T1: alice lives in paris
        _patch_now(monkeypatch, "2026-01-01 00:00:00")
        await kg.add_fact("alice", "lives_in", "paris", source="user")
        # T2: moves to berlin -> paris superseded (valid_to + superseded_at set)
        _patch_now(monkeypatch, "2026-02-01 00:00:00")
        await kg.add_fact("alice", "lives_in", "berlin", source="user")
        # T3: moves back to paris -> the paris row is REVIVED (UNIQUE(s,p,o))
        _patch_now(monkeypatch, "2026-03-01 00:00:00")
        await kg.add_fact("alice", "lives_in", "paris", source="user")

        # The revived paris fact must be the current belief — not hidden by a
        # stale superseded_at. This is the exact regression.
        current = kg.query_as_of("alice")  # no-arg => superseded_at IS NULL filter
        objs = {r["object"].lower() for r in current}
        assert "paris" in objs, f"revived fact invisible to query_as_of(): {objs}"
        assert "berlin" not in objs  # berlin was superseded by the revival

        # And its row must have clean transaction-time fields.
        row = db.fetchone(
            "SELECT valid_to, superseded_at, superseded_by, created_at "
            "FROM kg_facts WHERE LOWER(subject)='alice' AND object='paris'"
        )
        assert row["valid_to"] is None
        assert row["superseded_at"] is None
        assert row["superseded_by"] is None
        assert row["created_at"] == "2026-03-01 00:00:00"  # re-recorded as of T3

    @pytest.mark.asyncio
    async def test_revived_fact_visible_in_recorded_at_belief(self, db, monkeypatch):
        kg = KnowledgeGraph(db)
        _patch_now(monkeypatch, "2026-01-01 00:00:00")
        await kg.add_fact("alice", "lives_in", "paris", source="user")
        _patch_now(monkeypatch, "2026-02-01 00:00:00")
        await kg.add_fact("alice", "lives_in", "berlin", source="user")
        _patch_now(monkeypatch, "2026-03-01 00:00:00")
        await kg.add_fact("alice", "lives_in", "paris", source="user")

        # "What did we believe on 2026-03-15?" must include the revived paris.
        belief = kg.query_as_of("alice", recorded_at="2026-03-15 00:00:00")
        assert {r["object"].lower() for r in belief} == {"paris"}
