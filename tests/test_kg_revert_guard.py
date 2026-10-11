"""G3 (plan 2026-10-09): a retired value re-read from an OLDER source must not
reactivate (arXiv 2610.07715). Measured 2026-10-09: 37 A-B-A chains in the live
KG, every one a revival whose evidence was no newer than the supersession.

With G2 the revival request carries the evidence's valid_from; the guard
compares it with the live superseding row's valid_from.
"""

from __future__ import annotations

import logging

import pytest

from app.core.kg import KnowledgeGraph


@pytest.fixture
def kg(db):
    return KnowledgeGraph(db)


async def _live(kg, s, p):
    rows = kg._db.fetchall(
        "SELECT object, valid_from, valid_to FROM kg_facts WHERE LOWER(subject)=LOWER(?) AND predicate=? ORDER BY id",
        (s, p))
    return [(r["object"], r["valid_to"] is None) for r in rows]


@pytest.mark.asyncio
async def test_revert_with_older_evidence_is_refused(kg, caplog):
    assert await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-01-10", source="researched", provenance="deep_research:a")
    assert await kg.add_fact("Acme", "leads", "Bob", valid_from="2026-06-01", source="researched", provenance="deep_research:b")
    assert await _live(kg, "Acme", "leads") == [("Alice", False), ("Bob", True)]
    with caplog.at_level(logging.INFO, logger="app.core.kg"):
        ok = await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-03-01", source="researched", provenance="deep_research:c")
    assert ok is False
    assert await _live(kg, "Acme", "leads") == [("Alice", False), ("Bob", True)]
    assert any("revert refused" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_revert_dated_on_the_supersession_day_is_refused(kg):
    await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-01-10", source="researched", provenance="deep_research:a")
    await kg.add_fact("Acme", "leads", "Bob", valid_from="2026-06-01", source="researched", provenance="deep_research:b")
    assert await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-06-01", source="researched", provenance="deep_research:c") is False
    assert await _live(kg, "Acme", "leads") == [("Alice", False), ("Bob", True)]


@pytest.mark.asyncio
async def test_revert_with_newer_evidence_revives_and_supersedes(kg):
    await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-01-10", source="researched", provenance="deep_research:a")
    await kg.add_fact("Acme", "leads", "Bob", valid_from="2026-06-01", source="researched", provenance="deep_research:b")
    assert await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-09-01", source="researched", provenance="deep_research:c") is True
    assert await _live(kg, "Acme", "leads") == [("Alice", True), ("Bob", False)]


@pytest.mark.asyncio
async def test_retired_without_a_superseding_row_revives(kg):
    await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-01-10", source="researched", provenance="deep_research:a")
    kg._db.execute("UPDATE kg_facts SET valid_to = '2026-02-01 00:00:00' WHERE LOWER(subject)='acme'")
    assert await kg.add_fact("Acme", "leads", "Alice", valid_from="2026-01-05", source="researched", provenance="deep_research:c") is True
    assert await _live(kg, "Acme", "leads") == [("Alice", True)]


@pytest.mark.asyncio
async def test_multi_valued_predicate_is_unaffected(kg):
    await kg.add_fact("Acme", "contains", "Widgets", valid_from="2026-01-10", source="researched", provenance="deep_research:a")
    await kg.add_fact("Acme", "contains", "Gadgets", valid_from="2026-06-01", source="researched", provenance="deep_research:b")
    kg._db.execute("UPDATE kg_facts SET valid_to = '2026-02-01 00:00:00' WHERE LOWER(object)='widgets'")
    assert await kg.add_fact("Acme", "contains", "Widgets", valid_from="2026-03-01", source="researched", provenance="deep_research:c") is True
    assert await _live(kg, "Acme", "contains") == [("Widgets", True), ("Gadgets", True)]
