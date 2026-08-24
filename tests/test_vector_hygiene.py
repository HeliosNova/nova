"""KG vector-index lifecycle hygiene (2026-08-14).

Supersessions, expiries, and quarantine purges never deleted their VECTORS —
the kg_facts collection grew to 3× the live set (15,018 vs 5,023) and every
semantic top-k was ~2/3 dead rows. prune_stale_vectors removes entries whose
fact is no longer live.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.core.kg import KnowledgeGraph


class _FakeCollection:
    def __init__(self, ids):
        self.ids = list(ids)
        self.deleted: list[list[str]] = []

    def get(self, include=None):
        return {"ids": list(self.ids)}

    def delete(self, ids):
        self.deleted.append(list(ids))
        self.ids = [i for i in self.ids if i not in set(ids)]


def test_prune_removes_only_stale_vectors(db):
    kg = KnowledgeGraph(db)
    live_ids = []
    for i in range(3):
        cur = db.execute(
            "INSERT INTO kg_facts (subject, predicate, object, confidence) "
            "VALUES (?, 'is_a', 'thing', 0.9)", (f"live-{i}",))
        live_ids.append(str(cur.lastrowid))
    # a superseded and a quarantined fact — their vectors must go
    cur = db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, superseded_at) "
        "VALUES ('old', 'is_a', 'thing', 0.9, datetime('now'))")
    stale_a = str(cur.lastrowid)
    cur = db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, quarantined) "
        "VALUES ('sus', 'is_a', 'thing', 0.9, 1)")
    stale_b = str(cur.lastrowid)

    fake = _FakeCollection(live_ids + [stale_a, stale_b, "ghost-999"])
    with patch.object(kg, "_get_collection", return_value=fake):
        n = asyncio.run(kg.prune_stale_vectors())
    assert n == 3                                   # 2 stale + 1 ghost
    assert sorted(fake.ids) == sorted(live_ids)     # live vectors untouched
