"""Vector-index rot defenses (2026-08-25).

hnswlib never compacts delete-tombstones: Chroma's delete() only marks
elements dead in the HNSW graph. A churny collection (lessons hit ~335
tombstones vs 42 live rows on 2026-08-22) eventually fails every query
with n_results >= the surviving neighborhood ("Cannot return the results
in a contigious 2D array") and the vector arm dies silently — 133
warnings over 3 days with no alert and no self-heal path.

These tests cover the three defenses:
- telemetry (record_failure / failures_in_window) that System Health
  turns into a loud alert,
- assess()/sweep() rot detection that daily maintenance uses to trigger
  drop+rebuild BEFORE queries fail,
- rebuild mechanics (drop + backfill from the SQLite source of truth)
  and the k-degrade retry inside get_relevant_lessons.
"""

from __future__ import annotations

import time

import pytest


TOMBSTONE_MSG = (
    "Cannot return the results in a contigious 2D array. "
    "Probably ef or M is too small"
)


def _seed_lessons(db, ids: range) -> None:
    for i in ids:
        db.execute(
            "INSERT INTO lessons (id, topic, correct_answer, lesson_text) "
            "VALUES (?, ?, ?, ?)",
            (i, f"topic {i}", f"answer {i}", f"lesson body {i} about subject{i}"),
        )


class TestTombstoneErrorMatch:
    def test_matches_hnswlib_typo_message(self):
        from app.core.vector_health import is_tombstone_error

        assert is_tombstone_error(RuntimeError(TOMBSTONE_MSG))

    def test_matches_corrected_spelling_variant(self):
        from app.core.vector_health import is_tombstone_error

        assert is_tombstone_error(
            RuntimeError("Cannot return the results in a contiguous 2D array")
        )

    def test_ignores_unrelated_errors(self):
        from app.core.vector_health import is_tombstone_error

        assert not is_tombstone_error(ValueError("dimension mismatch"))


class TestFailureTelemetry:
    def test_failures_counted_per_store_in_window(self):
        from app.core import vector_health as vh

        vh.reset()
        vh.record_failure("lessons")
        vh.record_failure("lessons")
        vh.record_failure("kg_facts")
        counts = vh.failures_in_window(hours=24)
        assert counts["lessons"] == 2
        assert counts["kg_facts"] == 1

    def test_old_failures_age_out_of_window(self):
        from app.core import vector_health as vh

        vh.reset()
        vh.record_failure("lessons", when=time.time() - 25 * 3600)
        vh.record_failure("lessons")
        assert vh.failures_in_window(hours=24)["lessons"] == 1


class TestAssess:
    def test_canary_tombstone_failure_triggers_rebuild(self):
        from app.core.vector_health import assess

        def canary():
            raise RuntimeError(TOMBSTONE_MSG)

        verdict = assess(live=42, ever=50, canary=canary)
        assert verdict["needs_rebuild"] is True
        assert verdict["reason"] == "canary"

    def test_churn_without_watermark_triggers_rebuild(self):
        from app.core.vector_health import assess

        # The live 2026-08-24 lessons numbers: 42 live rows, ids up to 380.
        verdict = assess(live=42, ever=380, canary=lambda: None)
        assert verdict["needs_rebuild"] is True
        assert verdict["reason"] == "churn"

    def test_healthy_collection_not_rebuilt(self):
        from app.core.vector_health import assess

        verdict = assess(live=5000, ever=5100, canary=lambda: None)
        assert verdict["needs_rebuild"] is False

    def test_watermark_suppresses_prerebuild_churn(self):
        from app.core.vector_health import assess

        # After a rebuild at ever=380/live=42, the historical churn must not
        # re-trigger a rebuild every day forever.
        verdict = assess(
            live=42, ever=380, canary=lambda: None, watermark=(380, 42)
        )
        assert verdict["needs_rebuild"] is False

    def test_non_tombstone_canary_error_does_not_rebuild(self):
        from app.core.vector_health import assess

        def canary():
            raise ValueError("connection refused")

        verdict = assess(live=42, ever=380, canary=canary)
        # An embedder hiccup is not index rot; churn logic still applies,
        # but the canary error itself must not be read as rot.
        assert verdict["reason"] != "canary"


class TestWatermarkPersistence:
    def test_watermark_roundtrip(self, db):
        from app.core import vector_health as vh

        assert vh.get_watermark(db, "lessons") is None
        vh.set_watermark(db, "lessons", ever=380, live=42)
        assert vh.get_watermark(db, "lessons") == (380, 42)
        vh.set_watermark(db, "lessons", ever=400, live=50)
        assert vh.get_watermark(db, "lessons") == (400, 50)


class TestSweep:
    def test_sweep_rebuilds_rotten_and_leaves_healthy(self):
        from app.core.vector_health import sweep

        calls = []
        targets = [
            {
                "name": "lessons",
                "live": 42,
                "ever": 380,
                "canary": lambda: None,
                "watermark": None,
                "rebuild": lambda: calls.append("lessons") or 42,
                "record_watermark": lambda: calls.append("wm-lessons"),
            },
            {
                "name": "kg_facts",
                "live": 5000,
                "ever": 5100,
                "canary": lambda: None,
                "watermark": None,
                "rebuild": lambda: calls.append("kg_facts") or 5000,
                "record_watermark": lambda: calls.append("wm-kg"),
            },
        ]
        lines = sweep(targets)
        assert "lessons" in calls
        assert "wm-lessons" in calls
        assert "kg_facts" not in calls
        assert any("REBUILT" in ln and "lessons" in ln for ln in lines)

    def test_sweep_survives_rebuild_failure(self):
        from app.core.vector_health import sweep

        def boom():
            raise RuntimeError("chroma exploded")

        targets = [
            {
                "name": "lessons",
                "live": 42,
                "ever": 380,
                "canary": lambda: None,
                "watermark": None,
                "rebuild": boom,
                "record_watermark": lambda: None,
            }
        ]
        lines = sweep(targets)  # must not raise
        assert any("FAILED" in ln for ln in lines)


class TestLessonsRebuildMechanics:
    def test_rebuild_drops_tombstones_and_matches_sql(self, db, tmp_path):
        """Rebuild must produce a fresh index holding exactly the live rows,
        serving k=10 queries — the k that failed on the saturated index."""
        from app.core.learning import LearningEngine

        _seed_lessons(db, range(339, 381))  # 42 live rows, ids 339..380
        engine = LearningEngine(db=db)

        collection = engine._get_lessons_collection()
        assert collection is not None
        # Simulate the churn history: 380 docs ever indexed, 338 deleted.
        ids = [str(i) for i in range(1, 381)]
        collection.upsert(ids=ids, documents=[f"doc {i}" for i in ids])
        collection.delete(ids=[str(i) for i in range(1, 339)])

        n = engine.rebuild_lessons_vectors(reason="test")
        assert n == 42

        rebuilt = engine._get_lessons_collection()
        assert rebuilt.count() == 42
        res = rebuilt.query(query_texts=["subject350"], n_results=10)
        assert len(res["ids"][0]) == 10

    def test_rebuild_records_watermark(self, db):
        from app.core import vector_health as vh
        from app.core.learning import LearningEngine

        _seed_lessons(db, range(339, 381))
        engine = LearningEngine(db=db)
        engine.rebuild_lessons_vectors(reason="test")
        assert vh.get_watermark(db, "lessons") == (380, 42)


class _TombstoneCollection:
    """Fake collection reproducing the live failure: k>5 raises, k<=5 works."""

    def __init__(self, lesson_id: int, count: int = 42):
        self._id = str(lesson_id)
        self._count = count
        self.queries: list[int] = []

    def count(self) -> int:
        return self._count

    def query(self, query_texts=None, n_results=10, include=None):
        self.queries.append(n_results)
        if n_results > 5:
            raise RuntimeError(TOMBSTONE_MSG)
        return {
            "ids": [[self._id]],
            "distances": [[0.2]],
            "documents": [["stored doc text"]],
            "metadatas": [[{"document_id": "d1", "source": "s", "title": "t"}]],
        }


class TestGetRelevantLessonsDegrade:
    def test_degrades_to_small_k_instead_of_losing_vector_arm(self, db):
        """On tombstone failure the vector arm must retry at a k below the
        failure floor — not silently fall back to keyword-only."""
        from app.core import vector_health as vh
        from app.core.learning import LearningEngine

        vh.reset()
        # One lesson whose text shares NO keywords with the query: only the
        # vector arm can surface it.
        db.execute(
            "INSERT INTO lessons (id, topic, correct_answer, lesson_text) "
            "VALUES (77, 'astronomy', 'Neptune', 'furthest planet fact')",
        )
        engine = LearningEngine(db=db)
        engine._lessons_collection = _TombstoneCollection(77)

        results = engine.get_relevant_lessons("what orbits outermost", limit=5)
        assert any(lesson.id == 77 for lesson in results)

    def test_tombstone_failure_is_recorded_for_alerting(self, db):
        from app.core import vector_health as vh
        from app.core.learning import LearningEngine

        vh.reset()
        db.execute(
            "INSERT INTO lessons (id, topic, correct_answer, lesson_text) "
            "VALUES (77, 'astronomy', 'Neptune', 'furthest planet fact')",
        )
        engine = LearningEngine(db=db)
        engine._lessons_collection = _TombstoneCollection(77)
        engine.get_relevant_lessons("what orbits outermost", limit=5)
        assert vh.failures_in_window(hours=1).get("lessons", 0) >= 1


class TestKgRebuildMechanics:
    def test_rebuild_matches_live_set_and_serves_k10(self, db):
        import asyncio

        from app.core import vector_health as vh
        from app.core.kg import KnowledgeGraph

        kg = KnowledgeGraph(db)

        async def _seed():
            for i in range(40):
                await kg.add_fact(f"Entity{i}", "related_to", f"Object{i}",
                                  confidence=0.95)

        asyncio.run(_seed())

        collection = kg._get_collection()
        assert collection is not None
        # Simulate churn history: stale ids indexed then deleted.
        stale = [str(i) for i in range(10_000, 10_300)]
        collection.upsert(ids=stale, documents=[f"stale {i}" for i in stale])
        collection.delete(ids=stale)

        n = kg.rebuild_vectors(reason="test")
        assert n == 40
        rebuilt = kg._get_collection()
        assert rebuilt.count() == 40
        res = rebuilt.query(query_texts=["Entity7"], n_results=10)
        assert len(res["ids"][0]) == 10
        assert vh.get_watermark(db, "kg_facts") is not None

    def test_kg_vector_arm_degrades_on_tombstone_failure(self, db):
        import asyncio

        from app.core import vector_health as vh
        from app.core.kg import KnowledgeGraph

        vh.reset()
        kg = KnowledgeGraph(db)

        async def _seed():
            await kg.add_fact("Zanthor", "leads", "Acme Corp", confidence=0.95)

        asyncio.run(_seed())
        row = db.fetchone("SELECT id FROM kg_facts WHERE subject='Zanthor'")
        kg._collection = _TombstoneCollection(row["id"], count=5000)

        facts = kg.get_relevant_facts("who runs the company", limit=5)
        assert any(f.subject == "Zanthor" for f in facts)
        assert vh.failures_in_window(hours=1).get("kg_facts", 0) >= 1


class TestRetrieverDegrade:
    def test_vector_search_degrades_instead_of_returning_empty(self):
        from app.core import vector_health as vh
        from app.core.retriever import Retriever

        vh.reset()
        r = Retriever(chroma_collection=_TombstoneCollection(1, count=100))
        chunks = r._vector_search("any query", top_k=10)
        assert len(chunks) == 1
        assert chunks[0].content == "stored doc text"
        assert vh.failures_in_window(hours=1).get("documents", 0) >= 1
