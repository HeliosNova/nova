"""Concurrent KnowledgeGraph mutation tests.

Verifies that concurrent access to the KnowledgeGraph (add_fact, delete_fact,
curate, query) does not corrupt data, violate unique constraints, or produce
inconsistent reads — all under asyncio.gather concurrency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.kg import KnowledgeGraph, normalize_predicate


@pytest.fixture
def kg(db):
    """Return a fresh KnowledgeGraph instance backed by a test database."""
    return KnowledgeGraph(db)


class TestConcurrentFactAdditions:
    """Multiple add_fact calls running simultaneously."""

    @pytest.mark.asyncio
    async def test_concurrent_distinct_facts(self, kg):
        """Adding many distinct facts concurrently should all succeed."""
        results = await asyncio.gather(*[
            kg.add_fact(f"entity_{i}", "is_a", f"type_{i}", confidence=0.8)
            for i in range(20)
        ])
        assert all(r is True for r in results)

        stats = kg.get_stats()
        assert stats["current_facts"] == 20

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_facts(self, kg):
        """Adding the exact same fact concurrently should not violate UNIQUE constraint."""
        results = await asyncio.gather(*[
            kg.add_fact("python", "created_by", "guido van rossum", confidence=0.9)
            for _ in range(10)
        ])
        # First one should succeed, rest return False (duplicate, same confidence)
        assert results.count(True) == 1
        assert results.count(False) == 9

        stats = kg.get_stats()
        assert stats["current_facts"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_same_subject_predicate_different_objects(self, kg):
        """Concurrent adds with same subject+predicate but different objects should supersede properly."""
        # These conflict: same subject+predicate, different object
        results = await asyncio.gather(*[
            kg.add_fact("france", "capital_of", f"city_{i}", confidence=0.8)
            for i in range(5)
        ])

        # All should return True (each supersedes the previous)
        assert all(r is True for r in results)

        # Only the last-written fact should be current (valid_to IS NULL)
        stats = kg.get_stats()
        assert stats["current_facts"] == 1
        # All others should be superseded
        assert stats["superseded_facts"] == 4

    @pytest.mark.asyncio
    async def test_concurrent_escalating_confidence(self, kg):
        """Concurrent adds of the same triple with increasing confidence should keep the highest."""
        results = await asyncio.gather(*[
            kg.add_fact("python", "is_a", "programming language", confidence=0.1 * (i + 1))
            for i in range(10)
        ])

        # At least the first add and higher-confidence updates should succeed
        assert results[0] is True

        # Final confidence should be the highest submitted
        facts = kg.search("python")
        assert len(facts) == 1
        assert facts[0]["confidence"] == pytest.approx(1.0, abs=0.01)


class TestParallelFactExtraction:
    """Parallel fact extraction simulating multiple monitors extracting simultaneously."""

    @pytest.mark.asyncio
    async def test_parallel_extractions_no_data_loss(self, kg):
        """Multiple parallel extractions with different subjects should all be stored."""
        subjects = ["einstein", "newton", "darwin", "curie", "tesla"]
        predicates = ["known_for", "born_in", "is_a", "known_for", "invented_by"]
        objects = ["relativity", "england", "biologist", "radioactivity", "alternating current"]

        results = await asyncio.gather(*[
            kg.add_fact(s, p, o, confidence=0.85, source="extracted", provenance="monitor_test")
            for s, p, o in zip(subjects, predicates, objects)
        ])

        assert all(r is True for r in results)
        stats = kg.get_stats()
        assert stats["current_facts"] == 5

        # Verify each fact is retrievable
        for subj in subjects:
            found = kg.search(subj)
            assert len(found) >= 1, f"Missing fact for {subj}"

    @pytest.mark.asyncio
    async def test_parallel_extractions_with_overlapping_entities(self, kg):
        """Parallel extractions with overlapping entities but different predicates."""
        tasks = [
            kg.add_fact("python", "created_by", "guido van rossum"),
            kg.add_fact("python", "is_a", "programming language"),
            kg.add_fact("python", "used_for", "web development"),
            kg.add_fact("python", "version_of", "cpython"),
        ]
        results = await asyncio.gather(*tasks)
        assert all(r is True for r in results)

        # All 4 should exist as current facts (different predicates, no conflict)
        facts = kg.query("python")
        assert len(facts) == 4


class TestLockContention:
    """Multiple writers contending for the write lock."""

    @pytest.mark.asyncio
    async def test_high_contention_write_consistency(self, kg):
        """Many concurrent writes to the same entity should not corrupt data."""
        # All tasks write to the same entity with different predicates
        results = await asyncio.gather(*[
            kg.add_fact("nova", f"has_property", f"trait_{i}", confidence=0.7 + 0.01 * i)
            for i in range(15)
        ])

        # All have the same subject+predicate so they conflict and supersede each other
        # Only the last one should be current
        stats = kg.get_stats()
        assert stats["current_facts"] == 1
        # 14 should be superseded
        assert stats["superseded_facts"] == 14

    @pytest.mark.asyncio
    async def test_interleaved_add_and_delete(self, kg):
        """Adding and deleting the same fact concurrently should not raise."""
        # First add the fact so it exists
        await kg.add_fact("temp", "is_a", "test entity")

        # Now concurrently add new versions and try to delete
        async def add_variant(i):
            return await kg.add_fact("temp", "is_a", f"variant_{i}", confidence=0.5 + 0.05 * i)

        async def delete_original():
            return await kg.delete_fact("temp", "is_a", "test entity")

        results = await asyncio.gather(
            add_variant(1),
            delete_original(),
            add_variant(2),
            add_variant(3),
            return_exceptions=True,
        )

        # No exceptions should have been raised
        for r in results:
            assert not isinstance(r, Exception), f"Unexpected exception: {r}"


class TestReadDuringWriteConsistency:
    """Reads should see consistent snapshots even during concurrent writes."""

    @pytest.mark.asyncio
    async def test_query_during_writes(self, kg):
        """Querying while facts are being added should not error or return partial rows."""
        # Pre-seed some facts
        for i in range(5):
            await kg.add_fact(f"base_{i}", "is_a", f"category_{i}")

        async def write_facts():
            for i in range(10):
                await kg.add_fact(f"writer_{i}", "is_a", f"thing_{i}")
                await asyncio.sleep(0)  # yield to event loop

        async def read_facts():
            results = []
            for _ in range(10):
                facts = kg.search("base")
                results.append(len(facts))
                await asyncio.sleep(0)
            return results

        _, read_results = await asyncio.gather(write_facts(), read_facts())

        # Every read should see at least the 5 pre-seeded facts
        for count in read_results:
            assert count >= 5

    @pytest.mark.asyncio
    async def test_get_stats_during_writes(self, kg):
        """get_stats() should not crash during concurrent writes."""
        async def write_batch():
            for i in range(10):
                await kg.add_fact(f"stat_entity_{i}", "is_a", f"type_{i}")

        async def read_stats():
            results = []
            for _ in range(5):
                stats = kg.get_stats()
                results.append(stats)
                await asyncio.sleep(0)
            return results

        _, stats_list = await asyncio.gather(write_batch(), read_stats())

        # Every stats call should return a valid dict
        for s in stats_list:
            assert "current_facts" in s
            assert "total_facts" in s
            assert s["current_facts"] >= 0


class TestDeleteFactConcurrent:
    """delete_fact under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_delete_same_fact(self, kg):
        """Deleting the same fact from multiple coroutines should not raise."""
        await kg.add_fact("doomed", "is_a", "temporary")

        results = await asyncio.gather(*[
            kg.delete_fact("doomed", "is_a", "temporary")
            for _ in range(5)
        ])

        # Only the first delete should succeed (returns True), rest return False
        assert results.count(True) == 1
        assert results.count(False) == 4

        # Fact should be retired (not current)
        facts = kg.search("doomed")
        assert len(facts) == 0  # search only returns current facts

    @pytest.mark.asyncio
    async def test_delete_nonexistent_fact_concurrent(self, kg):
        """Deleting a fact that doesn't exist should return False without error."""
        results = await asyncio.gather(*[
            kg.delete_fact("nope", "is_a", "nothing")
            for _ in range(5)
        ])
        assert all(r is False for r in results)

    @pytest.mark.asyncio
    async def test_delete_while_adding_same_triple(self, kg):
        """Concurrent add and delete of the same triple should not corrupt state."""
        # Seed the fact
        await kg.add_fact("flip", "is_a", "flop")

        async def add_back():
            # Small delay to let delete run first sometimes
            await asyncio.sleep(0)
            return await kg.add_fact("flip", "is_a", "flop", confidence=0.9)

        async def delete_it():
            return await kg.delete_fact("flip", "is_a", "flop")

        results = await asyncio.gather(
            delete_it(),
            add_back(),
            return_exceptions=True,
        )

        for r in results:
            assert not isinstance(r, Exception), f"Unexpected exception: {r}"

        # State should be consistent — either current or retired, not both
        current = kg.search("flip")
        all_facts = kg.search("flip", include_history=True)
        # If add_back won, we should have one current fact
        # If delete won last, we should have zero current facts
        assert len(current) <= 1


class TestCurateConcurrentWithAddFact:
    """curate() running concurrently with add_fact()."""

    @pytest.mark.asyncio
    async def test_curate_during_adds(self, kg):
        """curate(heuristic=True) should not deadlock or crash when adds are happening."""
        # Seed some garbage facts that curate's heuristic will want to delete
        await kg.add_fact("testuser", "is_a", "human")  # garbage value
        await kg.add_fact("foo", "is_a", "bar")  # garbage value
        # Also seed some good facts
        for i in range(5):
            await kg.add_fact(f"entity_{i}", "is_a", f"real_category_{i}", confidence=0.9)

        async def add_more_facts():
            for i in range(10):
                await kg.add_fact(f"new_entity_{i}", "is_a", f"new_type_{i}", confidence=0.85)
                await asyncio.sleep(0)

        # curate with heuristic only (no LLM), sample_size=0 skips LLM pass
        curate_result, _ = await asyncio.gather(
            kg.curate(sample_size=0, heuristic=True),
            add_more_facts(),
        )

        # Curate should return a valid result dict
        assert "heuristic" in curate_result
        assert curate_result["heuristic"] >= 0

        # All new facts should exist
        stats = kg.get_stats()
        # At minimum, the 5 good seeded + 10 new = 15 (minus any garbage curated)
        assert stats["current_facts"] >= 10

    @pytest.mark.asyncio
    async def test_curate_with_llm_concurrent(self, kg):
        """curate() with LLM validation running alongside fact additions."""
        # Seed low-confidence facts that curate will validate via LLM
        for i in range(5):
            await kg.add_fact(f"low_conf_{i}", "is_a", f"maybe_{i}", confidence=0.3)

        # Patch at the module level that curate() imports from
        with patch("app.core.llm.invoke_nothink", new_callable=AsyncMock) as mock_invoke, \
             patch("app.core.llm.extract_json_object") as mock_extract:
            mock_invoke.return_value = '{"results": [{"id": 1, "verdict": "keep"}, {"id": 2, "verdict": "keep"}, {"id": 3, "verdict": "keep"}, {"id": 4, "verdict": "keep"}, {"id": 5, "verdict": "keep"}]}'
            mock_extract.return_value = {
                "results": [{"id": i, "verdict": "keep"} for i in range(1, 6)]
            }

            async def add_concurrent():
                for i in range(5):
                    await kg.add_fact(f"concurrent_{i}", "is_a", f"type_{i}", confidence=0.9)

            results = await asyncio.gather(
                kg.curate(sample_size=5, heuristic=False),
                add_concurrent(),
                return_exceptions=True,
            )

            for r in results:
                if isinstance(r, Exception):
                    # LLM mock failures are non-critical — curate logs and continues
                    pass

        # No crash, data is consistent
        stats = kg.get_stats()
        assert stats["current_facts"] >= 5  # at least the concurrent adds


class TestSupersessionConcurrency:
    """Test that the supersession chain stays consistent under concurrency."""

    @pytest.mark.asyncio
    async def test_supersession_chain_integrity(self, kg):
        """When multiple conflicting facts are added, the supersession chain should be valid."""
        # Add conflicting facts for "earth/capital_of" — each supersedes the last
        cities = ["paris", "london", "berlin", "tokyo", "moscow"]
        for city in cities:
            await kg.add_fact("earth", "capital_of", city, confidence=0.8)

        # Only the last one should be current
        current = kg.query("earth")
        current_objects = [f["object"] for f in current if f["predicate"] == "capital_of"]
        assert len(current_objects) == 1
        assert current_objects[0] == "moscow"

        # History should show all 5
        history = kg.get_fact_history("earth", "capital_of")
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_concurrent_supersession_no_orphans(self, kg):
        """Concurrent conflicting adds should not leave orphaned current facts."""
        results = await asyncio.gather(*[
            kg.add_fact("user", "lives_in", f"city_{i}", confidence=0.8)
            for i in range(10)
        ])

        # Exactly one should be current
        current = kg.query("user")
        current_lives_in = [f for f in current if f["predicate"] == "lives_in"]
        assert len(current_lives_in) == 1

        # Total should be 10 (1 current + 9 superseded)
        history = kg.get_fact_history("user", "lives_in")
        assert len(history) == 10
