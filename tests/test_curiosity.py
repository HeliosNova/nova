"""Tests for the Curiosity Engine — gap detection, queue, and topic tracking."""

from __future__ import annotations

import pytest

from app.core.curiosity import (
    detect_gaps,
    CuriosityQueue,
    TopicTracker,
    MAX_PENDING,
    MAX_ATTEMPTS,
)


# ===========================================================================
# Gap Detection
# ===========================================================================

class TestGapDetection:
    def test_admission_detected(self):
        gaps = detect_gaps(
            query="What is quantum computing?",
            answer="I don't know much about quantum computing specifics.",
            tool_results=[],
            had_lessons=False, had_kg=False, had_docs=False,
        )
        assert len(gaps) == 1
        assert gaps[0]["source"] == "admission"
        assert gaps[0]["urgency"] == 0.8

    def test_tool_failure_detected(self):
        gaps = detect_gaps(
            query="What is the weather?",
            answer="I tried searching but couldn't find results.",
            tool_results=[{"tool": "web_search", "output": "Error: connection timed out"}],
            had_lessons=False, had_kg=False, had_docs=False,
        )
        assert len(gaps) == 1
        assert gaps[0]["source"] == "tool_failure"
        assert gaps[0]["urgency"] == 0.6

    def test_hedging_detected(self):
        gaps = detect_gaps(
            query="How does X work?",
            answer="I think it might be related to something. It could be that approximately 50% of cases show this.",
            tool_results=[],
            had_lessons=False, had_kg=False, had_docs=False,
        )
        assert len(gaps) == 1
        assert gaps[0]["source"] == "hedging"

    def test_context_gap_detected(self):
        gaps = detect_gaps(
            query="Tell me about the detailed history of the Byzantine Empire and its fall",
            answer="The Byzantine Empire was a major civilization.",
            tool_results=[],
            had_lessons=False, had_kg=False, had_docs=False,
        )
        assert len(gaps) == 1
        assert gaps[0]["source"] == "context_gap"
        assert gaps[0]["urgency"] == 0.3

    def test_no_gap_with_good_answer(self):
        gaps = detect_gaps(
            query="What is Python?",
            answer="Python is a high-level programming language created by Guido van Rossum.",
            tool_results=[],
            had_lessons=True, had_kg=True, had_docs=False,
        )
        assert len(gaps) == 0

    def test_no_gap_short_query_without_context(self):
        """Short queries should not trigger context_gap."""
        gaps = detect_gaps(
            query="Hello",
            answer="Hi there!",
            tool_results=[],
            had_lessons=False, had_kg=False, had_docs=False,
        )
        assert len(gaps) == 0

    def test_topic_capped_at_200(self):
        long_query = "x" * 300
        gaps = detect_gaps(
            query=long_query,
            answer="I don't know the answer to this.",
            tool_results=[],
            had_lessons=False, had_kg=False, had_docs=False,
        )
        assert len(gaps) == 1
        assert len(gaps[0]["topic"]) == 200


# ===========================================================================
# CuriosityQueue
# ===========================================================================

class TestCuriosityQueue:
    @pytest.fixture
    def queue(self, db):
        return CuriosityQueue(db)

    def test_add_item(self, queue):
        item_id = queue.add("quantum computing", source="admission", urgency=0.8)
        assert item_id > 0

    def test_add_empty_rejected(self, queue):
        assert queue.add("") == -1

    def test_dedup_boosts_urgency(self, queue):
        queue.add("quantum computing", urgency=0.5)
        queue.add("quantum computing", urgency=0.3)
        item = queue.get_next()
        assert item is not None
        assert item.urgency == 0.6  # 0.5 + 0.1 boost

    def test_get_next_by_urgency(self, queue):
        queue.add("low priority", urgency=0.2)
        queue.add("high priority", urgency=0.9)
        queue.add("medium priority", urgency=0.5)
        item = queue.get_next()
        assert item is not None
        assert item.topic == "high priority"

    def test_resolve_item(self, queue):
        item_id = queue.add("test topic")
        queue.resolve(item_id, "Found the answer: it's 42")
        # Should not appear in get_next()
        assert queue.get_next() is None
        stats = queue.get_stats()
        assert stats["resolved"] == 1

    def test_fail_item_increments_attempts(self, queue):
        item_id = queue.add("hard topic")
        queue.fail(item_id)
        queue.fail(item_id)
        item = queue.get_next()
        assert item is not None
        assert item.attempts == 2

    def test_fail_exhausts_after_max_attempts(self, queue):
        item_id = queue.add("impossible topic")
        for _ in range(MAX_ATTEMPTS):
            queue.fail(item_id)
        # Should no longer be pending
        assert queue.get_next() is None
        stats = queue.get_stats()
        assert stats["failed"] == 1

    def test_dismiss_item(self, queue):
        item_id = queue.add("not interesting")
        queue.dismiss(item_id)
        assert queue.get_next() is None

    def test_max_pending_evicts_lowest(self, queue):
        # Fill to max
        for i in range(MAX_PENDING):
            queue.add(f"topic_{i}", urgency=0.5)
        # Adding one more should evict lowest urgency
        queue.add("urgent topic", urgency=0.9)
        stats = queue.get_stats()
        assert stats["pending"] <= MAX_PENDING

    def test_get_stats(self, queue):
        queue.add("topic1")
        queue.add("topic2")
        item_id = queue.add("topic3")
        queue.resolve(item_id, "done")
        stats = queue.get_stats()
        assert stats["total"] == 3
        assert stats["pending"] == 2
        assert stats["resolved"] == 1

    def test_get_recent(self, queue):
        queue.add("topic1")
        queue.add("topic2")
        recent = queue.get_recent(limit=5)
        assert len(recent) == 2

    def test_prune_old_items(self, db):
        queue = CuriosityQueue(db)
        item_id = queue.add("old topic")
        queue.resolve(item_id, "done")
        # Force item to be old
        db.execute(
            "UPDATE curiosity_queue SET created_at = datetime('now', '-60 days') WHERE id = ?",
            (item_id,),
        )
        pruned = queue.prune(days=30)
        assert pruned == 1


# ===========================================================================
# TopicTracker
# ===========================================================================

class TestTopicTracker:
    @pytest.fixture
    def tracker(self, db):
        return TopicTracker(db)

    def test_record_topic(self, tracker):
        tracker.record_topic("What is bitcoin?")
        candidates = tracker.get_monitor_candidates(min_count=1, days=7)
        assert len(candidates) == 1
        assert candidates[0]["topic"] == "bitcoin"

    def test_increment_count(self, tracker):
        tracker.record_topic("What is bitcoin?")
        tracker.record_topic("What is bitcoin?")
        tracker.record_topic("What is bitcoin?")
        candidates = tracker.get_monitor_candidates(min_count=3, days=7)
        assert len(candidates) == 1
        assert candidates[0]["query_count"] == 3

    def test_no_candidates_below_threshold(self, tracker):
        tracker.record_topic("What is bitcoin?")
        tracker.record_topic("What is bitcoin?")
        candidates = tracker.get_monitor_candidates(min_count=3, days=7)
        assert len(candidates) == 0

    def test_short_topic_ignored(self, tracker):
        tracker.record_topic("Hi")
        candidates = tracker.get_monitor_candidates(min_count=1, days=7)
        assert len(candidates) == 0

    def test_extract_topic_strips_prefix(self):
        assert TopicTracker._extract_topic("What is bitcoin?") == "bitcoin"
        assert TopicTracker._extract_topic("How do I cook pasta?") == "i cook pasta"
        assert TopicTracker._extract_topic("Tell me about quantum physics") == "quantum physics"

    def test_extract_topic_caps_length(self):
        long_query = "x" * 200
        result = TopicTracker._extract_topic(long_query)
        assert len(result) <= 100
