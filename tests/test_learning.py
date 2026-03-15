"""Tests for Phase 5: Learning engine and Skills."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core.learning import LearningEngine, Correction, is_likely_correction
from app.core.skills import SkillStore


# ===========================================================================
# Correction Detection (regex)
# ===========================================================================

class TestCorrectionDetection:
    def test_detects_actually(self):
        assert is_likely_correction("Actually, the answer is 42")

    def test_detects_thats_wrong(self):
        assert is_likely_correction("That's wrong, it should be X")

    def test_detects_youre_wrong(self):
        assert is_likely_correction("You're wrong about that")

    def test_detects_correct_answer_is(self):
        assert is_likely_correction("The correct answer is Y")

    def test_detects_not_quite(self):
        assert is_likely_correction("Not quite, let me explain")

    def test_normal_message_not_correction(self):
        assert not is_likely_correction("What's the weather today?")

    def test_empty_not_correction(self):
        assert not is_likely_correction("")

    def test_greeting_not_correction(self):
        assert not is_likely_correction("Hello, how are you?")


# ===========================================================================
# Learning Engine
# ===========================================================================

class TestLearningEngine:
    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_save_lesson(self, engine):
        correction = Correction(
            user_message="Actually, Python was created by Guido van Rossum",
            previous_answer="Python was created by James Gosling",
            topic="Python creator",
            correct_answer="Python was created by Guido van Rossum",
            wrong_answer="James Gosling created Python",
        )
        lesson_id = engine.save_lesson(correction)
        assert lesson_id > 0

    def test_get_relevant_lessons(self, engine):
        # Save a lesson
        correction = Correction(
            user_message="The capital of France is Paris",
            previous_answer="The capital of France is London",
            topic="capital France",
            correct_answer="The capital of France is Paris",
            wrong_answer="The capital of France is London",
        )
        engine.save_lesson(correction)

        # Query should find it
        lessons = engine.get_relevant_lessons("What is the capital of France?")
        assert len(lessons) == 1
        assert "Paris" in lessons[0].correct_answer

    def test_no_matching_lessons(self, engine):
        correction = Correction(
            user_message="fix", previous_answer="wrong",
            topic="Python version", correct_answer="Python version is 3.12",
        )
        engine.save_lesson(correction)

        # Unrelated query
        lessons = engine.get_relevant_lessons("What is quantum physics?")
        assert len(lessons) == 0

    def test_lesson_retrieval_count_increments(self, engine):
        correction = Correction(
            user_message="correction", previous_answer="bad answer here",
            topic="test topic keyword",
            correct_answer="the correct good answer for this topic",
        )
        lesson_id = engine.save_lesson(correction)

        # Retrieve twice
        engine.get_relevant_lessons("test topic keyword")
        engine.get_relevant_lessons("test topic keyword")

        # Check retrieval count
        from app.database import get_db
        row = engine._db.fetchone("SELECT times_retrieved FROM lessons WHERE id = ?", (lesson_id,))
        assert row["times_retrieved"] == 2

    def test_mark_lesson_helpful(self, engine):
        correction = Correction(
            user_message="fix this mistake please", previous_answer="the bad answer",
            topic="test helpful", correct_answer="the correct answer here",
        )
        lesson_id = engine.save_lesson(correction)
        engine.mark_lesson_helpful(lesson_id)

        row = engine._db.fetchone("SELECT times_helpful, confidence FROM lessons WHERE id = ?", (lesson_id,))
        assert row["times_helpful"] == 1
        assert row["confidence"] > 0.8  # Should have increased

    @pytest.mark.asyncio
    async def test_save_training_pair(self, engine, tmp_path):
        training_path = str(tmp_path / "training.jsonl")

        with patch("app.core.learning.config") as mock_config:
            mock_config.TRAINING_DATA_PATH = training_path
            mock_config.MAX_TRAINING_PAIRS = 10000

            await engine.save_training_pair(
                query="What created Python?",
                bad_answer="James Gosling",
                good_answer="Guido van Rossum",
            )

        # Read the JSONL file
        path = tmp_path / "training.jsonl"
        assert path.exists()
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["query"] == "What created Python?"
        assert entry["chosen"] == "Guido van Rossum"
        assert entry["rejected"] == "James Gosling"
        assert "timestamp" in entry

    def test_get_all_lessons(self, engine):
        for i in range(3):
            engine.save_lesson(Correction(
                user_message=f"correction {i}",
                previous_answer="the bad answer",
                topic=f"topic {i}",
                correct_answer=f"the correct answer for topic {i}",
            ))
        lessons = engine.get_all_lessons()
        assert len(lessons) == 3

    def test_delete_lesson(self, engine):
        lesson_id = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="the bad answer",
            topic="test delete", correct_answer="the correct answer here",
        ))
        assert engine.delete_lesson(lesson_id)
        assert not engine.delete_lesson(lesson_id)  # Already deleted

    def test_get_metrics(self, engine):
        engine.save_lesson(Correction(
            user_message="fix this", previous_answer="the bad answer",
            topic="test metrics", correct_answer="the correct answer here",
        ))
        metrics = engine.get_metrics()
        assert metrics["total_lessons"] == 1
        assert metrics["total_corrections"] == 1

    @pytest.mark.asyncio
    async def test_detect_correction_with_llm(self, engine):
        """Test LLM-based correction detection (mocked)."""
        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({
                "is_correction": True,
                "topic": "Python creator",
                "wrong_answer": "James Gosling",
                "correct_answer": "Guido van Rossum",
            }))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            correction = await engine.detect_correction(
                "Actually, Python was created by Guido van Rossum",
                "Python was created by James Gosling",
            )
            assert correction is not None
            assert correction.correct_answer == "Guido van Rossum"

    @pytest.mark.asyncio
    async def test_detect_non_correction(self, engine):
        """Non-correction messages should return None."""
        correction = await engine.detect_correction(
            "What's the weather?",
            "Previous answer here",
        )
        assert correction is None

    @pytest.mark.asyncio
    async def test_detect_correction_timeout_returns_none(self, engine):
        """Item 2: LLM timeout during correction detection should return None."""
        import asyncio

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(60)  # Way longer than 15s timeout
            return "{}"

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = slow_llm

            correction = await engine.detect_correction(
                "Actually, the answer is 42",
                "The answer is 0",
            )
            assert correction is None

    @pytest.mark.asyncio
    async def test_detect_correction_llm_error_returns_none(self, engine):
        """Item 2: LLM exception during correction detection should return None."""
        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=RuntimeError("LLM crashed"))

            correction = await engine.detect_correction(
                "Actually, the answer is 42",
                "The answer is 0",
            )
            assert correction is None


# ===========================================================================
# Bounded Lessons (Item 6)
# ===========================================================================

class TestBoundedLessons:
    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_prune_removes_low_confidence_first(self, engine):
        """Item 6: When over MAX_LESSONS, lowest confidence lessons are pruned."""
        with patch("app.core.learning.config") as mock_config:
            mock_config.MAX_LESSONS = 3
            mock_config.MAX_LESSONS_IN_PROMPT = 5
            mock_config.TRAINING_DATA_PATH = "/tmp/unused.jsonl"
            mock_config.MAX_TRAINING_PAIRS = 10000

            # Insert 4 lessons — the 4th should trigger pruning
            for i in range(4):
                correction = Correction(
                    user_message=f"correction for bounded topic {i}",
                    previous_answer="the bad answer here",
                    topic=f"bounded_topic_{i}",
                    correct_answer=f"correct answer for bounded topic {i}",
                )
                engine.save_lesson(correction)

            # Set different confidences
            engine._db.execute("UPDATE lessons SET confidence = 0.2 WHERE topic = 'bounded_topic_0'")
            engine._db.execute("UPDATE lessons SET confidence = 0.9 WHERE topic = 'bounded_topic_1'")
            engine._db.execute("UPDATE lessons SET confidence = 0.5 WHERE topic = 'bounded_topic_2'")

            # Save one more to trigger pruning
            engine.save_lesson(Correction(
                user_message="one more correction added",
                previous_answer="the bad answer here",
                topic="bounded_topic_4",
                correct_answer="correct answer for bounded topic 4",
            ))

            count = engine._db.fetchone("SELECT COUNT(*) as c FROM lessons")
            assert count["c"] <= 3

    def test_no_prune_under_limit(self, engine):
        """Item 6: No pruning when under MAX_LESSONS."""
        with patch("app.core.learning.config") as mock_config:
            mock_config.MAX_LESSONS = 100
            mock_config.MAX_LESSONS_IN_PROMPT = 5
            mock_config.TRAINING_DATA_PATH = "/tmp/unused.jsonl"
            mock_config.MAX_TRAINING_PAIRS = 10000

            for i in range(3):
                engine.save_lesson(Correction(
                    user_message=f"correction for noprunetest {i}",
                    previous_answer="the bad answer here",
                    topic=f"noprunetest_{i}",
                    correct_answer=f"correct answer for noprunetest topic {i}",
                ))

            count = engine._db.fetchone("SELECT COUNT(*) as c FROM lessons")
            assert count["c"] == 3


# ===========================================================================
# Fuzzy Lesson Dedup (Item 8)
# ===========================================================================

class TestFuzzyLessonDedup:
    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_case_insensitive_dedup(self, engine):
        """Item 8: Lessons with same content but different case should dedup."""
        s1 = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="bad",
            topic="Python Creator",
            correct_answer="Guido van Rossum created Python",
        ))
        s2 = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="bad",
            topic="python creator",
            correct_answer="guido van rossum created python",
        ))
        # Should dedup — same lesson ID returned
        assert s1 == s2

    def test_reworded_dedup(self, engine):
        """Item 8: Similar lessons with minor rewording should dedup (Jaccard >= 0.7)."""
        s1 = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="bad",
            topic="Capital Australia",
            correct_answer="The capital of Australia is Canberra",
        ))
        # Slightly reworded version
        s2 = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="bad",
            topic="Australia Capital",
            correct_answer="Canberra is the capital of Australia",
        ))
        assert s1 == s2  # Should dedup

    def test_different_content_no_dedup(self, engine):
        """Completely different lessons should NOT dedup."""
        s1 = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="bad",
            topic="Python Creator",
            correct_answer="Guido van Rossum created Python",
        ))
        s2 = engine.save_lesson(Correction(
            user_message="fix this", previous_answer="bad",
            topic="Mars Sky Color",
            correct_answer="The Mars sky is butterscotch colored",
        ))
        assert s1 != s2  # Different lessons


# ===========================================================================
# Skills
# ===========================================================================

class TestSkillStore:
    @pytest.fixture
    def store(self, db):
        return SkillStore(db)

    def test_create_skill(self, store):
        skill_id = store.create_skill(
            name="crypto_price",
            trigger_pattern=r"(?i)price of \w+",
            steps=[{"tool": "web_search", "args_template": {"query": "current price of {entity}"}}],
            answer_template="The current price is {result}.",
        )
        assert skill_id > 0

    def test_get_matching_skill(self, store):
        store.create_skill(
            name="crypto_price",
            trigger_pattern=r"(?i)price of \w+",
            steps=[{"tool": "web_search", "args_template": {"query": "price"}}],
        )

        skill = store.get_matching_skill("What's the price of Bitcoin?")
        assert skill is not None
        assert skill.name == "crypto_price"

    def test_no_matching_skill(self, store):
        store.create_skill(
            name="crypto_price",
            trigger_pattern=r"(?i)price of \w+",
            steps=[],
        )
        skill = store.get_matching_skill("How do I cook pasta?")
        assert skill is None

    def test_get_all_skills(self, store):
        store.create_skill("s1", r"pattern1", [])
        store.create_skill("s2", r"pattern2", [])
        skills = store.get_all_skills()
        assert len(skills) == 2

    def test_toggle_skill(self, store):
        skill_id = store.create_skill("test", r"pattern", [])
        store.toggle_skill(skill_id, False)
        skill = store.get_skill(skill_id)
        assert not skill.enabled

        store.toggle_skill(skill_id, True)
        skill = store.get_skill(skill_id)
        assert skill.enabled

    def test_disabled_skill_not_matched(self, store):
        skill_id = store.create_skill(
            "test", r"(?i)hello", [])
        store.toggle_skill(skill_id, False)

        skill = store.get_matching_skill("hello world")
        assert skill is None

    def test_record_use(self, store):
        skill_id = store.create_skill("test", r"pattern", [])
        store.record_use(skill_id, True)
        store.record_use(skill_id, True)
        store.record_use(skill_id, False)

        skill = store.get_skill(skill_id)
        assert skill.times_used == 3
        # EMA (alpha=0.15) from initial 0.7: 0.7 -> 0.745 -> 0.78325 -> 0.6657625
        assert 0.6 < skill.success_rate < 0.7

    def test_delete_skill(self, store):
        skill_id = store.create_skill("test", r"pattern", [])
        assert store.delete_skill(skill_id)
        assert not store.delete_skill(skill_id)

    def test_get_active_skills(self, store):
        store.create_skill("active1", r"p1", [])
        s2_id = store.create_skill("active2", r"p2", [])
        store.create_skill("active3", r"p3", [])
        store.toggle_skill(s2_id, False)

        active = store.get_active_skills()
        assert len(active) == 2
        names = {s.name for s in active}
        assert "active2" not in names

    def test_skill_steps_stored_as_json(self, store):
        steps = [
            {"tool": "web_search", "args_template": {"query": "price of {entity}"}},
            {"tool": "calculator", "args_template": {"expression": "{price} * 100"}},
        ]
        skill_id = store.create_skill("multi_step", r"test", steps)
        skill = store.get_skill(skill_id)
        assert len(skill.steps) == 2
        assert skill.steps[0]["tool"] == "web_search"


# ===========================================================================
# Lesson confidence decay (Fix 5)
# ===========================================================================

class TestLessonDecay:
    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_decay_stale_lessons(self, engine):
        """Fix 5: Unretrieved lessons older than N days should lose confidence."""
        correction = Correction(
            user_message="fix this", previous_answer="bad",
            topic="stale decay test topic",
            correct_answer="correct answer for stale decay test",
        )
        lesson_id = engine.save_lesson(correction)

        # Force old timestamp and 0 retrievals
        engine._db.execute(
            "UPDATE lessons SET created_at = datetime('now', '-60 days'), times_retrieved = 0 WHERE id = ?",
            (lesson_id,),
        )

        original = engine._db.fetchone("SELECT confidence FROM lessons WHERE id = ?", (lesson_id,))
        original_conf = original["confidence"]

        decayed_count = engine.decay_stale_lessons(days=30)
        assert decayed_count == 1

        row = engine._db.fetchone("SELECT confidence FROM lessons WHERE id = ?", (lesson_id,))
        assert row["confidence"] < original_conf

    def test_no_decay_for_fresh_lessons(self, engine):
        """Fix 5: Recently created lessons should NOT decay."""
        correction = Correction(
            user_message="fix this", previous_answer="bad",
            topic="fresh decay test topic",
            correct_answer="correct answer for fresh decay test",
        )
        engine.save_lesson(correction)

        decayed_count = engine.decay_stale_lessons(days=30)
        assert decayed_count == 0

    def test_no_decay_for_retrieved_lessons(self, engine):
        """Fix 5: Lessons that have been retrieved should NOT decay even if old."""
        correction = Correction(
            user_message="fix this", previous_answer="bad",
            topic="retrieved decay test topic",
            correct_answer="correct answer for retrieved decay test",
        )
        lesson_id = engine.save_lesson(correction)

        # Force old timestamp BUT has retrievals
        engine._db.execute(
            "UPDATE lessons SET created_at = datetime('now', '-60 days'), times_retrieved = 5 WHERE id = ?",
            (lesson_id,),
        )

        decayed_count = engine.decay_stale_lessons(days=30)
        assert decayed_count == 0


# ===========================================================================
# Lessons_used accuracy (Fix 8)
# ===========================================================================

class TestLessonRelevanceAccuracy:
    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_single_word_overlap_rejected_for_multiword_query(self, engine):
        """Fix 8: A lesson sharing only 1 word should NOT match a 2+ word query."""
        # Lesson about earthquakes in Japan
        engine.save_lesson(Correction(
            user_message="fix", previous_answer="bad",
            topic="earthquake data sources",
            correct_answer="Use USGS for earthquake data in Japan",
        ))

        # "capital of japan" shares "japan" but has 2 non-stop words
        lessons = engine.get_relevant_lessons("What is the capital of Japan?")
        assert len(lessons) == 0

    def test_two_word_overlap_still_matches(self, engine):
        """Fix 8: A lesson sharing 2+ words should still match."""
        engine.save_lesson(Correction(
            user_message="fix", previous_answer="bad",
            topic="capital Japan",
            correct_answer="The capital of Japan is Tokyo",
        ))

        lessons = engine.get_relevant_lessons("What is the capital of Japan?")
        assert len(lessons) == 1

    def test_single_word_query_still_matches(self, engine):
        """Fix 8: A single non-stop-word query should still match on 1 overlap."""
        engine.save_lesson(Correction(
            user_message="fix", previous_answer="bad",
            topic="Bitcoin halving",
            correct_answer="Bitcoin halving occurs every 4 years",
        ))

        # "bitcoin" alone — 1 non-stop word, 1 overlap — should match
        lessons = engine.get_relevant_lessons("Bitcoin")
        assert len(lessons) == 1
