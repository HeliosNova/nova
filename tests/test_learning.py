"""Tests for Phase 5: Learning engine and Skills."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.learning import (
    LearningEngine,
    Correction,
    is_likely_correction,
    _extract_answer_from_message,
    _fallback_correction,
)
from app.core.prompt import build_system_prompt, format_lessons_for_prompt
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
            wrong_answer="the bad answer",
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
            mock_config.DEDUP_JACCARD_THRESHOLD = 0.85

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
            mock_config.DEDUP_JACCARD_THRESHOLD = 0.85
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


# ===========================================================================
# Knowledge Lessons (add_knowledge_lesson)
# ===========================================================================

class TestKnowledgeLesson:
    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_add_knowledge_lesson(self, engine):
        """Knowledge lessons are saved with default confidence 0.7 and empty wrong_answer."""
        lesson_id = engine.add_knowledge_lesson(
            topic="Mars atmosphere",
            correct_answer="Mars has a thin atmosphere mostly composed of carbon dioxide",
            lesson_text="Mars atmosphere is 95% CO2",
            context="Curiosity research on: Mars atmosphere",
        )
        assert lesson_id > 0

        row = engine._db.fetchone("SELECT * FROM lessons WHERE id = ?", (lesson_id,))
        assert row["confidence"] == 0.7
        assert row["wrong_answer"] == ""
        assert row["topic"] == "Mars atmosphere"
        assert "CO2" in row["lesson_text"]

    def test_knowledge_lesson_dedup(self, engine):
        """Duplicate knowledge lessons should dedup and boost confidence."""
        id1 = engine.add_knowledge_lesson(
            topic="Mars atmosphere",
            correct_answer="Mars has a thin atmosphere mostly composed of carbon dioxide",
            lesson_text="Mars atmosphere is 95% CO2",
        )
        id2 = engine.add_knowledge_lesson(
            topic="Mars atmosphere",
            correct_answer="Mars has a thin atmosphere mostly composed of carbon dioxide",
            lesson_text="Mars atmosphere is 95% CO2",
        )
        assert id1 == id2

        row = engine._db.fetchone("SELECT confidence FROM lessons WHERE id = ?", (id1,))
        assert abs(row["confidence"] - 0.8) < 1e-9  # 0.7 + 0.1

    def test_knowledge_lesson_rejects_garbage(self, engine):
        """Short or error-phrase content should be rejected."""
        assert engine.add_knowledge_lesson(
            topic="test", correct_answer="short", lesson_text="short",
        ) == -1

        assert engine.add_knowledge_lesson(
            topic="test",
            correct_answer="i don't know the answer to that question",
            lesson_text="i don't know",
        ) == -1


# ===========================================================================
# Wrong Answer Fallback — Audit Fix 6 (from test_audit_fixes2)
# ===========================================================================

class TestWrongAnswerFallback:
    """Fix 6: Empty wrong_answer should fall back to previous_answer."""

    @pytest.mark.asyncio
    async def test_empty_wrong_answer_uses_previous(self):
        """When LLM omits wrong_answer, it should fall back to previous_answer."""
        import asyncio
        from unittest.mock import MagicMock

        mock_llm_response = '{"is_correction": true, "topic": "Python creator", "wrong_answer": "", "correct_answer": "Guido van Rossum", "lesson_text": "Python was created by Guido van Rossum"}'

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_llm_response)
            mock_llm.extract_json_object = MagicMock(return_value={
                "is_correction": True,
                "topic": "Python creator",
                "wrong_answer": "",
                "correct_answer": "Guido van Rossum",
                "lesson_text": "Python was created by Guido van Rossum",
            })

            engine = LearningEngine.__new__(LearningEngine)
            engine._db = MagicMock()
            engine._lessons_collection = None
            engine._training_lock = asyncio.Lock()

            correction = await engine.detect_correction(
                user_message="Actually, Python was created by Guido van Rossum",
                previous_answer="Python was created by James Gosling",
                original_query="Who created Python?",
            )

        assert correction is not None
        assert correction.wrong_answer == "Python was created by James Gosling"

    @pytest.mark.asyncio
    async def test_nonempty_wrong_answer_preserved(self):
        """When LLM provides wrong_answer, it should NOT be overwritten."""
        import asyncio
        from unittest.mock import MagicMock

        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{}')
            mock_llm.extract_json_object = MagicMock(return_value={
                "is_correction": True,
                "topic": "Capital of France",
                "wrong_answer": "London",
                "correct_answer": "Paris",
                "lesson_text": "The capital of France is Paris",
            })

            engine = LearningEngine.__new__(LearningEngine)
            engine._db = MagicMock()
            engine._lessons_collection = None
            engine._training_lock = asyncio.Lock()

            correction = await engine.detect_correction(
                user_message="No, the capital of France is Paris",
                previous_answer="The capital of France is London",
                original_query="What is the capital of France?",
            )

        assert correction is not None
        assert correction.wrong_answer == "London"


# ===========================================================================
# Reflexion Critique Facts (from test_audit_consolidated)
# ===========================================================================

class TestReflexionCritiqueFacts:

    def test_signature_accepts_facts(self):
        from app.core.reflexion import critique_response
        params = list(inspect.signature(critique_response).parameters.keys())
        assert "user_facts" in params
        assert "kg_facts" in params

    def test_facts_in_prompt(self):
        from app.core.reflexion import critique_response
        captured = []

        async def mock_invoke(messages, **kwargs):
            captured.append(messages[0]["content"])
            return '{"score": 0.9, "critique": "good"}'

        with patch("app.core.llm.invoke_nothink", side_effect=mock_invoke):
            with patch("app.core.llm.extract_json_object", return_value={"score": 0.9, "critique": "good"}):
                asyncio.get_event_loop().run_until_complete(
                    critique_response("q", "a", [], user_facts="name: X", kg_facts="X works_at Y")
                )

        assert captured
        assert "Owner facts" in captured[0]
        assert "Knowledge graph facts" in captured[0]

    def test_empty_facts_no_section(self):
        from app.core.reflexion import critique_response
        captured = []

        async def mock_invoke(messages, **kwargs):
            captured.append(messages[0]["content"])
            return '{"score": 0.8, "critique": "ok"}'

        with patch("app.core.llm.invoke_nothink", side_effect=mock_invoke):
            with patch("app.core.llm.extract_json_object", return_value={"score": 0.8, "critique": "ok"}):
                asyncio.get_event_loop().run_until_complete(
                    critique_response("q", "a", [])
                )

        assert "Owner facts" not in captured[0]


# ===========================================================================
# Should Use LLM Critique (from test_audit_consolidated)
# ===========================================================================

class TestShouldUseLLMCritique:

    def test_clean_tools_skip(self):
        from app.core.reflexion import should_use_llm_critique
        assert should_use_llm_critique("general", "x" * 600, [{"tool": "t", "output": "ok"}]) is False

    def test_failed_tools_trigger(self):
        from app.core.reflexion import should_use_llm_critique
        assert should_use_llm_critique("general", "short", [{"tool": "t", "output": "error: fail"}]) is True

    def test_short_toolless_skip(self):
        from app.core.reflexion import should_use_llm_critique
        assert should_use_llm_critique("general", "x" * 150, []) is False

    def test_long_toolless_trigger(self):
        from app.core.reflexion import should_use_llm_critique
        assert should_use_llm_critique("general", "x" * 600, []) is True

    def test_non_correction_triggers(self):
        from app.core.reflexion import should_use_llm_critique
        # All non-correction intents with long tool-less answers trigger critique
        assert should_use_llm_critique("greeting", "x" * 600, []) is True
        assert should_use_llm_critique("factual", "x" * 600, []) is True

    def test_correction_skip(self):
        from app.core.reflexion import should_use_llm_critique
        # Correction intent always skips LLM critique
        assert should_use_llm_critique("correction", "x" * 600, []) is False


# ===========================================================================
# Correction → Lesson → Better Answer (full loop)
# ===========================================================================

class TestCorrectionToLessonLoop:
    """Verify the full learning loop:
    1. User asks a question -> Nova gives wrong answer
    2. User corrects Nova
    3. Lesson is saved
    4. User asks the same question -> lesson is injected -> correct answer
    """

    @pytest.fixture
    def engine(self, db):
        return LearningEngine(db)

    def test_lesson_roundtrip(self, engine):
        """Save a correction as a lesson, then verify it's found for the same query."""
        correction = Correction(
            user_message="Actually, the telephone was invented by Antonio Meucci",
            previous_answer="The telephone was invented by Alexander Graham Bell",
            topic="Telephone inventor",
            correct_answer="Antonio Meucci invented the telephone",
            wrong_answer="Alexander Graham Bell",
            original_query="Who invented the telephone?",
            lesson_text="The telephone was invented by Antonio Meucci, not Alexander Graham Bell",
        )

        lesson_id = engine.save_lesson(correction)
        assert lesson_id > 0

        # Same query should find the lesson
        lessons = engine.get_relevant_lessons("Who invented the telephone?")
        assert len(lessons) >= 1
        assert any("Meucci" in (l.correct_answer or "") or "Meucci" in (l.lesson_text or "") for l in lessons)

    def test_lesson_injected_in_prompt(self, engine):
        """Verify that a saved lesson appears in the system prompt."""
        correction = Correction(
            user_message="No, Python was created by Guido",
            previous_answer="Python was created by James Gosling",
            topic="Python creator",
            correct_answer="Guido van Rossum",
            wrong_answer="James Gosling",
            lesson_text="Python was created by Guido van Rossum, not James Gosling",
        )
        engine.save_lesson(correction)

        lessons = engine.get_relevant_lessons("Who created Python?")
        lessons_text = format_lessons_for_prompt([
            {
                "topic": l.topic,
                "wrong_answer": l.wrong_answer or "",
                "correct_answer": l.correct_answer or "",
                "lesson_text": l.lesson_text or "",
            }
            for l in lessons
        ])

        prompt = build_system_prompt(lessons_text=lessons_text)
        assert "Guido" in prompt
        assert "Lessons From Past Corrections" in prompt

    def test_unrelated_query_doesnt_get_lesson(self, engine):
        """Lessons should NOT appear for unrelated queries."""
        correction = Correction(
            user_message="fix", previous_answer="bad",
            topic="Python creator",
            correct_answer="Guido van Rossum",
        )
        engine.save_lesson(correction)

        # Completely unrelated query
        lessons = engine.get_relevant_lessons("What's the mass of Jupiter?")
        assert len(lessons) == 0

    def test_lesson_helpful_confidence_grows(self, engine):
        """Lessons that are used and not corrected should gain confidence."""
        correction = Correction(
            user_message="fix", previous_answer="bad",
            topic="test confidence",
            correct_answer="right answer",
        )
        lesson_id = engine.save_lesson(correction)

        # Mark as helpful 5 times
        for _ in range(5):
            engine.mark_lesson_helpful(lesson_id)

        row = engine._db.fetchone(
            "SELECT confidence, times_helpful FROM lessons WHERE id = ?",
            (lesson_id,),
        )
        assert row["times_helpful"] == 5
        assert row["confidence"] > 0.8  # Started at 0.8, should have grown

    @pytest.mark.asyncio
    async def test_training_pair_has_correct_format(self, engine, tmp_path):
        """DPO training pair should have query=original, chosen=good, rejected=bad."""
        training_path = str(tmp_path / "training.jsonl")

        with patch("app.core.learning.config") as mock_config:
            mock_config.TRAINING_DATA_PATH = training_path

            await engine.save_training_pair(
                query="Who invented the telephone?",
                bad_answer="Alexander Graham Bell invented the telephone.",
                good_answer="Antonio Meucci invented the telephone.",
            )

        with open(training_path) as f:
            entry = json.loads(f.readline())

        assert entry["query"] == "Who invented the telephone?"
        assert entry["chosen"] == "Antonio Meucci invented the telephone."
        assert entry["rejected"] == "Alexander Graham Bell invented the telephone."


# ===========================================================================
# Fallback Correction Extraction
# ===========================================================================

class TestFallbackExtraction:
    def test_extract_actually(self):
        result = _extract_answer_from_message("Actually, Python was made by Guido")
        assert "Guido" in result

    def test_extract_correct_answer_is(self):
        result = _extract_answer_from_message("The correct answer is 42")
        assert "42" in result

    def test_extract_remember_that(self):
        result = _extract_answer_from_message("Remember that I prefer dark mode")
        assert "dark mode" in result

    def test_no_pattern_returns_none(self):
        result = _extract_answer_from_message("Use Vue instead")
        assert result is None

    def test_fallback_correction_object(self):
        c = _fallback_correction(
            "Actually, it's Canberra",
            "The capital of Australia is Sydney",
            "What's the capital of Australia?",
        )
        assert c.topic == "general"
        assert "Canberra" in c.correct_answer
        assert c.original_query == "What's the capital of Australia?"


# ===========================================================================
# Additional Correction Patterns (unique from behavioral tests)
# ===========================================================================

class TestCorrectionPatternsExtended:
    """Additional correction detection patterns not covered by TestCorrectionDetection."""

    def test_instead_of(self):
        # "instead of" now requires preceding assistant reference to reduce false positives
        assert is_likely_correction("You recommended React instead of Vue")
        assert is_likely_correction("The answer said X instead of Y")
        # Bare "instead of" without assistant context no longer triggers
        assert not is_likely_correction("Instead of React, use Vue")

    def test_you_should_use(self):
        assert is_likely_correction("You should use CoinGecko for crypto prices")

    def test_next_time(self):
        assert is_likely_correction("Next time, search for the latest data")

    def test_dont_use(self):
        assert is_likely_correction("Don't use that approach, it's outdated")

    def test_remember_that(self):
        assert is_likely_correction("Remember that I prefer dark mode")

    def test_from_now_on(self):
        assert is_likely_correction("From now on, always use metric units")

    def test_it_should_be(self):
        assert is_likely_correction("It should be Python 3.12, not 3.11")

    def test_statement_not_correction(self):
        assert not is_likely_correction("The sky is blue")

    def test_request_not_correction(self):
        assert not is_likely_correction("Can you help me write a function?")

    def test_data_incomplete_matches(self):
        """'That data is incomplete' should trigger correction detection."""
        assert is_likely_correction("That data is incomplete")

    def test_you_missed_matches(self):
        """'You missed the latest update' should trigger correction detection."""
        assert is_likely_correction("You missed the latest update")

    def test_missing_dog_no_match(self):
        """'I'm missing my dog' should NOT trigger correction detection."""
        assert not is_likely_correction("I'm missing my dog")

    def test_missing_piece_no_match(self):
        """'I found the missing piece' should NOT trigger correction detection."""
        assert not is_likely_correction("I found the missing piece")


# ===========================================================================
# Reflexion Browser Selector Scoring
# ===========================================================================

class TestReflexionBrowserScoring:
    """Verify reflexion scoring treats browser selector misses softly."""

    def test_browser_selector_miss_mild_penalty(self):
        from app.core.reflexion import assess_quality

        tool_results = [
            {"output": "", "error": "Selector 'button.fake' not found. Available elements:\n..."},
        ]
        score, reason = assess_quality("I clicked the button", tool_results, 5)
        # Browser selector miss = -0.05, not -0.15
        assert score >= 0.9, f"Score too low for browser selector miss: {score}"
        assert "selector miss" in reason.lower()

    def test_hard_tool_failure_still_penalized(self):
        from app.core.reflexion import assess_quality

        tool_results = [
            {"output": "Connection failed: timeout", "error": ""},
        ]
        score, reason = assess_quality("Here is the result", tool_results, 5)
        assert score <= 0.85, f"Hard failure not penalized enough: {score}"

    def test_all_tools_clean_ignores_selector_miss(self):
        from app.core.reflexion import _all_tools_clean

        tool_results = [
            {"output": "", "error": "Selector '#missing' not found"},
        ]
        assert _all_tools_clean(tool_results), "Browser selector miss should not count as dirty"

    def test_all_tools_clean_catches_real_failure(self):
        from app.core.reflexion import _all_tools_clean

        tool_results = [
            {"output": "HTTP request failed with error 500"},
        ]
        assert not _all_tools_clean(tool_results), "Real failure should be caught"


# ===========================================================================
# Autonomous Skill Creation (from test_auto_skills)
# ===========================================================================

class TestAutoSkillCreation:
    @pytest.fixture
    def mock_skills(self, db):
        return SkillStore(db)

    @pytest.mark.asyncio
    async def test_auto_skill_skips_when_disabled(self, mock_skills):
        from app.core.auto_skills import maybe_extract_skill
        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": False})()):
            await maybe_extract_skill(
                "test query",
                [{"tool": "web_search", "args": {}}, {"tool": "calculator", "args": {}}],
                "answer",
                mock_skills,
            )
        assert len(mock_skills.get_all_skills()) == 0

    @pytest.mark.asyncio
    async def test_auto_skill_skips_single_tool(self, mock_skills):
        from app.core.auto_skills import maybe_extract_skill
        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True})()):
            await maybe_extract_skill(
                "test query",
                [{"tool": "web_search", "args": {}}],
                "answer",
                mock_skills,
            )
        assert len(mock_skills.get_all_skills()) == 0

    @pytest.mark.asyncio
    async def test_auto_skill_skips_delegate(self, mock_skills):
        from app.core.auto_skills import maybe_extract_skill
        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True})()):
            await maybe_extract_skill(
                "compare weather",
                [{"tool": "delegate", "args": {}}, {"tool": "delegate", "args": {}}],
                "answer",
                mock_skills,
            )
        assert len(mock_skills.get_all_skills()) == 0

    @pytest.mark.asyncio
    async def test_auto_skill_creates_on_valid_extraction(self, mock_skills):
        from app.core.auto_skills import maybe_extract_skill
        llm_response = json.dumps({
            "name": "price_compare",
            "trigger_pattern": r"(?i)compare.*price",
            "steps": [
                {"tool": "web_search", "args_template": {"query": "{query} price"}, "output_key": "result"},
            ],
            "answer_template": "Based on search: {result}",
        })

        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True, "INTERNAL_LLM_TIMEOUT": 30})()):
            with patch("app.core.auto_skills.llm") as mock_llm:
                mock_llm.invoke_nothink = AsyncMock(return_value=llm_response)
                mock_llm.extract_json_object = MagicMock(return_value=json.loads(llm_response))

                await maybe_extract_skill(
                    "compare prices of iPhone and Samsung",
                    [
                        {"tool": "web_search", "args": {"query": "iPhone price"}},
                        {"tool": "web_search", "args": {"query": "Samsung price"}},
                    ],
                    "iPhone costs X, Samsung costs Y",
                    mock_skills,
                )

        skills = mock_skills.get_all_skills()
        assert len(skills) == 1
        assert skills[0].name == "price_compare"

    @pytest.mark.asyncio
    async def test_auto_skill_rejects_broad_pattern(self, mock_skills):
        from app.core.auto_skills import maybe_extract_skill
        llm_response = json.dumps({
            "name": "broad_skill",
            "trigger_pattern": r".*",
            "steps": [{"tool": "web_search", "args_template": {"query": "{query}"}}],
        })

        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True, "INTERNAL_LLM_TIMEOUT": 30})()):
            with patch("app.core.auto_skills.llm") as mock_llm:
                mock_llm.invoke_nothink = AsyncMock(return_value=llm_response)
                mock_llm.extract_json_object = MagicMock(return_value=json.loads(llm_response))

                await maybe_extract_skill(
                    "query",
                    [{"tool": "web_search", "args": {}}, {"tool": "calculator", "args": {}}],
                    "answer",
                    mock_skills,
                )

        assert len(mock_skills.get_all_skills()) == 0

    @pytest.mark.asyncio
    async def test_auto_skill_handles_llm_skip(self, mock_skills):
        from app.core.auto_skills import maybe_extract_skill
        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True, "INTERNAL_LLM_TIMEOUT": 30})()):
            with patch("app.core.auto_skills.llm") as mock_llm:
                mock_llm.invoke_nothink = AsyncMock(return_value='{"skip": true}')
                mock_llm.extract_json_object = MagicMock(return_value={"skip": True})

                await maybe_extract_skill(
                    "query",
                    [{"tool": "web_search", "args": {}}, {"tool": "calculator", "args": {}}],
                    "answer",
                    mock_skills,
                )

        assert len(mock_skills.get_all_skills()) == 0
