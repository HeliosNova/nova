"""Core internal tests — tool exception handling, JSON logging, correction bounds,
lesson dedup threshold, batch retrieval, decay improvement, training rotation,
summarization fallback, skill recovery.

Consolidated from B1-B4 bug fixes + R1-R3, R6, R8 robustness items.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


# ---------------------------------------------------------------------------
# B1: Tool execution exception handler
# ---------------------------------------------------------------------------

class TestToolExceptionHandler:
    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_string(self, db):
        """When a tool raises, _execute_tool should return (error_str, ToolResult)."""
        from app.core.brain import _execute_tool

        mock_registry = MagicMock()
        mock_registry.execute_full = AsyncMock(side_effect=RuntimeError("tool broke"))
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            tool_registry=mock_registry,
        )
        set_services(svc)

        output, tool_result = await _execute_tool("bad_tool", {"arg": "val"})
        assert "failed" in output.lower()
        assert "tool broke" in output
        assert tool_result is not None
        assert not tool_result.success


# ---------------------------------------------------------------------------
# B2: JSON extraction logging
# ---------------------------------------------------------------------------

class TestJSONExtractionLogging:
    def test_empty_input_logs_debug(self, caplog):
        from app.core.llm import extract_json_object
        with caplog.at_level(logging.DEBUG, logger="app.core.llm"):
            result = extract_json_object("")
        assert result == {}
        assert any("empty input" in r.message for r in caplog.records)

    def test_no_json_logs_debug(self, caplog):
        from app.core.llm import extract_json_object
        with caplog.at_level(logging.DEBUG, logger="app.core.llm"):
            result = extract_json_object("just plain text no json here")
        assert result == {}
        assert any("no balanced JSON" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# B3: Correction handler bounds check
# ---------------------------------------------------------------------------

class TestCorrectionBoundsCheck:
    @pytest.mark.asyncio
    async def test_correction_no_prior_messages(self, db, caplog):
        """Correction with no prior messages should log and not crash."""
        from app.core.brain import think
        from app.core.learning import LearningEngine

        db.init_schema()
        learning = LearningEngine(db)
        conversations = ConversationStore(db)
        user_facts = UserFactStore(db)

        svc = Services(
            conversations=conversations,
            user_facts=user_facts,
            learning=learning,
        )
        set_services(svc)

        conv_id = conversations.create_conversation()

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.generate_with_tools = AsyncMock(return_value=MagicMock(
                content="I understand the correction.",
                tool_calls=[],
                raw={},
            ))
            mock_llm.invoke_nothink = AsyncMock(return_value="Correction Note")

            with caplog.at_level(logging.INFO, logger="app.core.brain"):
                events = []
                async for event in think(
                    query="Actually, the answer is 42",
                    conversation_id=conv_id,
                ):
                    events.append(event)

        assert any(e.type.value == "done" for e in events)


# ---------------------------------------------------------------------------
# B4: Skill auto-disable recovery
# ---------------------------------------------------------------------------

class TestSkillRecovery:
    def test_reenable_resets_stats(self, db):
        from app.core.skills import SkillStore

        db.init_schema()
        store = SkillStore(db)

        skill_id = db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, times_used, success_rate, enabled)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("test_skill", r"\btest\b", "[]", 10, 0.2, 0),
        ).lastrowid

        store.toggle_skill(skill_id, enabled=True)

        skill = store.get_skill(skill_id)
        assert skill.enabled is True
        assert skill.times_used == 0
        assert skill.success_rate == 0.7

    def test_disable_preserves_stats(self, db):
        from app.core.skills import SkillStore

        db.init_schema()
        store = SkillStore(db)

        skill_id = db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, times_used, success_rate, enabled)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("test_skill", r"\btest\b", "[]", 10, 0.5, 1),
        ).lastrowid

        store.toggle_skill(skill_id, enabled=False)

        skill = store.get_skill(skill_id)
        assert skill.enabled is False
        assert skill.times_used == 10
        assert skill.success_rate == 0.5


# ---------------------------------------------------------------------------
# R1: Lesson dedup threshold 0.7→0.85
# ---------------------------------------------------------------------------

class TestLessonDedupThreshold:
    def test_threshold_is_085(self, db):
        """Lessons with ~85% Jaccard overlap should be deduplicated."""
        from app.core.learning import LearningEngine

        db.init_schema()
        engine = LearningEngine(db)

        # Insert an initial lesson
        engine.add_knowledge_lesson(
            topic="python data structures",
            correct_answer="python lists are mutable ordered collections that support indexing",
            lesson_text="Lists are mutable in Python",
        )

        # Insert a near-duplicate (~85%+ overlap) — should be deduplicated
        engine.add_knowledge_lesson(
            topic="python data structures",
            correct_answer="python lists are mutable ordered collections that support indexing and slicing",
            lesson_text="Lists are mutable in Python",
        )

        # Should have only 1 lesson (second was deduped)
        all_lessons = engine.get_all_lessons(limit=10)
        assert len(all_lessons) == 1, f"Expected dedup to merge similar lessons, got {len(all_lessons)}"

        # Insert a clearly different lesson — should NOT be deduped
        engine.add_knowledge_lesson(
            topic="python data structures",
            correct_answer="dictionaries preserve insertion order since python 3.7",
            lesson_text="Dict ordering in Python",
        )
        all_lessons = engine.get_all_lessons(limit=10)
        assert len(all_lessons) == 2, f"Expected distinct lesson to be kept, got {len(all_lessons)}"


# ---------------------------------------------------------------------------
# R2: Lesson retrieval N+1 batch
# ---------------------------------------------------------------------------

class TestLessonRetrievalBatch:
    def test_batch_update_times_retrieved(self, db):
        from app.core.learning import LearningEngine

        db.init_schema()
        engine = LearningEngine(db)

        for topic in ["Python creator", "Python version", "Python typing"]:
            db.execute(
                """INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, context, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (topic, "wrong", "correct", f"lesson about {topic}", "python", 0.8),
            )

        lessons = engine.get_relevant_lessons("Python")
        assert len(lessons) == 3

        for lesson in lessons:
            row = db.fetchone("SELECT times_retrieved FROM lessons WHERE id = ?", (lesson.id,))
            assert row["times_retrieved"] == 1


# ---------------------------------------------------------------------------
# R3: Lesson decay improvement (last_retrieved_at)
# ---------------------------------------------------------------------------

class TestLessonDecayImproved:
    def test_old_retrieved_lesson_decays(self, db):
        from app.core.learning import LearningEngine

        db.init_schema()
        engine = LearningEngine(db)

        db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, context, confidence,
               times_retrieved, last_retrieved_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-90 days'), datetime('now', '-90 days'))""",
            ("old topic", "wrong", "correct", "context", 0.8, 5),
        )

        decayed = engine.decay_stale_lessons(days=30)
        assert decayed == 1

        row = db.fetchone("SELECT confidence FROM lessons WHERE topic = 'old topic'")
        assert row["confidence"] < 0.8

    def test_recently_retrieved_no_decay(self, db):
        from app.core.learning import LearningEngine

        db.init_schema()
        engine = LearningEngine(db)

        db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, context, confidence,
               times_retrieved, last_retrieved_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            ("fresh topic", "wrong", "correct", "context", 0.8, 5),
        )

        decayed = engine.decay_stale_lessons(days=30)
        assert decayed == 0


# ---------------------------------------------------------------------------
# R6: Training data rotation fix
# ---------------------------------------------------------------------------

class TestTrainingDataRotation:
    def test_rotation_keeps_max(self, db, tmp_path):
        from app.core.learning import _rotate_training_data

        path = tmp_path / "training.jsonl"
        with open(path, "w") as f:
            for i in range(15):
                f.write(json.dumps({"query": f"q{i}", "chosen": f"a{i}", "rejected": f"r{i}"}) + "\n")

        with patch("app.core.learning.config") as mock_config:
            mock_config.MAX_TRAINING_PAIRS = 10
            _rotate_training_data(path)

        with open(path) as f:
            remaining = f.readlines()
        assert len(remaining) == 10


# ---------------------------------------------------------------------------
# R8: Summarization failure fallback
# ---------------------------------------------------------------------------

class TestSummarizationFallback:
    @pytest.mark.asyncio
    async def test_failure_returns_truncation_note(self):
        from app.core.brain import _manage_context

        system_prompt = "x" * 4000
        history = [{"role": "user", "content": f"message {i}" * 50} for i in range(20)]
        query = "test query"

        with patch("app.core.brain.config") as mock_config:
            mock_config.MAX_CONTEXT_TOKENS = 100
            mock_config.RECENT_MESSAGES_KEEP = 4
            mock_config.RESPONSE_TOKEN_BUDGET = 600

            with patch("app.core.brain.llm") as mock_llm:
                mock_llm.invoke_nothink = AsyncMock(side_effect=Exception("LLM down"))

                trimmed, summary = await _manage_context(system_prompt, history, query)

        assert len(trimmed) == 4
        assert "truncated" in summary.lower()
        assert "older messages" in summary.lower()
