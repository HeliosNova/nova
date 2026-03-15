"""Behavioral anchor tests — end-to-end verification of Nova's learning loop.

These tests verify the full pipeline:
  correction → lesson → better answer on same topic
  conversation → fact extraction → facts in prompt
  skill creation → skill firing on matching query
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.brain import Services, set_services, think
from app.core.learning import (
    Correction,
    LearningEngine,
    is_likely_correction,
    _extract_answer_from_message,
    _fallback_correction,
)
from app.core.memory import (
    ConversationStore,
    UserFactStore,
    has_fact_signals,
    extract_facts_from_message,
)
from app.core.prompt import build_system_prompt, format_lessons_for_prompt
from app.core.skills import SkillStore
from app.schema import EventType


# ===========================================================================
# Fact Signal Detection (regex pre-filter)
# ===========================================================================

class TestFactSignals:
    def test_detects_my_name(self):
        assert has_fact_signals("My name is Alex")

    def test_detects_i_work(self):
        assert has_fact_signals("I work at Google as an engineer")

    def test_detects_i_prefer(self):
        assert has_fact_signals("I prefer Python for backend work")

    def test_detects_im_from(self):
        assert has_fact_signals("I'm from Tokyo")

    def test_detects_i_live(self):
        assert has_fact_signals("I live in San Francisco")

    def test_detects_call_me(self):
        assert has_fact_signals("Call me Alex")

    def test_detects_i_use(self):
        assert has_fact_signals("I use Vim for editing")

    def test_detects_my_favorite(self):
        assert has_fact_signals("My favorite language is Rust")

    def test_detects_i_dont_like(self):
        assert has_fact_signals("I don't like Java")

    def test_detects_remember_my(self):
        assert has_fact_signals("Remember that I always use tabs")

    def test_no_signal_generic_question(self):
        assert not has_fact_signals("What's the weather today?")

    def test_no_signal_greeting(self):
        assert not has_fact_signals("Hello, how are you?")

    def test_no_signal_factual(self):
        assert not has_fact_signals("Who invented the telephone?")


# ===========================================================================
# Fact Extraction (LLM-based, mocked)
# ===========================================================================

class TestFactExtraction:
    @pytest.mark.asyncio
    async def test_extracts_name_and_job(self):
        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({
                "name": "Alex",
                "employer": "Google",
                "job_title": "Senior Engineer",
            }))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            facts = await extract_facts_from_message(
                "My name is Alex and I work at Google as a senior engineer"
            )
            assert facts["name"]["value"] == "Alex"
            assert facts["employer"]["value"] == "Google"
            assert facts["job_title"]["value"] == "Senior Engineer"

    @pytest.mark.asyncio
    async def test_extracts_preferences(self):
        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({
                "preferred_language": "Python",
                "preferred_framework": "Vue",
            }))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            facts = await extract_facts_from_message(
                "I prefer Python for backend and Vue for frontend"
            )
            assert "preferred_language" in facts
            assert facts["preferred_language"]["value"] == "Python"

    @pytest.mark.asyncio
    async def test_empty_for_no_facts(self):
        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="{}")
            mock_llm.extract_json_object = lambda x: json.loads(x)

            facts = await extract_facts_from_message("What's the weather?")
            assert facts == {}

    @pytest.mark.asyncio
    async def test_rejects_garbage_keys(self):
        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({
                "x": "too short key",
                "INVALID KEY WITH SPACES": "value",
                "valid_key": "good value",
            }))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            facts = await extract_facts_from_message("I am a developer")
            assert "valid_key" in facts
            assert "x" not in facts
            assert "INVALID KEY WITH SPACES" not in facts

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self):
        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=Exception("LLM down"))

            facts = await extract_facts_from_message("My name is Alex")
            assert facts == {}


# ===========================================================================
# Behavioral Anchor: Correction → Lesson → Better Answer
# ===========================================================================

class TestCorrectionToLessonLoop:
    """Verify the full learning loop:
    1. User asks a question → Nova gives wrong answer
    2. User corrects Nova
    3. Lesson is saved
    4. User asks the same question → lesson is injected → correct answer
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
# Behavioral Anchor: Skills Creation + Firing
# ===========================================================================

class TestSkillBehavior:
    @pytest.fixture
    def store(self, db):
        return SkillStore(db)

    def test_skill_matches_and_fires(self, store):
        """A created skill should match queries fitting its trigger pattern."""
        store.create_skill(
            name="crypto_price_check",
            trigger_pattern=r"(?i)(?:price|value)\s+of\s+\w+",
            steps=[
                {"tool": "web_search", "args_template": {"query": "current price of {entity}"}},
            ],
            answer_template="The current price of {entity} is {result}.",
        )

        skill = store.get_matching_skill("What's the price of Bitcoin?")
        assert skill is not None
        assert skill.name == "crypto_price_check"
        assert len(skill.steps) == 1
        assert skill.steps[0]["tool"] == "web_search"

    def test_skill_doesnt_match_wrong_query(self, store):
        """Skills should not fire on non-matching queries."""
        store.create_skill(
            name="crypto_price_check",
            trigger_pattern=r"(?i)price\s+of\s+\w+",
            steps=[{"tool": "web_search", "args_template": {"query": "price"}}],
        )

        skill = store.get_matching_skill("How do I make pasta?")
        assert skill is None

    def test_disabled_skill_doesnt_match(self, store):
        """Disabled skills should not fire."""
        skill_id = store.create_skill(
            name="test_skill",
            trigger_pattern=r"(?i)test",
            steps=[],
        )
        store.toggle_skill(skill_id, False)

        skill = store.get_matching_skill("this is a test")
        assert skill is None

    def test_skill_success_rate_tracking(self, store):
        """Skills should track success rate via EMA (alpha=0.15)."""
        skill_id = store.create_skill("tracked", r"test", [])

        # 3 successes, 1 failure
        store.record_use(skill_id, True)
        store.record_use(skill_id, True)
        store.record_use(skill_id, True)
        store.record_use(skill_id, False)

        skill = store.get_skill(skill_id)
        assert skill.times_used == 4
        # EMA (alpha=0.15) from initial 0.7:
        # 0.7 -> 0.745 -> 0.78325 -> 0.8157625 -> 0.69339813 (failure)
        assert abs(skill.success_rate - 0.6934) < 0.01


# ===========================================================================
# Behavioral Anchor: User Facts in Prompt
# ===========================================================================

class TestFactsInPrompt:
    def test_facts_appear_in_system_prompt(self, db):
        """User facts should be injected into Block 2 of the system prompt."""
        store = UserFactStore(db)
        store.set("name", "Alex", source="user_stated")
        store.set("preferred_language", "Python", source="extracted")

        facts_text = store.format_for_prompt()
        prompt = build_system_prompt(user_facts_text=facts_text)

        assert "Alex" in prompt
        assert "Python" in prompt
        assert "What You Know About Your Owner" in prompt

    def test_facts_not_truncated(self, db):
        """User facts (Block 2) should never be truncated, even with huge context."""
        store = UserFactStore(db)
        store.set("name", "Alex")
        store.set("city", "Tokyo")

        facts_text = store.format_for_prompt()
        big_context = "x" * 50000

        prompt = build_system_prompt(
            user_facts_text=facts_text,
            retrieved_context=big_context,
        )

        # Facts must survive
        assert "Alex" in prompt
        assert "Tokyo" in prompt
        # Context should be truncated
        assert "truncated" in prompt.lower()


# ===========================================================================
# Behavioral Anchor: Fact Extraction Wired to Brain
# ===========================================================================

class TestFactExtractionInBrain:
    @pytest.fixture
    def services(self, db):
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_fact_extracted_during_think(self, services):
        """When user says 'My name is Alex', a fact should be extracted."""
        with patch("app.core.brain.llm") as mock_brain_llm, \
             patch("app.core.memory.llm") as mock_mem_llm:

            # Brain LLM: normal response
            mock_result = AsyncMock()
            mock_result.content = "Nice to meet you, Alex!"
            mock_result.tool_call = None
            mock_brain_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_brain_llm.invoke_nothink = AsyncMock(return_value="Introduction")

            # Memory LLM: fact extraction
            mock_mem_llm.invoke_nothink = AsyncMock(return_value=json.dumps({
                "name": "Alex",
            }))
            mock_mem_llm.extract_json_object = lambda x: json.loads(x)

            async for event in think("My name is Alex"):
                pass

        # Fact should have been saved
        fact = services.user_facts.get("name")
        assert fact is not None
        assert fact.value == "Alex"
        assert fact.source == "extracted"

    @pytest.mark.asyncio
    async def test_no_extraction_for_generic_message(self, services):
        """Generic messages without fact signals should NOT trigger extraction."""
        with patch("app.core.brain.llm") as mock_brain_llm:
            mock_result = AsyncMock()
            mock_result.content = "It's sunny today."
            mock_result.tool_call = None
            mock_brain_llm.generate_with_tools = AsyncMock(return_value=mock_result)
            mock_brain_llm.invoke_nothink = AsyncMock(return_value="Weather")

            async for event in think("What's the weather?"):
                pass

        # No facts should exist
        facts = services.user_facts.get_all()
        assert len(facts) == 0


# ===========================================================================
# Correction Pattern Coverage
# ===========================================================================

class TestCorrectionPatterns:
    """Ensure correction detection catches various phrasing styles."""

    def test_actually(self):
        assert is_likely_correction("Actually, it's Canberra")

    def test_thats_wrong(self):
        assert is_likely_correction("That's wrong, the answer is 42")

    def test_youre_wrong(self):
        assert is_likely_correction("You're wrong about that")

    def test_not_quite(self):
        assert is_likely_correction("Not quite, it should be X")

    def test_instead_of(self):
        assert is_likely_correction("Instead of React, use Vue")

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

    def test_the_correct_answer_is(self):
        assert is_likely_correction("The correct answer is 42")

    def test_it_should_be(self):
        assert is_likely_correction("It should be Python 3.12, not 3.11")

    # Negatives
    def test_question_not_correction(self):
        assert not is_likely_correction("What's the capital of France?")

    def test_statement_not_correction(self):
        assert not is_likely_correction("The sky is blue")

    def test_request_not_correction(self):
        assert not is_likely_correction("Can you help me write a function?")

    # Item 4: Tightened correction patterns — true positives
    def test_data_incomplete_matches(self):
        """'That data is incomplete' should trigger correction detection."""
        assert is_likely_correction("That data is incomplete")

    def test_you_missed_matches(self):
        """'You missed the latest update' should trigger correction detection."""
        assert is_likely_correction("You missed the latest update")

    # Item 4: Tightened correction patterns — true negatives
    def test_missing_dog_no_match(self):
        """'I'm missing my dog' should NOT trigger correction detection."""
        assert not is_likely_correction("I'm missing my dog")

    def test_missing_piece_no_match(self):
        """'I found the missing piece' should NOT trigger correction detection."""
        assert not is_likely_correction("I found the missing piece")


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

    def test_fallback_whole_message(self):
        result = _extract_answer_from_message("Use Vue instead")
        assert "Vue" in result

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
# Conversation Search
# ===========================================================================

class TestConversationSearch:
    def test_search_messages(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation("Python Chat")
        store.add_message(conv_id, "user", "How do I use Python decorators?")
        store.add_message(conv_id, "assistant", "Decorators in Python use the @ syntax.")

        results = store.search_messages("Python decorators")
        assert len(results) >= 1
        assert any("decorator" in r["content"].lower() for r in results)

    def test_search_conversations(self, db):
        store = ConversationStore(db)
        cid1 = store.create_conversation("Python Chat")
        store.add_message(cid1, "user", "Tell me about Python")
        cid2 = store.create_conversation("Weather Chat")
        store.add_message(cid2, "user", "Tell me about the weather")

        results = store.search_conversations("Python")
        assert len(results) >= 1
        assert results[0]["conversation_id"] == cid1

    def test_search_no_results(self, db):
        store = ConversationStore(db)
        conv_id = store.create_conversation("Test")
        store.add_message(conv_id, "user", "Hello world")

        results = store.search_messages("quantum physics")
        assert len(results) == 0
