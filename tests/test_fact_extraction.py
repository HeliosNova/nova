"""Item 60: Test the fact extraction pipeline in app/core/memory.py.

Send messages with explicit fact signals, mock the LLM to return extracted
facts, and verify they're properly stored.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core.memory import (
    ConversationStore,
    UserFactStore,
    extract_facts_from_message,
    has_fact_signals,
)


class TestHasFactSignals:
    """Test the regex pre-filter for fact-bearing messages."""

    def test_name_detected(self):
        assert has_fact_signals("My name is John")

    def test_employer_detected(self):
        assert has_fact_signals("I work at Google as an engineer")

    def test_preference_detected(self):
        assert has_fact_signals("I prefer dark mode")

    def test_living_detected(self):
        assert has_fact_signals("I live in Tokyo")

    def test_generic_question_not_detected(self):
        assert not has_fact_signals("What is the weather today?")

    def test_calculation_not_detected(self):
        assert not has_fact_signals("Calculate 2 + 2")

    def test_always_instruction_detected(self):
        assert has_fact_signals("Always include code examples")

    def test_remember_detected(self):
        assert has_fact_signals("Remember that I use Python")


class TestExtractFactsFromMessage:
    """Test the LLM-based fact extraction."""

    @pytest.mark.asyncio
    async def test_extracts_name_fact(self, db):
        """LLM returns a name fact -> should be returned."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"name": {"value": "John", "category": "fact"}}')
            mock_llm.extract_json_object = lambda x: {"name": {"value": "John", "category": "fact"}}

            facts = await extract_facts_from_message(
                "My name is John",
                "Nice to meet you, John!",
                fact_store=fact_store,
            )
            assert "name" in facts
            assert facts["name"]["value"] == "John"
            assert facts["name"]["category"] == "fact"

    @pytest.mark.asyncio
    async def test_extracts_multiple_facts(self, db):
        """LLM returns multiple facts -> all should be returned."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"name": {"value": "John", "category": "fact"}, "employer": {"value": "Google", "category": "fact"}}')
            mock_llm.extract_json_object = lambda x: {
                "name": {"value": "John", "category": "fact"},
                "employer": {"value": "Google", "category": "fact"},
            }

            facts = await extract_facts_from_message(
                "My name is John and I work at Google",
                fact_store=fact_store,
            )
            assert len(facts) == 2
            assert "name" in facts
            assert "employer" in facts

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_empty(self, db):
        """LLM returns empty dict -> should return empty."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{}')
            mock_llm.extract_json_object = lambda x: {}

            facts = await extract_facts_from_message(
                "What is the weather?",
                fact_store=fact_store,
            )
            assert facts == {}

    @pytest.mark.asyncio
    async def test_filters_meta_keys(self, db):
        """Keys like 'error', 'message' should be filtered."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"error": "blah", "name": {"value": "John", "category": "fact"}}')
            mock_llm.extract_json_object = lambda x: {
                "error": "blah",
                "name": {"value": "John", "category": "fact"},
            }

            facts = await extract_facts_from_message(
                "My name is John",
                fact_store=fact_store,
            )
            assert "error" not in facts
            assert "name" in facts

    @pytest.mark.asyncio
    async def test_filters_error_value_phrases(self, db):
        """Facts with error-like values should be filtered."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"location": {"value": "unable to determine", "category": "fact"}}')
            mock_llm.extract_json_object = lambda x: {
                "location": {"value": "unable to determine", "category": "fact"},
            }

            facts = await extract_facts_from_message(
                "I live somewhere",
                fact_store=fact_store,
            )
            assert "location" not in facts

    @pytest.mark.asyncio
    async def test_instruction_downgraded_without_permanence(self, db):
        """Instruction category should be downgraded to fact without permanence signal."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"pref_format": {"value": "use bullet points", "category": "instruction"}}')
            mock_llm.extract_json_object = lambda x: {
                "pref_format": {"value": "use bullet points", "category": "instruction"},
            }

            facts = await extract_facts_from_message(
                "I like bullet points",  # no "always"/"never"
                fact_store=fact_store,
            )
            assert facts.get("pref_format", {}).get("category") == "fact"

    @pytest.mark.asyncio
    async def test_instruction_kept_with_permanence(self, db):
        """Instruction category should be kept with permanence signal."""
        fact_store = UserFactStore(db)

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"pref_format": {"value": "Always use bullet points", "category": "instruction"}}')
            mock_llm.extract_json_object = lambda x: {
                "pref_format": {"value": "Always use bullet points", "category": "instruction"},
            }

            facts = await extract_facts_from_message(
                "Always use bullet points",  # permanence signal
                fact_store=fact_store,
            )
            assert facts.get("pref_format", {}).get("category") == "instruction"

    @pytest.mark.asyncio
    async def test_max_facts_cap_skips_extraction(self, db):
        """When already at MAX_USER_FACTS, extraction should be skipped."""
        fact_store = UserFactStore(db)

        # Fill up to max
        from app.config import config
        for i in range(config.MAX_USER_FACTS):
            fact_store.set(f"fact_{i}", f"value_{i}", source="extracted")

        with patch("app.core.memory.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"new_fact": {"value": "test", "category": "fact"}}')

            facts = await extract_facts_from_message(
                "My name is John",
                fact_store=fact_store,
            )
            assert facts == {}
            # LLM should not have been called
            mock_llm.invoke_nothink.assert_not_called()
