"""Tests for Phase 8.1: Auto-skill extraction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.auto_skills import maybe_extract_skill
from app.core.skills import SkillStore


# ===========================================================================
# Helpers
# ===========================================================================

def _make_tool_results(count: int = 2) -> list[dict]:
    """Create N mock tool result dicts."""
    return [
        {"tool": "web_search", "args": {"query": f"search {i}"}, "output": f"Result {i}"}
        for i in range(count)
    ]


def _valid_llm_response() -> dict:
    """A well-formed skill extraction response."""
    return {
        "name": "stock_price_lookup",
        "trigger_pattern": r"(?:what(?:'s| is) the )?(?:stock )?price of (\w+)",
        "steps": [
            {"tool": "web_search", "args_template": {"query": "stock price {query}"}, "output_key": "search_result"},
            {"tool": "calculator", "args_template": {"expression": "{search_result}"}, "output_key": "price"},
        ],
        "answer_template": "The stock price is {price}.",
    }


# ===========================================================================
# Threshold: 2+ tool results required
# ===========================================================================

class TestSkillExtractionThreshold:
    @pytest.mark.asyncio
    async def test_skips_with_zero_tool_results(self, db, monkeypatch):
        """No extraction when tool_results is empty."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm:
            await maybe_extract_skill("hello", [], "answer", skills)
            mock_llm.invoke_nothink.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_with_one_tool_result(self, db, monkeypatch):
        """No extraction when only 1 tool result."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm:
            await maybe_extract_skill("hello", _make_tool_results(1), "answer", skills)
            mock_llm.invoke_nothink.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_with_two_tool_results(self, db, monkeypatch):
        """Extraction runs when 2+ tool results."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({"skip": True}))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)
            mock_llm.invoke_nothink.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, db, monkeypatch):
        """No extraction when ENABLE_AUTO_SKILL_CREATION is false."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "false")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm:
            await maybe_extract_skill("hello", _make_tool_results(3), "answer", skills)
            mock_llm.invoke_nothink.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_delegate_interactions(self, db, monkeypatch):
        """No extraction when tool results include delegate (sub-agent)."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        tool_results = [
            {"tool": "web_search", "args": {"query": "test"}, "output": "result"},
            {"tool": "delegate", "args": {"task": "sub-task"}, "output": "delegated result"},
        ]
        with patch("app.core.auto_skills.llm") as mock_llm:
            await maybe_extract_skill("hello", tool_results, "answer", skills)
            mock_llm.invoke_nothink.assert_not_called()


# ===========================================================================
# Valid LLM response parsing
# ===========================================================================

class TestValidLLMResponse:
    @pytest.mark.asyncio
    async def test_creates_skill_from_valid_response(self, db, monkeypatch):
        """A well-formed LLM response should create a skill."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill(
                "What's the stock price of AAPL?",
                _make_tool_results(2),
                "The stock price is $150.",
                skills,
            )

        # Verify skill was created
        skill = skills.get_matching_skill("What is the stock price of GOOG")
        assert skill is not None
        assert skill.name == "stock_price_lookup"
        assert len(skill.steps) == 2

    @pytest.mark.asyncio
    async def test_skip_response_creates_nothing(self, db, monkeypatch):
        """LLM returning {"skip": true} should not create a skill."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({"skip": True}))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        # No skills should exist
        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_empty_name_creates_nothing(self, db, monkeypatch):
        """LLM returning an empty name should not create a skill."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()
        resp["name"] = ""

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0


# ===========================================================================
# Invalid / malformed LLM responses
# ===========================================================================

class TestMalformedLLMResponse:
    @pytest.mark.asyncio
    async def test_invalid_json_does_not_crash(self, db, monkeypatch):
        """Malformed JSON from LLM should be handled gracefully."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="not valid json at all")
            mock_llm.extract_json_object = MagicMock(return_value={})

            # Should not raise
            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_invalid_regex_rejected(self, db, monkeypatch):
        """Invalid regex in trigger_pattern should be rejected."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()
        resp["trigger_pattern"] = "[invalid(regex"

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_unknown_tool_in_steps_rejected(self, db, monkeypatch):
        """Steps referencing unknown tools should cause rejection."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()
        resp["steps"] = [
            {"tool": "nonexistent_tool", "args_template": {}, "output_key": "r"},
        ]

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_llm_exception_does_not_crash(self, db, monkeypatch):
        """LLM raising an exception should be caught and logged."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        with patch("app.core.auto_skills.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=RuntimeError("LLM down"))

            # Should not raise
            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_empty_trigger_pattern_rejected(self, db, monkeypatch):
        """Empty trigger_pattern should be rejected."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()
        resp["trigger_pattern"] = ""

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0


# ===========================================================================
# Broad pattern rejection
# ===========================================================================

class TestBroadPatternRejection:
    @pytest.mark.asyncio
    async def test_broad_pattern_rejected(self, db, monkeypatch):
        """A trigger pattern like '.*' should be rejected as too broad."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()
        resp["trigger_pattern"] = ".*"  # matches everything

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_single_word_pattern_rejected(self, db, monkeypatch):
        """A very generic single-word pattern should be rejected."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()
        resp["trigger_pattern"] = r"\w+"  # matches any word

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill("hello", _make_tool_results(2), "answer", skills)

        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 0


# ===========================================================================
# Deduplication
# ===========================================================================

class TestDeduplication:
    @pytest.mark.asyncio
    async def test_duplicate_pattern_boosts_confidence(self, db, monkeypatch):
        """Creating a skill with an existing trigger pattern boosts its confidence."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)
        resp = _valid_llm_response()

        # Create the skill once directly
        first_id = skills.create_skill(
            name=resp["name"],
            trigger_pattern=resp["trigger_pattern"],
            steps=resp["steps"],
            answer_template=resp["answer_template"],
            initial_success_rate=0.7,
        )
        assert first_id is not None

        # Get initial success rate
        initial = db.fetchone("SELECT success_rate FROM skills WHERE id = ?", (first_id,))
        initial_rate = initial["success_rate"]

        # Now extract the same pattern again via auto_skills
        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill(
                "What's the stock price of MSFT?",
                _make_tool_results(2),
                "The stock price is $400.",
                skills,
            )

        # Only one skill should exist (deduped)
        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 1

        # Confidence should have been boosted
        updated = db.fetchone("SELECT success_rate FROM skills WHERE id = ?", (first_id,))
        assert updated["success_rate"] > initial_rate

    @pytest.mark.asyncio
    async def test_duplicate_name_updates_existing(self, db, monkeypatch):
        """Creating a skill with an existing name (case-insensitive) updates the existing one."""
        monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        skills = SkillStore(db)

        # Create a skill with original name
        first_id = skills.create_skill(
            name="Stock_Price_Lookup",
            trigger_pattern=r"price of (\w+) stock",
            steps=[{"tool": "web_search", "args_template": {"query": "stock {query}"}, "output_key": "r"}],
            answer_template="Price: {r}",
        )
        assert first_id is not None

        # Now create a skill with same name (different case) but different trigger
        resp = _valid_llm_response()
        resp["name"] = "stock_price_lookup"  # same name, lowercase

        with patch("app.core.auto_skills.llm") as mock_llm, \
             patch("app.core.auto_skills._get_tool_names", return_value={"web_search", "calculator"}):
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(resp))
            mock_llm.extract_json_object = lambda x: json.loads(x)

            await maybe_extract_skill(
                "What's the stock price of AAPL?",
                _make_tool_results(2),
                "The stock price is $150.",
                skills,
            )

        # Still only one skill (name-deduped)
        rows = db.fetchall("SELECT * FROM skills")
        assert len(rows) == 1
