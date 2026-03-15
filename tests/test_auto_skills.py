"""Tests for autonomous skill creation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.auto_skills import maybe_extract_skill


@pytest.fixture
def mock_skills(db):
    from app.core.skills import SkillStore
    return SkillStore(db)


class TestAutoSkillCreation:
    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, mock_skills):
        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": False})()):
            await maybe_extract_skill(
                "test query",
                [{"tool": "web_search", "args": {}}, {"tool": "calculator", "args": {}}],
                "answer",
                mock_skills,
            )
        # No skill should be created
        assert len(mock_skills.get_all_skills()) == 0

    @pytest.mark.asyncio
    async def test_skips_single_tool(self, mock_skills):
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
    async def test_skips_delegate_interactions(self, mock_skills):
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
    async def test_creates_skill_on_valid_extraction(self, mock_skills):
        llm_response = json.dumps({
            "name": "price_compare",
            "trigger_pattern": r"(?i)compare.*price",
            "steps": [
                {"tool": "web_search", "args_template": {"query": "{query} price"}, "output_key": "result"},
            ],
            "answer_template": "Based on search: {result}",
        })

        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True})()):
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
    async def test_rejects_broad_pattern(self, mock_skills):
        llm_response = json.dumps({
            "name": "broad_skill",
            "trigger_pattern": r".*",
            "steps": [{"tool": "web_search", "args_template": {"query": "{query}"}}],
        })

        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True})()):
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
    async def test_handles_llm_skip_response(self, mock_skills):
        with patch("app.core.auto_skills.config",
                   type("C", (), {"ENABLE_AUTO_SKILL_CREATION": True})()):
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
