"""Scheduled procedure-skill induction from proven memory (audit 2026-08-23).

Root cause of 0-organic-skills-ever: chat-path extraction is gated on
tool-using live chat, which a monitor-driven personal AI barely generates.
`induce_procedure_skills` decouples acquisition from chat (the pattern that
makes auto_tools the one working self-improvement path): it mines the durable
substrate — success reflexions + proven lessons — clusters by topic keywords,
requires >=2 corroborating items per cluster, and LLM-distills each into an NL
procedure skill (migration-30 representation, matched semantically into the
prompt as guidance).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.auto_skills import induce_procedure_skills
from app.core.skills import SkillStore


def _seed_lesson(db, topic: str, text: str, helpful: int = 5):
    db.execute(
        "INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, "
        "confidence, times_helpful) VALUES (?, 'w', 'c', ?, 0.9, ?)",
        (topic, text, helpful),
    )


def _seed_reflexion(db, summary: str, reflection: str, quality: float = 0.9):
    db.execute(
        "INSERT INTO reflexions (task_summary, outcome, reflection, quality_score, "
        "tools_used, is_eval) VALUES (?, 'success', ?, ?, 'web_search', 0)",
        (summary, reflection, quality),
    )


def _valid_induction() -> dict:
    return {
        "name": "semiconductor_supply_analysis",
        "description": "Analyze semiconductor supply questions with sourced data",
        "procedure_text": (
            "When to use: queries about semiconductor supply, fab capacity or "
            "chip export flows.\n"
            "Steps:\n"
            "1. Search for the latest supply/capacity figures from at least two "
            "independent sources.\n"
            "2. Cross-check numbers against prior knowledge-graph facts before "
            "citing.\n"
            "3. Answer with specific figures, dates and named fabs.\n"
            "Pitfalls: do not extrapolate capacity numbers across process nodes."
        ),
    }


def _skill_rows(db):
    return db.fetchall(
        "SELECT id, name, kind, source, enabled FROM skills WHERE source='induced'"
    )


@pytest.mark.asyncio
async def test_two_corroborating_lessons_induce_skill(db, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
    from app.config import reset_config

    reset_config()
    _seed_lesson(db, "semiconductor forecasting", "Always cite two sources for fab data")
    _seed_lesson(db, "forecasting semiconductor supply", "Cross-check capacity numbers")
    skills = SkillStore(db)
    with patch("app.core.auto_skills.llm") as mock_llm:
        mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(_valid_induction()))
        mock_llm.extract_json_object = lambda x: json.loads(x)
        created = await induce_procedure_skills(db, skills, max_new=2)
    rows = _skill_rows(db)
    assert created == 1 and len(rows) == 1, f"expected 1 induced skill, got {rows}"
    assert rows[0]["kind"] == "procedure"


@pytest.mark.asyncio
async def test_single_item_cluster_not_induced(db, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
    from app.config import reset_config

    reset_config()
    _seed_lesson(db, "quantum computing errors", "Check error-correction context")
    skills = SkillStore(db)
    with patch("app.core.auto_skills.llm") as mock_llm:
        mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(_valid_induction()))
        mock_llm.extract_json_object = lambda x: json.loads(x)
        created = await induce_procedure_skills(db, skills, max_new=2)
    assert created == 0
    mock_llm.invoke_nothink.assert_not_called()


@pytest.mark.asyncio
async def test_success_reflexions_also_cluster(db, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
    from app.config import reset_config

    reset_config()
    _seed_reflexion(db, "semiconductor forecasting task", "Cited two sources; worked well")
    _seed_reflexion(db, "forecasting semiconductor exports", "Cross-checking KG facts helped")
    skills = SkillStore(db)
    with patch("app.core.auto_skills.llm") as mock_llm:
        mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps(_valid_induction()))
        mock_llm.extract_json_object = lambda x: json.loads(x)
        created = await induce_procedure_skills(db, skills, max_new=2)
    assert created == 1


@pytest.mark.asyncio
async def test_max_new_cap_respected(db, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
    from app.config import reset_config

    reset_config()
    _seed_lesson(db, "semiconductor forecasting", "a" * 40)
    _seed_lesson(db, "forecasting semiconductor supply", "b" * 40)
    _seed_lesson(db, "biotech genomics pipelines", "c" * 40)
    _seed_lesson(db, "genomics biotech assays", "d" * 40)
    skills = SkillStore(db)

    calls = {"n": 0}

    async def _one_per_call(*a, **k):
        calls["n"] += 1
        obj = _valid_induction()
        obj["name"] = f"induced_skill_{calls['n']}"
        return json.dumps(obj)

    with patch("app.core.auto_skills.llm") as mock_llm:
        mock_llm.invoke_nothink = AsyncMock(side_effect=_one_per_call)
        mock_llm.extract_json_object = lambda x: json.loads(x)
        created = await induce_procedure_skills(db, skills, max_new=1)
    assert created == 1 and len(_skill_rows(db)) == 1


@pytest.mark.asyncio
async def test_llm_skip_creates_nothing(db, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "true")
    from app.config import reset_config

    reset_config()
    _seed_lesson(db, "semiconductor forecasting", "x" * 40)
    _seed_lesson(db, "forecasting semiconductor supply", "y" * 40)
    skills = SkillStore(db)
    with patch("app.core.auto_skills.llm") as mock_llm:
        mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({"skip": True}))
        mock_llm.extract_json_object = lambda x: json.loads(x)
        created = await induce_procedure_skills(db, skills, max_new=2)
    assert created == 0 and len(_skill_rows(db)) == 0


@pytest.mark.asyncio
async def test_disabled_flag_no_llm(db, monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_SKILL_CREATION", "false")
    from app.config import reset_config

    reset_config()
    _seed_lesson(db, "semiconductor forecasting", "x" * 40)
    _seed_lesson(db, "forecasting semiconductor supply", "y" * 40)
    skills = SkillStore(db)
    with patch("app.core.auto_skills.llm") as mock_llm:
        created = await induce_procedure_skills(db, skills, max_new=2)
        mock_llm.invoke_nothink.assert_not_called()
    assert created == 0
