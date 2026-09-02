"""Knowing-first chat (audit 2026-09-01).

Measured: the owner's one real knowledge question ("what did your monitors
learn about AI…") took 131 s, five tool rounds and shipped a 0.05-quality
fragment while the dossier and storylines were already in the prompt; both
chronic memory-eval failures had the seeded lesson at rank 1 in the prompt but
the 9B web-searched past it; get_relevant_dossiers scanned only the 60 most
recently updated dossiers (29 unreachable), injected one, cut to 900 of a
median 3,141 chars, and never showed Open questions; memory_search could not
see lessons although the prompt told the model to use it for 'my notes'.
"""
from __future__ import annotations

import pytest

from app.core import brain
from app.core.dossiers import get_relevant_dossiers


def _put(db, title, body, dkey, kind="domain", age_days=0):
    db.execute(
        "INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, '', 1, datetime('now', ?), datetime('now', ?))",
        (kind, dkey, title, body, f"-{age_days} days", f"-{age_days} days"))


def test_all_dossiers_are_searchable_not_just_the_newest_60(db):
    for i in range(70):
        _put(db, f"Filler Domain {i}", f"## Current understanding\nFiller {i} text about nothing.", f"filler-{i}", age_days=i)
    _put(db, "Anthropic", "## Current understanding\nAnthropic is preparing an IPO amid a Sony lawsuit.",
         "anthropic", kind="entity", age_days=90)
    got = get_relevant_dossiers(db, "what do you know about Anthropic?", limit=2)
    assert got and got[0]["title"] == "Anthropic"


def test_excerpt_is_generous_and_open_questions_appear_when_asked(db):
    cu = "The domain moved a lot this week. " * 120  # ~4,000 chars
    body = ("## Current understanding\n" + cu + "\n\n## How we got here\nx\n\n"
            "## Open questions\n- Will the merger close before Q4?\n- Who funds the consortium?\n")
    _put(db, "AI and ML", body, "ai-and-ml")
    got = get_relevant_dossiers(db, "what is the state of AI and ML right now?", limit=1)
    assert got and len(got[0]["excerpt"]) > 1500 and "Open questions" not in got[0]["excerpt"]
    got2 = get_relevant_dossiers(db, "what don't you know yet about AI and ML?", limit=1)
    assert got2 and "Open questions" in got2[0]["excerpt"] and "Who funds the consortium?" in got2[0]["excerpt"]


def test_two_dossiers_can_inject(db):
    _put(db, "Microsoft", "## Current understanding\nMicrosoft shipped Copilot changes.", "microsoft", kind="entity")
    _put(db, "Technology", "## Current understanding\nMicrosoft and Apple both shipped updates this week.", "technology")
    got = get_relevant_dossiers(db, "what's the latest on Microsoft updates?", limit=2)
    assert {g["title"] for g in got} == {"Microsoft", "Technology"}


DOSSIER_BLOCK = ("Standing knowledge dossiers (Nova's accumulated understanding)...\n"
                 "### AI and ML\nOpenAI restricted GPT-5.6 access at government request.\n"
                 "### Microsoft\nMicrosoft shipped Copilot changes.")
LESSONS_BLOCK = ("- [HIGH] Nova scheduler codename: Nova's internal task scheduler is codenamed 'Chronos'.\n"
                 "- [HIGH] Dr. Lena Voss location: Per the user's notes, Dr. Lena Voss is located in Reykjavik.")


def test_knowing_questions_run_tool_less_when_a_dossier_is_in_context():
    assert brain._knowing_answers_query("What did your monitors learn about AI developments in the last day or two?", DOSSIER_BLOCK, "")
    assert brain._knowing_answers_query("According to your dossiers, what is the Fed rate outlook?", DOSSIER_BLOCK, "")
    assert brain._knowing_answers_query("what's the latest on Microsoft?", DOSSIER_BLOCK, "")
    assert not brain._knowing_answers_query("What did your monitors learn about AI?", "", "")
    assert not brain._knowing_answers_query("What is the capital of France?", DOSSIER_BLOCK, "")
    assert not brain._knowing_answers_query("Search the web for the latest on Microsoft", DOSSIER_BLOCK, "")


def test_owner_taught_lessons_answer_tool_less():
    assert brain._knowing_answers_query("Remind me: which codename did we assign to the scheduler component in Nova?", "", LESSONS_BLOCK)
    assert brain._knowing_answers_query("According to my notes, where is the researcher Dr. Lena Voss based?", "", LESSONS_BLOCK)
    assert not brain._knowing_answers_query("How do I bake sourdough bread?", "", LESSONS_BLOCK)


def test_lesson_prompt_makes_owner_facts_authoritative():
    from app.core.prompt import format_lessons_for_prompt
    from types import SimpleNamespace
    out = format_lessons_for_prompt([SimpleNamespace(topic="t", lesson_text="x is y", correct_answer="x is y",
                                                     wrong_answer="", confidence=0.9)])
    assert "authoritative" in out and "do not web-search" in out.lower().replace("web_search", "web-search")


@pytest.mark.asyncio
async def test_memory_search_sees_lessons(db, monkeypatch):
    from app.core.learning import LearningEngine
    from app.tools.memory_tool import MemorySearchTool
    eng = LearningEngine(db)
    monkeypatch.setattr(eng, "_get_lessons_collection", lambda: None)
    db.execute("INSERT INTO lessons (topic, correct_answer, wrong_answer, lesson_text, confidence, times_helpful) "
               "VALUES ('Dr. Lena Voss location', 'Dr. Lena Voss is located in Reykjavik.', '', "
               "'Per the user notes, Dr. Lena Voss is located in Reykjavik.', 0.95, 0)")
    from app.config import config as _cfg
    old = _cfg.MIN_RRF_SCORE
    _cfg.update(MIN_RRF_SCORE=0.005)
    try:
        tool = MemorySearchTool(conversations=None, user_facts=None, learning=eng)
        res = await tool.execute(query="where is Dr. Lena Voss based")
    finally:
        _cfg.update(MIN_RRF_SCORE=old)
    assert res.success and "Reykjavik" in res.output and "Lesson" in res.output


def test_token_estimate_matches_measured_ratio():
    from app.core.text_utils import estimate_tokens
    n = estimate_tokens("a" * 3250)
    assert 950 <= n <= 1050, n
