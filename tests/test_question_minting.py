"""Item 2 (2026-10-12): dossier open questions must ask after something public.

Of 40 failed dossier_open_question items, the recurring shapes asked for what is
not published ("undisclosed conditions", "specific technical mechanisms", a
vendor's patch rollout); the 20 that resolved asked after public processes.
"""

from __future__ import annotations

import pytest

from app.core.dossiers import _UPDATE_PROMPT, _WORLD_PROMPT, _is_unpublished_question
from app.core.questions import extract_questions


@pytest.mark.parametrize("q", [
    "What are the specific undisclosed conditions Iran has demanded for relinquishing control over the Strait?",
    "What specific technical mechanisms allowed the agent to bypass denied access controls?",
    "What specific metrics does MAWILE use to quantify the sensitivity of LLM judge verdicts?",
    "What is the current status of Tencent's patch deployment for the WeChat VoIP stack vulnerability?",
    "What is the current status of Agility Robotics' post-merger integration plan for its fleet?",
    "What is the current verified production yield rate for TSMC's CoWoS packaging?",
    "What is the internal reallocation status of the NSF's $1 billion?",
])
def test_unpublished_shapes_are_caught(q):
    assert _is_unpublished_question(q)


@pytest.mark.parametrize("q", [
    "What is the current status of the SEC custody proposal for investment advisers under White House review?",
    "What is the current status of the federal corruption investigation into Flavio Bolsonaro?",
    "What is the current status of the Senate negotiations on the federal permitting overhaul?",
    "What is the current total contracted revenue value of Micron's 16 Strategic Customer Agreements?",
    "Who holds the chair of the FOMC's communications subcommittee?",
])
def test_public_process_questions_pass(q):
    assert not _is_unpublished_question(q)


def test_extract_questions_drops_unpublished_shapes():
    body = (
        "## Open questions\n"
        "- What are the specific undisclosed conditions Iran has demanded for the Strait?\n"
        "- What is the current status of the SEC custody proposal under White House review?\n"
        "- Will the Fed cut rates in December?\n"
        "## Key facts\n- x\n"
    )
    assert extract_questions(body) == ["What is the current status of the SEC custody proposal under White House review?"]


def test_prompts_ask_for_public_settleable_questions():
    for p in (_UPDATE_PROMPT, _WORLD_PROMPT):
        assert "PUBLIC" in p and "undisclosed" in p
