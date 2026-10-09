"""Chat retrieval reads the KG's history when the question is about the past
(2026-10-09).

The long-memory eval's temporal task shipped failing on purpose: "Where did
Ilse Varga live before she moved to Valletta?" needs the SUPERSEDED fact, and
`format_for_prompt` skips every superseded row. The bitemporal store has kept
that history since 2026-05-16 and chat never read it.

Rule: when the query carries a temporal cue (before / previously / used to /
formerly / earlier / moved from / history / changed), the retrieved facts'
(subject, predicate) pairs are looked up in the KG history and their
superseded versions are appended under a "History" heading, dated, naming
what replaced them. A question about the present is unchanged — the priming
A/B showed prior context costs grounding when it is not asked for.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.core import brain as brain_mod
from app.core.kg import KnowledgeGraph


@pytest.fixture
def kg(db):
    return KnowledgeGraph(db)


async def _seed_move(kg: KnowledgeGraph) -> None:
    await kg.add_fact("Ilse Varga", "lives_in", "Tromso", confidence=0.95, valid_from="2026-03-01")
    await kg.add_fact("Ilse Varga", "lives_in", "Valletta", confidence=0.95, valid_from="2026-08-01")


def test_history_block_names_the_superseded_value_and_its_replacement(kg):
    asyncio.run(_seed_move(kg))
    current = kg.get_relevant_facts("Ilse Varga lives", limit=8)
    assert any(f.object == "Valletta" for f in current), current
    block = kg.format_history_for_prompt(current)
    assert "Tromso" in block, block
    assert "Valletta" in block, block
    assert "2026-03-01" in block, block


def test_history_block_is_empty_when_nothing_was_ever_revised(kg):
    asyncio.run(kg.add_fact("Odalys Ferrante", "works_at", "Pelagic Labs", confidence=0.95))
    current = kg.get_relevant_facts("Odalys Ferrante works", limit=8)
    assert kg.format_history_for_prompt(current) == ""


def test_history_block_is_bounded(kg):
    async def seed():
        for i in range(12):
            await kg.add_fact("Corvin Ashlake", "lives_in", f"City{i:02d}", confidence=0.95,
                              valid_from=f"2025-{(i % 12) + 1:02d}-01")
    asyncio.run(seed())
    current = kg.get_relevant_facts("Corvin Ashlake lives", limit=8)
    block = kg.format_history_for_prompt(current, max_lines=4)
    assert block
    assert len([ln for ln in block.splitlines() if ln.strip().startswith("-")]) <= 4, block


@pytest.mark.parametrize("q", [
    "Where did Ilse Varga live before she moved to Valletta?",
    "Where did she live previously?",
    "Who used to lead the Halyard project?",
    "What was the price of X earlier this year?",
    "Has Nova's view on Broadcom changed since August?",
    "Give me the history of Ilse Varga's addresses",
])
def test_temporal_cues_are_recognised(q):
    assert brain_mod._TEMPORAL_QUERY_RE.search(q), q


@pytest.mark.parametrize("q", [
    "Where does Ilse Varga live now?",
    "What do you know about Corvin Ashlake?",
    "latest AI news",
])
def test_present_tense_questions_are_not_temporal(q):
    assert not brain_mod._TEMPORAL_QUERY_RE.search(q), q


def _svc(kg):
    return SimpleNamespace(
        user_facts=None, skills=None, learning=None, kg=kg,
        reflexions=None, retriever=None, external_skills=None,
    )


def test_gather_context_appends_history_for_a_temporal_question(kg):
    asyncio.run(_seed_move(kg))
    ctx = asyncio.run(brain_mod._gather_context(
        _svc(kg), "Where did Ilse Varga live before she moved to Valletta?", "general", "conv-t",
    ))
    assert "Valletta" in ctx.kg_facts_text, ctx.kg_facts_text
    assert "Tromso" in ctx.kg_facts_text, ctx.kg_facts_text


def test_gather_context_keeps_the_present_clean(kg):
    asyncio.run(_seed_move(kg))
    ctx = asyncio.run(brain_mod._gather_context(
        _svc(kg), "Where does Ilse Varga live now?", "general", "conv-p",
    ))
    assert "Valletta" in ctx.kg_facts_text, ctx.kg_facts_text
    assert "Tromso" not in ctx.kg_facts_text, ctx.kg_facts_text
