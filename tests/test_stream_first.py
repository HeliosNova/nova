"""Stream-first refine (2026-07-06): the validated DRAFT must stream BEFORE the
refine chain runs (time-to-first-token was 30-60s of blank screen), and when
refine changes the answer a REVISION event replaces the draft in place.

Deterministic: LLM fully mocked; ordering asserted on the event sequence."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services, think
from app.core.llm import GenerationResult, StreamChunk, _strip_think_tags, _extract_tool_calls
from app.core.memory import ConversationStore, UserFactStore
from app.schema import EventType


def _mock_llm(mock_llm, content="The draft answer is forty two."):
    async def _stream(*args, **kwargs):
        yield StreamChunk(content=content, done=False)
        yield StreamChunk(done=True)

    mock_llm.stream_with_thinking = MagicMock(side_effect=_stream)
    mock_llm.generate_with_tools = AsyncMock(return_value=GenerationResult(
        content=content, tool_calls=[], raw={}, thinking="",
    ))
    mock_llm.get_provider = MagicMock(return_value=MagicMock(
        capabilities=MagicMock(needs_emphatic_prompts=False),
    ))
    mock_llm._strip_think_tags = _strip_think_tags
    mock_llm._extract_tool_calls = _extract_tool_calls
    mock_llm.extract_json_object = MagicMock(return_value=None)
    mock_llm.invoke_nothink = AsyncMock(return_value="COMPLETE")
    mock_llm.GenerationResult = GenerationResult
    mock_llm.StreamChunk = StreamChunk


async def _collect_events(query="What is six times seven?"):
    events = []
    async for ev in think(query=query):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_draft_tokens_stream_before_refine(db):
    svc = Services(conversations=ConversationStore(db), user_facts=UserFactStore(db))
    set_services(svc)
    refine_calls = {"n": 0}

    async def fake_refine(*args, **kwargs):
        refine_calls["n"] += 1
        return "The draft answer is forty two.", 0.9, ""

    with patch("app.core.brain.llm") as mock_llm, \
         patch("app.core.brain._refine_response", side_effect=fake_refine):
        _mock_llm(mock_llm)
        events = await _collect_events()

    types = [e.type for e in events]
    assert EventType.TOKEN in types, f"no draft tokens emitted: {types}"
    assert EventType.DONE in types
    # refine unchanged answer → no REVISION
    assert EventType.REVISION not in types
    assert refine_calls["n"] == 1, "refine chain must still run after streaming"
    draft_text = "".join(e.data.get("text", "") for e in events if e.type == EventType.TOKEN)
    assert "forty two" in draft_text


@pytest.mark.asyncio
async def test_revision_event_replaces_changed_answer(db):
    svc = Services(conversations=ConversationStore(db), user_facts=UserFactStore(db))
    set_services(svc)

    async def fake_refine(*args, **kwargs):
        return "The REFINED answer is forty two exactly.", 0.9, ""

    with patch("app.core.brain.llm") as mock_llm, \
         patch("app.core.brain._refine_response", side_effect=fake_refine):
        _mock_llm(mock_llm)
        events = await _collect_events()

    types = [e.type for e in events]
    assert EventType.REVISION in types, f"refine changed the answer but no REVISION: {types}"
    # ordering: every TOKEN precedes the REVISION, which precedes DONE
    assert max(i for i, t in enumerate(types) if t == EventType.TOKEN) \
        < types.index(EventType.REVISION) < types.index(EventType.DONE)
    rev = next(e for e in events if e.type == EventType.REVISION)
    assert "REFINED" in rev.data["text"]
