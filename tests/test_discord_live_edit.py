"""Discord live-edit streaming (#47): the draft message is SENT the moment the
'refining' stage signal arrives and EDITED in place when the REVISION lands —
the reader gets an answer in generation time, not generation+refine time."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schema import EventType, StreamEvent


def _channel(monkeypatch, events):
    from app.channels.discord import DiscordBot

    ch = DiscordBot.__new__(DiscordBot)  # skip __init__ (no bot client)
    ch._send_lock = asyncio.Lock()
    ch._get_conversation_id = AsyncMock(return_value="conv-test")

    async def fake_think(**kwargs):
        for e in events:
            yield e

    monkeypatch.setattr("app.core.brain.think", fake_think)
    msg = MagicMock()
    draft_message = MagicMock()
    draft_message.edit = AsyncMock()
    msg.reply = AsyncMock(return_value=draft_message)
    msg.author.id = 1
    return ch, msg, draft_message


@pytest.mark.asyncio
async def test_draft_sent_on_refining_then_edited_with_revision(monkeypatch):
    ch, msg, draft_message = _channel(monkeypatch, [
        StreamEvent(type=EventType.TOKEN, data={"text": "Draft answer body."}),
        StreamEvent(type=EventType.THINKING, data={"stage": "refining"}),
        StreamEvent(type=EventType.REVISION, data={"text": "Refined final answer body."}),
        StreamEvent(type=EventType.DONE, data={}),
    ])
    await ch._stream_reply(msg, "test query")
    msg.reply.assert_awaited_once()
    sent = msg.reply.await_args.args[0]
    assert sent.startswith("Draft answer body.") and "refining" in sent
    draft_message.edit.assert_awaited_once_with(content="Refined final answer body.")


@pytest.mark.asyncio
async def test_unchanged_answer_edit_removes_refining_marker(monkeypatch):
    ch, msg, draft_message = _channel(monkeypatch, [
        StreamEvent(type=EventType.TOKEN, data={"text": "Stable answer body."}),
        StreamEvent(type=EventType.THINKING, data={"stage": "refining"}),
        StreamEvent(type=EventType.DONE, data={}),
    ])
    await ch._stream_reply(msg, "test query")
    draft_message.edit.assert_awaited_once_with(content="Stable answer body.")


@pytest.mark.asyncio
async def test_long_draft_falls_back_to_single_send(monkeypatch):
    long_text = "word " * 600  # > 2000 chars → no live-edit, one delivery at end
    ch, msg, draft_message = _channel(monkeypatch, [
        StreamEvent(type=EventType.TOKEN, data={"text": long_text}),
        StreamEvent(type=EventType.THINKING, data={"stage": "refining"}),
        StreamEvent(type=EventType.DONE, data={}),
    ])
    await ch._stream_reply(msg, "test query")
    draft_message.edit.assert_not_awaited()
    assert msg.reply.await_count >= 2  # split into chunks at the end
