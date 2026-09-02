"""Talking back from Telegram (audit 2026-09-01).

Measured: digests were delivered outside any conversation, so a reply like
"that's wrong, X is Y" had no prior assistant turn to correct (correction
detection requires prev_answer in the SAME conversation); the
channel_conversations table did not even exist in the live DB; Telegram chat
replies went out with no parse_mode, so markdown rendered as raw asterisks.
"""
from __future__ import annotations

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore
from app.database import ChannelConversationStore


def test_delivered_digest_becomes_an_assistant_turn(db):
    from app.monitors.heartbeat_loop import HeartbeatLoop
    convs = ConversationStore(db)
    set_services(Services(conversations=convs, user_facts=UserFactStore(db)))
    loop = HeartbeatLoop.__new__(HeartbeatLoop)
    digest = "[Domain Study: Finance]\n## 💵 finance — domain overview\nThe Fed held rates (reuters.com). " * 40
    loop._record_channel_turn("telegram", "123456", digest)
    conv_id = ChannelConversationStore(db).get("telegram", "123456")
    assert conv_id, "the owner's channel conversation must exist after a delivery"
    hist = convs.get_history(conv_id, limit=5)
    assert hist and hist[-1].role == "assistant"
    assert hist[-1].content.startswith("[Domain Study: Finance]")
    assert len(hist[-1].content) <= 1300, "digest turns are bounded so chat history stays cheap"
    # a second delivery reuses the same conversation
    loop._record_channel_turn("telegram", "123456", "[System Health]\nall good")
    assert ChannelConversationStore(db).get("telegram", "123456") == conv_id
    assert len(convs.get_history(conv_id, limit=5)) == 2


def test_record_turn_is_a_noop_without_a_chat_id(db):
    from app.monitors.heartbeat_loop import HeartbeatLoop
    set_services(Services(conversations=ConversationStore(db), user_facts=UserFactStore(db)))
    loop = HeartbeatLoop.__new__(HeartbeatLoop)
    loop._record_channel_turn("telegram", None, "x")
    assert db.fetchone("SELECT COUNT(*) AS c FROM channel_conversations")["c"] == 0


def test_telegram_chat_replies_are_html():
    from app.channels.telegram import TelegramBot
    chunks = TelegramBot._reply_chunks("**Bold** and *italic* with `code` https://example.com/x")
    assert chunks and "<b>Bold</b>" in chunks[0]
    assert "**" not in chunks[0]
