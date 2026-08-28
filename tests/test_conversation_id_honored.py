"""Client-supplied conversation_id is honored (2026-08-27).

The chat pipeline's `conv is None` fallback discarded any client-supplied
conversation_id and minted a fresh UUID, so an external API caller that
chose its own id (a validated, documented field) got a NEW conversation
every turn — no cross-turn recall, and GSW episodic summaries never reached
the >=4 messages they need. create_conversation now honors the supplied id.
"""
import sqlite3

import pytest

from app.core.memory import ConversationStore


class _DB:
    """Minimal SafeDB-shaped wrapper over in-memory sqlite."""

    def __init__(self):
        self._c = sqlite3.connect(":memory:")
        self._c.row_factory = sqlite3.Row
        self._c.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT,"
            " created_at TEXT DEFAULT (datetime('now')),"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )

    def execute(self, sql, params=()):
        cur = self._c.execute(sql, params); self._c.commit(); return cur

    def fetchone(self, sql, params=()):
        return self._c.execute(sql, params).fetchone()

    def fetchall(self, sql, params=()):
        return self._c.execute(sql, params).fetchall()


@pytest.fixture
def store():
    db = _DB()
    s = ConversationStore(db)
    s._test_db = db
    return s


class TestConversationIdHonored:
    def test_supplied_id_persists(self, store):
        got = store.create_conversation(conv_id="gsw-verify-123")
        assert got == "gsw-verify-123"
        assert store.get_conversation("gsw-verify-123") is not None

    def test_absent_id_still_mints_uuid(self, store):
        got = store.create_conversation()
        assert got and got != "New Chat"
        assert store.get_conversation(got) is not None

    def test_idempotent_under_repeat(self, store):
        a = store.create_conversation(title="first", conv_id="dup-id")
        b = store.create_conversation(title="second", conv_id="dup-id")
        assert a == b == "dup-id"
        # OR IGNORE: the first title wins, no duplicate row, no crash
        assert store.get_conversation("dup-id")["title"] == "first"
        rows = store._test_db.fetchall("SELECT id FROM conversations WHERE id='dup-id'")
        assert len(rows) == 1
