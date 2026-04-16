"""Tests for the Chat API endpoints (app/api/chat.py).

Covers: POST /chat, POST /chat/stream, conversations CRUD, user facts CRUD,
input validation, error cases.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore
from app.main import app, _rate_limit_requests
from app.schema import EventType, StreamEvent


@pytest.fixture
def client(db):
    """FastAPI test client with services wired (dev mode, no API_KEY)."""
    _rate_limit_requests.clear()
    conversations = ConversationStore(db)
    user_facts = UserFactStore(db)
    svc = Services(
        conversations=conversations,
        user_facts=user_facts,
    )
    set_services(svc)
    return TestClient(app)


@pytest.fixture
def svc(db):
    """Return Services for direct manipulation in tests."""
    conversations = ConversationStore(db)
    user_facts = UserFactStore(db)
    s = Services(
        conversations=conversations,
        user_facts=user_facts,
    )
    set_services(s)
    return s


# ===========================================================================
# POST /chat — synchronous endpoint
# ===========================================================================

class TestChatSync:
    def test_empty_query_rejected(self, client):
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_query_too_long_rejected(self, client):
        resp = client.post("/api/chat", json={"query": "x" * 50_001})
        assert resp.status_code == 422

    def test_invalid_conversation_id_rejected(self, client):
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": "../../etc/passwd",
        })
        assert resp.status_code == 422

    def test_valid_conversation_id_accepted(self, client):
        """Valid alphanumeric conversation IDs should not fail validation."""
        # The endpoint may fail for other reasons (LLM not connected),
        # but it should pass input validation (not 422).
        with patch("app.api.chat.think") as mock_think:
            mock_think.return_value = _mock_think_gen("hello response", "test-conv-id")
            resp = client.post("/api/chat", json={
                "query": "hello",
                "conversation_id": "test-conv-123",
            })
            assert resp.status_code != 422

    def test_sync_chat_returns_response(self, client):
        """Successful sync chat returns answer + conversation_id."""
        with patch("app.api.chat.think") as mock_think:
            mock_think.return_value = _mock_think_gen("Hello, world!", "conv-123")
            resp = client.post("/api/chat", json={"query": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == "Hello, world!"
            assert data["conversation_id"] == "conv-123"

    def test_sync_chat_error_returns_500(self, client):
        """Error events from think() return 500."""
        with patch("app.api.chat.think") as mock_think:
            mock_think.return_value = _mock_think_error("Something broke")
            resp = client.post("/api/chat", json={"query": "hello"})
            assert resp.status_code == 500


# ===========================================================================
# POST /chat/stream — SSE streaming endpoint
# ===========================================================================

class TestChatStream:
    def test_stream_returns_sse(self, client):
        """Streaming endpoint returns text/event-stream content type."""
        with patch("app.api.chat.think") as mock_think:
            mock_think.return_value = _mock_think_gen("Hello!", "conv-456")
            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_contains_done(self, client):
        """Stream should end with [DONE] event."""
        with patch("app.api.chat.think") as mock_think:
            mock_think.return_value = _mock_think_gen("Hello!", "conv-456")
            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert "[DONE]" in resp.text

    def test_stream_empty_query_rejected(self, client):
        resp = client.post("/api/chat/stream", json={"query": ""})
        assert resp.status_code == 422


# ===========================================================================
# Conversations — GET /chat/conversations
# ===========================================================================

class TestConversationsList:
    def test_list_empty(self, client):
        resp = client.get("/api/chat/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_conversations(self, client, svc):
        conv_id = svc.conversations.create_conversation()
        svc.conversations.add_message(conv_id, "user", "hello")
        resp = client.get("/api/chat/conversations")
        assert resp.status_code == 200
        convs = resp.json()
        assert len(convs) >= 1

    def test_list_limit(self, client, svc):
        for _ in range(5):
            svc.conversations.create_conversation()
        resp = client.get("/api/chat/conversations?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) <= 3

    def test_list_limit_too_high(self, client):
        resp = client.get("/api/chat/conversations?limit=999")
        assert resp.status_code == 422

    def test_list_limit_zero(self, client):
        resp = client.get("/api/chat/conversations?limit=0")
        assert resp.status_code == 422


# ===========================================================================
# Conversations — GET /chat/conversations/search
# ===========================================================================

class TestConversationsSearch:
    def test_search_empty_query(self, client):
        resp = client.get("/api/chat/conversations/search?q=")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_no_results(self, client):
        resp = client.get("/api/chat/conversations/search?q=nonexistent12345")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_finds_content(self, client, svc):
        conv_id = svc.conversations.create_conversation()
        svc.conversations.add_message(conv_id, "user", "tell me about quantum physics")
        resp = client.get("/api/chat/conversations/search?q=quantum")
        assert resp.status_code == 200


# ===========================================================================
# Conversations — GET /chat/conversations/{id}
# ===========================================================================

class TestConversationGet:
    def test_get_nonexistent(self, client):
        resp = client.get("/api/chat/conversations/nonexistent-id")
        assert resp.status_code == 404

    def test_get_existing(self, client, svc):
        conv_id = svc.conversations.create_conversation()
        svc.conversations.add_message(conv_id, "user", "hello")
        svc.conversations.add_message(conv_id, "assistant", "Hi there!")
        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conv_id
        assert len(data["messages"]) == 2


# ===========================================================================
# Conversations — PATCH /chat/conversations/{id} (rename)
# ===========================================================================

class TestConversationRename:
    def test_rename_nonexistent(self, client):
        resp = client.patch(
            "/api/chat/conversations/nonexistent-id",
            json={"title": "New Title"},
        )
        assert resp.status_code == 404

    def test_rename_success(self, client, svc):
        conv_id = svc.conversations.create_conversation()
        resp = client.patch(
            f"/api/chat/conversations/{conv_id}",
            json={"title": "My Conversation"},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "My Conversation"

    def test_rename_empty_title_rejected(self, client, svc):
        conv_id = svc.conversations.create_conversation()
        resp = client.patch(
            f"/api/chat/conversations/{conv_id}",
            json={"title": ""},
        )
        assert resp.status_code == 422


# ===========================================================================
# Conversations — DELETE /chat/conversations/{id}
# ===========================================================================

class TestConversationDelete:
    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/chat/conversations/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_success(self, client, svc):
        conv_id = svc.conversations.create_conversation()
        resp = client.delete(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's actually gone
        resp2 = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp2.status_code == 404


# ===========================================================================
# User Facts — CRUD
# ===========================================================================

class TestUserFacts:
    def test_list_empty(self, client):
        resp = client.get("/api/chat/facts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_fact(self, client):
        resp = client.post("/api/chat/facts", json={
            "key": "name",
            "value": "Alice",
        })
        assert resp.status_code == 200
        assert resp.json()["key"] == "name"

    def test_create_and_list(self, client):
        client.post("/api/chat/facts", json={"key": "city", "value": "Boston"})
        resp = client.get("/api/chat/facts")
        assert resp.status_code == 200
        facts = resp.json()
        assert any(f["key"] == "city" and f["value"] == "Boston" for f in facts)

    def test_delete_fact(self, client):
        client.post("/api/chat/facts", json={"key": "temp_fact", "value": "temp_value"})
        resp = client.delete("/api/chat/facts/temp_fact")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_fact(self, client):
        resp = client.delete("/api/chat/facts/nonexistent_key")
        assert resp.status_code == 404

    def test_create_with_source(self, client):
        resp = client.post("/api/chat/facts", json={
            "key": "role",
            "value": "engineer",
            "source": "user",
        })
        assert resp.status_code == 200

    def test_create_with_category(self, client):
        resp = client.post("/api/chat/facts", json={
            "key": "hobby",
            "value": "hiking",
            "category": "interest",
        })
        assert resp.status_code == 200


# ===========================================================================
# Messages search — GET /chat/messages/search
# ===========================================================================

class TestMessagesSearch:
    def test_search_empty_query(self, client):
        resp = client.get("/api/chat/messages/search?q=")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_no_results(self, client):
        resp = client.get("/api/chat/messages/search?q=nonexistent12345")
        assert resp.status_code == 200


# ===========================================================================
# Helpers
# ===========================================================================

async def _mock_think_gen(answer: str, conv_id: str):
    """Mock think() generator that yields token + done events."""
    yield StreamEvent(type=EventType.THINKING, data={"stage": "loading_context"})
    yield StreamEvent(type=EventType.TOKEN, data={"text": answer})
    yield StreamEvent(
        type=EventType.DONE,
        data={"conversation_id": conv_id, "lessons_used": 0, "kg_facts_used": 0, "reflexions_used": 0},
    )


async def _mock_think_error(message: str):
    """Mock think() generator that yields an error event."""
    yield StreamEvent(type=EventType.ERROR, data={"message": message})
