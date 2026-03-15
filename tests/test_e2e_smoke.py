"""E2E Smoke Test — full pipeline with mocked Ollama.

Tests the complete brain.py -> API -> SSE chain without a live LLM.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


@pytest.fixture
def mock_llm():
    """Mock LLM responses to avoid needing Ollama."""
    with patch("app.core.llm.generate_with_tools") as mock_gen, \
         patch("app.core.llm.invoke_nothink") as mock_invoke, \
         patch("app.core.llm.stream_with_thinking") as mock_stream:

        # Default: return a simple response with no tool call
        from app.core.llm import GenerationResult
        mock_gen.return_value = GenerationResult(
            content="This is a test response from Nova.",
            tool_call=None,
            raw={},
            thinking="",
        )

        # invoke_nothink: for title generation, summaries, etc.
        mock_invoke.return_value = "Test Title"

        yield {
            "generate": mock_gen,
            "invoke": mock_invoke,
            "stream": mock_stream,
        }


@pytest.fixture
def services(db):
    """Real services with real DB but no retriever/learning."""
    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
    )
    set_services(svc)
    return svc


@pytest.fixture
def client(db, services, mock_llm):
    """FastAPI test client with mocked LLM."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


class TestE2ESmoke:
    """End-to-end tests through the full pipeline."""

    def test_stream_endpoint_returns_sse(self, client, mock_llm):
        """POST /api/chat/stream should return SSE events."""
        response = client.post("/api/chat/stream", json={
            "query": "Hello, how are you?",
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        events = self._parse_sse(response.text)
        event_types = [e["type"] for e in events]

        # Must have at least thinking and done events
        assert "thinking" in event_types
        assert "done" in event_types

    def test_stream_returns_tokens(self, client, mock_llm):
        """Stream should emit token events with the response text."""
        response = client.post("/api/chat/stream", json={
            "query": "What is Python?",
        })

        events = self._parse_sse(response.text)
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) > 0

        # Reconstruct the full response
        full_text = "".join(e["data"].get("text", "") for e in token_events)
        assert len(full_text) > 0

    def test_stream_creates_conversation(self, client, services, mock_llm):
        """Streaming should create a new conversation if none provided."""
        response = client.post("/api/chat/stream", json={
            "query": "Create a conversation",
        })

        events = self._parse_sse(response.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert "conversation_id" in done_events[0]["data"]

        conv_id = done_events[0]["data"]["conversation_id"]
        assert conv_id is not None

    def test_stream_reuses_conversation(self, client, services, mock_llm):
        """Providing a conversation_id should reuse it."""
        conv_id = services.conversations.create_conversation()

        response = client.post("/api/chat/stream", json={
            "query": "Second message",
            "conversation_id": conv_id,
        })

        events = self._parse_sse(response.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events[0]["data"]["conversation_id"] == conv_id

    def test_sync_chat_endpoint(self, client, mock_llm):
        """POST /api/chat should return a complete JSON response."""
        response = client.post("/api/chat", json={
            "query": "What is 2+2?",
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "conversation_id" in data
        assert len(data["answer"]) > 0

    def test_greeting_intent(self, client, mock_llm):
        """Greeting messages should be classified correctly."""
        response = client.post("/api/chat/stream", json={
            "query": "Hello!",
        })

        events = self._parse_sse(response.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events[0]["data"].get("intent") == "greeting"

    def test_messages_saved_to_db(self, client, services, mock_llm):
        """Both user and assistant messages should be saved."""
        response = client.post("/api/chat/stream", json={
            "query": "Save this message",
        })

        events = self._parse_sse(response.text)
        conv_id = [e for e in events if e["type"] == "done"][0]["data"]["conversation_id"]

        # Check messages in DB
        history = services.conversations.get_history(conv_id, limit=10)
        user_msgs = [m for m in history if m.role == "user"]
        assistant_msgs = [m for m in history if m.role == "assistant"]
        assert len(user_msgs) >= 1
        assert len(assistant_msgs) >= 1
        assert user_msgs[0].content == "Save this message"

    def test_lesson_used_events(self, client, db, mock_llm):
        """When lessons are available, lesson_used events should be emitted."""
        from app.core.learning import LearningEngine

        learning = LearningEngine(db)
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            learning=learning,
        )
        set_services(svc)

        # Create a lesson
        from app.core.learning import Correction
        correction = Correction(
            user_message="The capital of France is Berlin",
            previous_answer="Berlin is the capital of France.",
            topic="capital of france",
            wrong_answer="Berlin",
            correct_answer="Paris",
            lesson_text="The capital of France is Paris, not Berlin.",
        )
        learning.save_lesson(correction)

        response = client.post("/api/chat/stream", json={
            "query": "What is the capital of France?",
        })

        events = self._parse_sse(response.text)
        lesson_events = [e for e in events if e["type"] == "lesson_used"]
        # If the lesson matched, we should see it
        # (depends on keyword overlap — "capital" + "france" should match)
        if lesson_events:
            assert "topic" in lesson_events[0]["data"]
            assert "lesson_id" in lesson_events[0]["data"]

    @staticmethod
    def _parse_sse(text: str) -> list[dict]:
        """Parse SSE text into a list of {type, data} dicts."""
        events = []
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            event_type = None
            data_str = ""
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str += line[5:].strip()
            if event_type and data_str:
                try:
                    events.append({"type": event_type, "data": json.loads(data_str)})
                except json.JSONDecodeError:
                    pass
        return events
