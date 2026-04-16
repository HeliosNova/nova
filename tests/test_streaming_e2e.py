"""Item 58: SSE endpoint integration test.

Uses FastAPI TestClient to POST to /chat/stream, validates SSE format
with proper `data:` lines. Mocks the LLM to return a simple response.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.llm import GenerationResult, StreamChunk, _strip_think_tags, _extract_tool_calls
from app.core.memory import ConversationStore, UserFactStore


def _setup_stream_mock(mock_llm, content="Hello world"):
    """Configure mock_llm with proper async generator for stream_with_thinking."""
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
    mock_llm.ToolCall = type("ToolCall", (), {})


@pytest.fixture
def client(db):
    """FastAPI test client with minimal services."""
    import importlib
    import app.config
    import app.auth

    importlib.reload(app.config)
    importlib.reload(app.auth)

    from fastapi.testclient import TestClient
    from app.main import app, _rate_limit_requests

    _rate_limit_requests.clear()

    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
    )
    set_services(svc)
    return TestClient(app)


class TestSSEStreaming:
    """POST /api/chat/stream returns valid SSE format."""

    def test_stream_returns_event_stream_content_type(self, client):
        """Response content-type should be text/event-stream."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm)

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_has_data_lines(self, client):
        """SSE response should contain event: and data: lines."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm)

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            body = resp.text

            assert "event:" in body
            assert "data:" in body

    def test_stream_ends_with_done(self, client):
        """SSE response should end with data: [DONE]."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm)

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert "data: [DONE]" in resp.text

    def test_stream_contains_thinking_event(self, client):
        """SSE response should contain at least one 'thinking' event."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm, "Hi there!")

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert "event: thinking" in resp.text

    def test_stream_contains_token_events(self, client):
        """SSE response should contain token events with text data."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm, "Hello from Nova!")

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert "event: token" in resp.text

    def test_stream_contains_done_event(self, client):
        """SSE response should contain a done event with conversation_id."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm, "Hello!")

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            assert "event: done" in resp.text
            # Extract done data
            for line in resp.text.split("\n"):
                if line.startswith("data:") and "conversation_id" in line:
                    data = json.loads(line[5:].strip())
                    assert "conversation_id" in data
                    break

    def test_stream_sse_data_is_valid_json(self, client):
        """Each data: line should contain valid JSON."""
        with patch("app.core.brain.llm") as mock_llm:
            _setup_stream_mock(mock_llm, "Test response")

            resp = client.post("/api/chat/stream", json={"query": "hello"})
            for line in resp.text.split("\n"):
                if line.startswith("data:"):
                    content = line[5:].strip()
                    if content == "[DONE]":
                        continue
                    # Should be valid JSON
                    data = json.loads(content)
                    assert isinstance(data, dict)
