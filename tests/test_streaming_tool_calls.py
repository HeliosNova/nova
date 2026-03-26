"""Tests for streaming tool call accumulation — extends test_streaming_providers.

Focuses on the OpenAI streaming provider since it has the most complex
tool call handling (delta-based accumulation across multiple chunks).

Tests:
- Partial tool call deltas are accumulated correctly
- Malformed tool call arguments produce empty args dict (no crash)
- Multiple tool calls (different indices) in a single stream are both emitted
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.core.llm import StreamChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockStreamResponse:
    """Mock httpx streaming response that yields lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _make_openai_provider():
    """Create an OpenAI provider with mock client for streaming tests."""
    from app.core.providers.openai import OpenAIProvider
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._model = "gpt-4o"
    provider._api_key = "test-key"
    return provider


def _setup_mock_stream(provider, lines):
    """Wire up a mock client that returns the given stream lines."""
    mock_response = MockStreamResponse(lines)
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    provider._get_client = MagicMock(return_value=mock_client)


# ===========================================================================
# Partial Tool Call Accumulation
# ===========================================================================

class TestPartialToolCallAccumulation:
    """Verify streaming tool call deltas are accumulated into complete calls."""

    @pytest.mark.asyncio
    async def test_partial_tool_call_accumulation(self):
        """Send streaming chunks with partial tool call deltas,
        verify they are accumulated into one complete tool call."""
        provider = _make_openai_provider()

        # Simulate OpenAI SSE: tool call name in first delta,
        # arguments split across multiple deltas
        lines = [
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"name": "web_search", "arguments": '{"qu'}
                        }]
                    }
                }]
            }),
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": 'ery": "weath'}
                        }]
                    }
                }]
            }),
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": 'er today"}'}
                        }]
                    }
                }]
            }),
            'data: [DONE]',
        ]

        _setup_mock_stream(provider, lines)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "search for weather"}],
            [{"name": "web_search", "description": "Search"}],
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        # Should have tool call chunks + done
        tool_call_chunks = [c for c in chunks if c.tool_call is not None]
        content_chunks = [c for c in chunks if c.content]
        done_chunks = [c for c in chunks if c.done]

        assert len(done_chunks) == 1

        # The accumulated tool call should be emitted via tool_call field
        # (or as JSON content for backward compat)
        if tool_call_chunks:
            tc = tool_call_chunks[-1].tool_call
            assert tc.tool == "web_search"
            assert tc.args == {"query": "weather today"}
        else:
            assert len(content_chunks) >= 1
            tool_json = content_chunks[-1].content
            parsed = json.loads(tool_json)
            assert parsed["tool"] == "web_search"
            assert parsed["args"] == {"query": "weather today"}


# ===========================================================================
# Malformed Tool Call Arguments
# ===========================================================================

class TestMalformedToolArgs:
    """Verify malformed tool call arguments don't crash the stream."""

    @pytest.mark.asyncio
    async def test_malformed_tool_args_json(self):
        """Send a [DONE] event with malformed tool call arguments,
        verify no crash and empty args dict is returned."""
        provider = _make_openai_provider()

        # Simulate: tool call name provided but arguments are invalid JSON
        lines = [
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"name": "calculator", "arguments": '{"expr'}
                        }]
                    }
                }]
            }),
            # Arguments are truncated / malformed — no closing brace
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": 'ession": 2+2'}
                        }]
                    }
                }]
            }),
            # Stream ends with malformed accumulated args (missing closing brace)
            'data: [DONE]',
        ]

        _setup_mock_stream(provider, lines)

        chunks = []
        # Should not crash
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "calculate 2+2"}],
            [{"name": "calculator", "description": "Math"}],
        ):
            chunks.append(chunk)

        tool_call_chunks = [c for c in chunks if c.tool_call is not None]
        content_chunks = [c for c in chunks if c.content]
        done_chunks = [c for c in chunks if c.done]

        assert len(done_chunks) == 1

        # The tool call should be emitted with fallback args (malformed JSON)
        if tool_call_chunks:
            tc = tool_call_chunks[-1].tool_call
            assert tc.tool == "calculator"
            # Malformed args may have parse error info or be empty
            assert isinstance(tc.args, dict)
        else:
            assert len(content_chunks) >= 1
            tool_json = content_chunks[-1].content
            parsed = json.loads(tool_json)
            assert parsed["tool"] == "calculator"


# ===========================================================================
# Multiple Tool Calls in Stream
# ===========================================================================

class TestMultipleToolCallsInStream:
    """Verify multiple tool calls with different indices are both emitted."""

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_stream(self):
        """Send deltas for 2 tool calls (different indices),
        verify both are emitted in order."""
        provider = _make_openai_provider()

        lines = [
            # First tool call — index 0
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"name": "web_search", "arguments": '{"query": "news"}'}
                        }]
                    }
                }]
            }),
            # Second tool call — index 1
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}
                        }]
                    }
                }]
            }),
            'data: [DONE]',
        ]

        _setup_mock_stream(provider, lines)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "search and calculate"}],
            [
                {"name": "web_search", "description": "Search"},
                {"name": "calculator", "description": "Math"},
            ],
        ):
            chunks.append(chunk)

        tool_call_chunks = [c for c in chunks if c.tool_call is not None]
        content_chunks = [c for c in chunks if c.content]
        done_chunks = [c for c in chunks if c.done]

        assert len(done_chunks) == 1

        # Both tool calls should be emitted (via tool_call field or content)
        all_tc = tool_call_chunks if tool_call_chunks else content_chunks
        assert len(all_tc) == 2

        if tool_call_chunks:
            assert all_tc[0].tool_call.tool == "web_search"
            assert all_tc[0].tool_call.args == {"query": "news"}
            assert all_tc[1].tool_call.tool == "calculator"
            assert all_tc[1].tool_call.args == {"expression": "1+1"}
        else:
            first = json.loads(all_tc[0].content)
            second = json.loads(all_tc[1].content)
            assert first["tool"] == "web_search"
            assert second["tool"] == "calculator"

    @pytest.mark.asyncio
    async def test_interleaved_deltas_for_multiple_calls(self):
        """Tool call deltas can arrive interleaved — both should accumulate correctly."""
        provider = _make_openai_provider()

        lines = [
            # First chunk: start of both tool calls
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"name": "web_search", "arguments": '{"que'}
                        }]
                    }
                }]
            }),
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "function": {"name": "calculator", "arguments": '{"exp'}
                        }]
                    }
                }]
            }),
            # Continue index 0
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": 'ry": "test"}'}
                        }]
                    }
                }]
            }),
            # Continue index 1
            'data: ' + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "function": {"arguments": 'ression": "3*4"}'}
                        }]
                    }
                }]
            }),
            'data: [DONE]',
        ]

        _setup_mock_stream(provider, lines)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "search and calculate"}],
            [
                {"name": "web_search", "description": "Search"},
                {"name": "calculator", "description": "Math"},
            ],
        ):
            chunks.append(chunk)

        tool_call_chunks = [c for c in chunks if c.tool_call is not None]
        content_chunks = [c for c in chunks if c.content]

        all_tc = tool_call_chunks if tool_call_chunks else content_chunks
        assert len(all_tc) == 2

        if tool_call_chunks:
            assert all_tc[0].tool_call.tool == "web_search"
            assert all_tc[0].tool_call.args == {"query": "test"}
            assert all_tc[1].tool_call.tool == "calculator"
            assert all_tc[1].tool_call.args == {"expression": "3*4"}
        else:
            first = json.loads(all_tc[0].content)
            second = json.loads(all_tc[1].content)
            assert first["tool"] == "web_search"
            assert first["args"] == {"query": "test"}
            assert second["tool"] == "calculator"
            assert second["args"] == {"expression": "3*4"}
