"""Tests for the brain.py generation/tool loop.

Covers tool dispatch, MAX_TOOL_ROUNDS exhaustion, per-conversation
tool result caching, and transient failure retry.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services, think
from app.core.llm import GenerationResult, ToolCall
from app.core.memory import ConversationStore, UserFactStore
from app.schema import EventType
from app.tools.base import BaseTool, ErrorCategory, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Helpers — lightweight mock tools
# ---------------------------------------------------------------------------


class MockSearchTool(BaseTool):
    """A fake web_search tool that records calls."""

    name = "web_search"
    description = "Search the web"
    parameters = "query: str"

    def __init__(self):
        self.call_log: list[dict] = []
        self._response = "Search results for: {query}"

    async def execute(self, **kwargs) -> ToolResult:
        self.call_log.append(kwargs)
        output = self._response.format(**kwargs) if "query" in kwargs else "results"
        return ToolResult(output=output, success=True)


class TransientFailTool(BaseTool):
    """A tool that fails with a transient error on first call, succeeds on retry."""

    name = "flaky_tool"
    description = "A flaky tool"
    parameters = "query: str"

    def __init__(self):
        self.call_count = 0

    async def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        if self.call_count == 1:
            return ToolResult(
                output="",
                success=False,
                error="Connection timed out",
                retriable=True,
                error_category=ErrorCategory.TRANSIENT,
            )
        return ToolResult(output="Success on retry", success=True)


class AlwaysSucceedTool(BaseTool):
    """A simple tool that always succeeds."""

    name = "calculator"
    description = "Calculate math"
    parameters = "expression: str"

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="4", success=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TestGenerationLoop:
    """Tests for the generation/tool loop in brain.py."""

    @pytest.fixture
    def services(self, db):
        """Minimal services with a tool registry."""
        registry = ToolRegistry()
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            tool_registry=registry,
        )
        set_services(svc)
        return svc

    # ------------------------------------------------------------------
    # 1. Tool dispatch — verify tool's execute() is called with correct args
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tool_dispatch(self, services):
        """Mock LLM to return a web_search tool call, verify the tool executes with correct args."""
        search_tool = MockSearchTool()
        services.tool_registry.register(search_tool)

        with patch("app.core.brain.llm") as mock_llm:
            # First call: LLM returns a tool call
            tool_gen = GenerationResult(
                content='{"tool": "web_search", "args": {"query": "latest news"}}',
                tool_calls=[ToolCall(tool="web_search", args={"query": "latest news"})],
                raw={},
            )
            # Second call: LLM returns final text (no tool call)
            final_gen = GenerationResult(
                content="Here are the latest news results.",
                tool_calls=[],
                raw={},
            )
            mock_llm.generate_with_tools = AsyncMock(
                side_effect=[tool_gen, final_gen]
            )
            mock_llm.invoke_nothink = AsyncMock(return_value="News Query")
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.get_provider = MagicMock()
            mock_llm.get_provider.return_value.capabilities.needs_emphatic_prompts = False

            events = []
            async for event in think("What's the latest news?"):
                events.append(event)

        # The search tool should have been called exactly once
        assert len(search_tool.call_log) == 1
        assert search_tool.call_log[0] == {"query": "latest news"}

        # Should have tool use events (executing + complete)
        tool_events = [e for e in events if e.type == EventType.TOOL_USE]
        assert len(tool_events) == 2
        assert tool_events[0].data["tool"] == "web_search"
        assert tool_events[0].data["status"] == "executing"
        assert tool_events[1].data["status"] == "complete"

    # ------------------------------------------------------------------
    # 2. Max rounds exhaustion — verify fallback after MAX_TOOL_ROUNDS
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_max_rounds_exhaustion(self, services, monkeypatch):
        """Mock LLM to always return tool calls, verify it stops after MAX_TOOL_ROUNDS."""
        search_tool = MockSearchTool()
        services.tool_registry.register(search_tool)

        # Set MAX_TOOL_ROUNDS to 2 for fast testing
        monkeypatch.setattr("app.config.config.MAX_TOOL_ROUNDS", 2)

        with patch("app.core.brain.llm") as mock_llm:
            # LLM always returns a tool call, never plain text
            tool_gen = GenerationResult(
                content='{"tool": "web_search", "args": {"query": "test"}}',
                tool_calls=[ToolCall(tool="web_search", args={"query": "test"})],
                raw={},
            )
            mock_llm.generate_with_tools = AsyncMock(return_value=tool_gen)
            mock_llm.invoke_nothink = AsyncMock(return_value="Test")
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.get_provider = MagicMock()
            mock_llm.get_provider.return_value.capabilities.needs_emphatic_prompts = False

            events = []
            async for event in think("Search forever"):
                events.append(event)

        # Should have exactly 2 rounds of tool calls
        tool_exec_events = [
            e for e in events
            if e.type == EventType.TOOL_USE and e.data.get("status") == "executing"
        ]
        assert len(tool_exec_events) == 2

        # Final content is streamed as TOKEN events — collect them
        # After exhausting tool rounds, brain does a synthesis LLM call
        # which returns the mock's response ("Test")
        token_text = "".join(
            e.data.get("text", "")
            for e in events
            if e.type == EventType.TOKEN
        )
        # The synthesis call may return the mock response or a canned message
        assert len(token_text) > 0

    # ------------------------------------------------------------------
    # 3. Tool result caching — same tool+args only executes once
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tool_result_caching(self, services):
        """Call the same tool with same args twice in a single think() call, verify only executed once."""
        search_tool = MockSearchTool()
        services.tool_registry.register(search_tool)

        with patch("app.core.brain.llm") as mock_llm:
            # Round 1: LLM requests web_search with same args
            tool_gen_1 = GenerationResult(
                content='{"tool": "web_search", "args": {"query": "bitcoin price"}}',
                tool_calls=[ToolCall(tool="web_search", args={"query": "bitcoin price"})],
                raw={},
            )
            # Round 2: LLM requests web_search again with same args
            tool_gen_2 = GenerationResult(
                content='{"tool": "web_search", "args": {"query": "bitcoin price"}}',
                tool_calls=[ToolCall(tool="web_search", args={"query": "bitcoin price"})],
                raw={},
            )
            # Round 3: LLM returns final answer
            final_gen = GenerationResult(
                content="Bitcoin is at $67,000.",
                tool_calls=[],
                raw={},
            )

            mock_llm.generate_with_tools = AsyncMock(
                side_effect=[tool_gen_1, tool_gen_2, final_gen]
            )
            mock_llm.invoke_nothink = AsyncMock(return_value="BTC Price")
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.get_provider = MagicMock()
            mock_llm.get_provider.return_value.capabilities.needs_emphatic_prompts = False

            events = []
            async for event in think("What is the bitcoin price?"):
                events.append(event)

        # The tool should have been executed only ONCE (second call used cache)
        assert len(search_tool.call_log) == 1

        # But there should be 2 rounds of tool use events (both rounds dispatched)
        tool_exec_events = [
            e for e in events
            if e.type == EventType.TOOL_USE and e.data.get("status") == "executing"
        ]
        assert len(tool_exec_events) == 2

    # ------------------------------------------------------------------
    # 4. Transient retry — tool fails with retriable=True, succeeds on retry
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_transient_retry(self, services):
        """Mock a tool to fail with TRANSIENT on first call, succeed on second. Verify retry happens."""
        flaky_tool = TransientFailTool()
        services.tool_registry.register(flaky_tool)

        with patch("app.core.brain.llm") as mock_llm:
            # LLM requests flaky_tool
            tool_gen = GenerationResult(
                content='{"tool": "flaky_tool", "args": {"query": "test"}}',
                tool_calls=[ToolCall(tool="flaky_tool", args={"query": "test"})],
                raw={},
            )
            # After retry succeeds, LLM returns final answer
            final_gen = GenerationResult(
                content="Got the result after retry.",
                tool_calls=[],
                raw={},
            )

            mock_llm.generate_with_tools = AsyncMock(
                side_effect=[tool_gen, final_gen]
            )
            mock_llm.invoke_nothink = AsyncMock(return_value="Flaky Test")
            mock_llm._strip_think_tags = lambda x: x
            mock_llm.get_provider = MagicMock()
            mock_llm.get_provider.return_value.capabilities.needs_emphatic_prompts = False

            events = []
            async for event in think("Test the flaky tool"):
                events.append(event)

        # The tool should have been called twice (first fail + retry)
        assert flaky_tool.call_count == 2

        # The tool result event should show success (retry succeeded)
        tool_complete = [
            e for e in events
            if e.type == EventType.TOOL_USE and e.data.get("status") == "complete"
        ]
        assert len(tool_complete) == 1
        assert "Success on retry" in tool_complete[0].data.get("result", "")
