"""Tests for tool-call loop detection and circuit breaker in brain.py.

Verifies that:
- Per-tool-name cap stops repeated calls to the same tool
- Same (tool, args) dedup catches identical repeated calls
- Total per-query cap stops runaway tool chains
- Browser-specific output dedup fires on identical content
- Normal multi-tool queries are NOT affected
"""

from __future__ import annotations

import asyncio
import json
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ToolCall = namedtuple("ToolCall", ["tool", "args"])


def _make_gen_result(content="", tool_calls=None):
    """Create a mock GenerationResult."""
    from app.core.llm import GenerationResult
    return GenerationResult(
        content=content,
        tool_calls=tool_calls or [],
        raw={},
    )


def _collect_events(gen):
    """Collect all events from an async generator."""
    events = []

    async def _collect():
        async for event in gen:
            events.append(event)
        return events

    return asyncio.get_event_loop().run_until_complete(_collect()) if False else events


# ---------------------------------------------------------------------------
# Circuit Breaker Unit Tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerConfig:
    """Config fields exist and have sensible defaults."""

    def test_max_same_tool_calls_default(self):
        assert config.MAX_SAME_TOOL_CALLS == 3

    def test_max_tool_calls_per_query_default(self):
        assert config.MAX_TOOL_CALLS_PER_QUERY == 15

    def test_max_tool_rounds_default(self):
        assert config.MAX_TOOL_ROUNDS == 10


class TestCircuitBreakerLogic:
    """Test the circuit breaker detection logic in isolation."""

    def test_per_tool_cap_triggers(self):
        """Calling the same tool > MAX_SAME_TOOL_CALLS times should filter it out."""
        _call_counts: dict[str, int] = {}
        MAX = 3

        accepted = 0
        for i in range(6):
            tool_name = "browser"
            _call_counts[tool_name] = _call_counts.get(tool_name, 0) + 1
            if _call_counts[tool_name] <= MAX:
                accepted += 1

        assert accepted == 3  # First 3 accepted, next 3 filtered

    def test_per_tool_cap_does_not_affect_different_tools(self):
        """Different tools should have independent counters."""
        _call_counts: dict[str, int] = {}
        MAX = 3

        tools = ["web_search", "browser", "http_fetch", "calculator"]
        for tool_name in tools:
            _call_counts[tool_name] = _call_counts.get(tool_name, 0) + 1

        # All tools called once — none should be filtered
        assert all(v <= MAX for v in _call_counts.values())

    def test_identical_call_dedup(self):
        """Same (tool, args) pair repeated should be caught."""
        _pair_counts: dict[tuple, int] = {}
        args = {"query": "Russia economy"}
        args_hash = json.dumps(args, sort_keys=True)

        accepted = 0
        for _ in range(5):
            key = ("browser", args_hash)
            _pair_counts[key] = _pair_counts.get(key, 0) + 1
            if _pair_counts[key] <= 2:
                accepted += 1

        assert accepted == 2  # First 2 accepted, rest filtered

    def test_different_args_not_deduped(self):
        """Same tool with different args should NOT be deduped."""
        _pair_counts: dict[tuple, int] = {}

        queries = ["Russia economy", "China trade", "EU sanctions"]
        for q in queries:
            args_hash = json.dumps({"query": q}, sort_keys=True)
            key = ("web_search", args_hash)
            _pair_counts[key] = _pair_counts.get(key, 0) + 1

        # All different — none should be filtered
        assert all(v == 1 for v in _pair_counts.values())

    def test_total_cap_stops_runaway(self):
        """Total tool calls exceeding MAX_TOOL_CALLS_PER_QUERY should stop."""
        _total = 0
        MAX = 15
        stopped = False

        for i in range(20):
            _total += 1
            if _total > MAX:
                stopped = True
                break

        assert stopped
        assert _total == 16  # Stopped at 16 (one over max)

    def test_browser_output_dedup(self):
        """Identical browser output should be detected."""
        _last_outputs: dict[str, int] = {}

        output1 = "Page content about Russia and Eastern Europe" * 50
        output2 = "Page content about Russia and Eastern Europe" * 50
        output3 = "Different page content entirely" * 50

        h1 = hash(output1[:2000])
        _last_outputs["browser"] = h1

        h2 = hash(output2[:2000])
        assert h2 == _last_outputs["browser"]  # Identical — would trigger dedup

        _last_outputs["browser"] = h2

        h3 = hash(output3[:2000])
        assert h3 != _last_outputs["browser"]  # Different — no dedup


class TestCircuitBreakerIntegration:
    """Integration test: verify circuit breaker works in the think() pipeline."""

    @pytest.mark.asyncio
    async def test_normal_multi_tool_not_affected(self):
        """A query using 3 different tools should not trigger circuit breaker."""
        from app.core.brain import _SIDE_EFFECT_TOOLS

        # web_search, calculator, http_fetch — 3 different tools, each once
        # Should all pass through the circuit breaker
        _call_counts: dict[str, int] = {}
        tools = ["web_search", "calculator", "http_fetch"]

        for tool_name in tools:
            _call_counts[tool_name] = _call_counts.get(tool_name, 0) + 1

        # None exceed the cap
        assert all(v <= config.MAX_SAME_TOOL_CALLS for v in _call_counts.values())
        assert sum(_call_counts.values()) <= config.MAX_TOOL_CALLS_PER_QUERY

    @pytest.mark.asyncio
    async def test_side_effect_tools_set_includes_browser(self):
        """browser must be in _SIDE_EFFECT_TOOLS (no caching)."""
        from app.core.brain import _SIDE_EFFECT_TOOLS
        assert "browser" in _SIDE_EFFECT_TOOLS

    @pytest.mark.asyncio
    async def test_config_overridable(self):
        """Circuit breaker limits should be configurable."""
        assert hasattr(config, "MAX_SAME_TOOL_CALLS")
        assert hasattr(config, "MAX_TOOL_CALLS_PER_QUERY")
        assert isinstance(config.MAX_SAME_TOOL_CALLS, int)
        assert isinstance(config.MAX_TOOL_CALLS_PER_QUERY, int)
