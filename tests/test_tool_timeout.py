"""Tool timeout tests — verify tools exceeding TOOL_TIMEOUT are cancelled.

Tests:
- Tool exceeding timeout is properly cancelled
- Timeout error message has correct format
- Normal tools complete within timeout
- Timeout result is marked as retriable
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import _execute_tool, Services, set_services
from app.core.memory import ConversationStore, UserFactStore
from app.tools.base import BaseTool, ToolResult, ToolRegistry


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class SlowTool(BaseTool):
    """A tool that takes a very long time to execute."""
    name = "slow_tool"
    description = "A tool that sleeps forever"
    parameters = "duration: int"

    async def execute(self, *, duration: int = 999, **kwargs) -> ToolResult:
        await asyncio.sleep(duration)
        return ToolResult(output="Should never reach here", success=True)


class FastTool(BaseTool):
    """A tool that completes instantly."""
    name = "fast_tool"
    description = "A tool that returns immediately"
    parameters = ""

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="Done instantly", success=True)


class FailingTool(BaseTool):
    """A tool that raises an exception."""
    name = "failing_tool"
    description = "A tool that crashes"
    parameters = ""

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Tool crashed unexpectedly")


# ===========================================================================
# Timeout Tests
# ===========================================================================

class TestToolTimeout:
    """Test that _execute_tool enforces TOOL_TIMEOUT."""

    @pytest.fixture
    def services_with_tools(self, db):
        """Set up Services with a ToolRegistry containing test tools."""
        registry = ToolRegistry()
        registry.register(SlowTool())
        registry.register(FastTool())
        registry.register(FailingTool())

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            tool_registry=registry,
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_slow_tool_times_out(self, services_with_tools):
        """A tool exceeding TOOL_TIMEOUT should be cancelled and return error."""
        with patch("app.core.brain.config") as mock_config:
            mock_config.TOOL_TIMEOUT = 0.1  # 100ms timeout

            output, result = await _execute_tool("slow_tool", {"duration": 999})

        assert "[Tool error: slow_tool]" in output
        assert "Timed out" in output
        assert result is not None
        assert result.success is False
        assert result.retriable is True

    @pytest.mark.asyncio
    async def test_fast_tool_completes_within_timeout(self, services_with_tools):
        """A tool that finishes quickly should succeed."""
        with patch("app.core.brain.config") as mock_config:
            mock_config.TOOL_TIMEOUT = 10.0  # 10s timeout — more than enough

            output, result = await _execute_tool("fast_tool", {})

        assert "Done instantly" in output
        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_error_is_retriable(self, services_with_tools):
        """Timeout errors should be marked as retriable."""
        with patch("app.core.brain.config") as mock_config:
            mock_config.TOOL_TIMEOUT = 0.1

            output, result = await _execute_tool("slow_tool", {"duration": 999})

        assert result.retriable is True
        assert "(retriable: yes)" in output

    @pytest.mark.asyncio
    async def test_exception_in_tool_returns_error(self, services_with_tools):
        """A tool that raises an exception should return error, not crash."""
        with patch("app.core.brain.config") as mock_config:
            mock_config.TOOL_TIMEOUT = 10.0

            output, result = await _execute_tool("failing_tool", {})

        assert "[Tool error: failing_tool]" in output
        assert result.success is False
        assert result.retriable is True

    @pytest.mark.asyncio
    async def test_nonexistent_tool_returns_error(self, services_with_tools):
        """Calling a tool that doesn't exist should return error."""
        with patch("app.core.brain.config") as mock_config:
            mock_config.TOOL_TIMEOUT = 10.0

            output, result = await _execute_tool("does_not_exist", {})

        assert "[Tool error: does_not_exist]" in output
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_registry_returns_error(self, db):
        """When no tool_registry is set, _execute_tool should return error."""
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            tool_registry=None,
        )
        set_services(svc)

        output, result = await _execute_tool("any_tool", {})

        assert "[Tool error: any_tool]" in output
        assert "Not yet available" in output


class TestToolTimeoutInBrain:
    """Test tool timeout through the brain.think() pipeline."""

    @pytest.fixture
    def services_with_slow_tool(self, db):
        registry = ToolRegistry()
        registry.register(SlowTool())

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            tool_registry=registry,
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_think_handles_tool_timeout(self, services_with_slow_tool):
        """think() should handle tool timeouts gracefully and continue."""
        from app.core.brain import think
        from app.core.llm import ToolCall
        from app.schema import EventType

        call_count = 0
        with patch("app.core.brain.llm") as mock_llm:
            async def tool_then_answer(msgs, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                result = AsyncMock()
                if call_count == 1:
                    # First call: try to use slow_tool
                    result.content = '{"tool": "slow_tool", "args": {"duration": 999}}'
                    result.tool_calls = [ToolCall(tool="slow_tool", args={"duration": 999})]
                else:
                    # Second call: give final answer after tool timeout
                    result.content = "The tool timed out, but here is my answer."
                    result.tool_calls = []
                return result

            mock_llm.generate_with_tools = AsyncMock(side_effect=tool_then_answer)

            from app.config import get_config
            original_timeout = get_config().TOOL_TIMEOUT
            object.__setattr__(get_config(), "TOOL_TIMEOUT", 0.1)  # 100ms
            try:
                events = []
                async for event in think("Use the slow tool"):
                    events.append(event)
            finally:
                object.__setattr__(get_config(), "TOOL_TIMEOUT", original_timeout)

        event_types = [e.type for e in events]
        assert EventType.DONE in event_types

        # Should have tool use events showing the timeout
        tool_events = [e for e in events if e.type == EventType.TOOL_USE]
        assert len(tool_events) >= 1


class TestToolRegistryTimeout:
    """Test ToolRegistry directly with timeout behavior."""

    @pytest.mark.asyncio
    async def test_registry_execute_with_asyncio_timeout(self):
        """Verify asyncio.wait_for correctly cancels a slow tool."""
        registry = ToolRegistry()
        registry.register(SlowTool())

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                registry.execute_full("slow_tool", {"duration": 999}),
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_registry_fast_tool_no_timeout(self):
        """Fast tool should complete without timeout."""
        registry = ToolRegistry()
        registry.register(FastTool())

        output, result = await asyncio.wait_for(
            registry.execute_full("fast_tool", {}),
            timeout=5.0,
        )
        assert "Done instantly" in output
        assert result.success is True
