"""Tests for multi-agent delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.schema import EventType, StreamEvent
from app.tools.delegate import (
    DelegateTool,
    get_delegation_depth,
    set_delegation_depth,
)


@pytest.fixture
def tool():
    return DelegateTool()


@pytest.fixture(autouse=True)
def reset_depth():
    set_delegation_depth(0)
    yield
    set_delegation_depth(0)


class TestDelegateToolBasics:
    @pytest.mark.asyncio
    async def test_no_task(self, tool):
        r = await tool.execute()
        assert not r.success
        assert "No task" in r.error

    @pytest.mark.asyncio
    async def test_disabled(self, tool):
        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": False,
                                  "MAX_DELEGATION_DEPTH": 1})()):
            r = await tool.execute(task="test")
            assert not r.success
            assert "disabled" in r.error.lower()

    @pytest.mark.asyncio
    async def test_max_depth_reached(self, tool):
        set_delegation_depth(1)
        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True,
                                  "MAX_DELEGATION_DEPTH": 1})()):
            r = await tool.execute(task="test")
            assert not r.success
            assert "depth" in r.error.lower()


class TestDelegateToolExecution:
    @pytest.mark.asyncio
    async def test_successful_delegation(self, tool):
        async def fake_think(**kwargs):
            yield StreamEvent(type=EventType.THINKING, data={"stage": "generating"})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "The weather "})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "is sunny."})
            yield StreamEvent(type=EventType.DONE, data={"conversation_id": "eph"})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True,
                                  "MAX_DELEGATION_DEPTH": 1})()):
            with patch("app.core.brain.think", side_effect=fake_think):
                r = await tool.execute(task="What is the weather?", role="researcher")

        assert r.success
        assert "sunny" in r.output

    @pytest.mark.asyncio
    async def test_delegation_error_handling(self, tool):
        async def failing_think(**kwargs):
            yield StreamEvent(type=EventType.ERROR, data={"message": "LLM down"})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True,
                                  "MAX_DELEGATION_DEPTH": 1})()):
            with patch("app.core.brain.think", side_effect=failing_think):
                r = await tool.execute(task="test")

        assert not r.success
        assert "LLM down" in r.error

    @pytest.mark.asyncio
    async def test_depth_restored_after_delegation(self, tool):
        async def fake_think(**kwargs):
            # Verify depth was incremented during execution
            assert get_delegation_depth() == 1
            yield StreamEvent(type=EventType.TOKEN, data={"text": "ok"})
            yield StreamEvent(type=EventType.DONE, data={})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True,
                                  "MAX_DELEGATION_DEPTH": 2})()):
            with patch("app.core.brain.think", side_effect=fake_think):
                await tool.execute(task="test")

        assert get_delegation_depth() == 0

    @pytest.mark.asyncio
    async def test_output_truncation(self, tool):
        long_text = "x" * 5000

        async def verbose_think(**kwargs):
            yield StreamEvent(type=EventType.TOKEN, data={"text": long_text})
            yield StreamEvent(type=EventType.DONE, data={})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True,
                                  "MAX_DELEGATION_DEPTH": 1})()):
            with patch("app.core.brain.think", side_effect=verbose_think):
                r = await tool.execute(task="test")

        assert r.success
        assert len(r.output) <= 3100  # 3000 + truncation notice
        assert "truncated" in r.output
