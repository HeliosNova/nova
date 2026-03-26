"""Tests for the delegate tool — multi-agent task delegation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.tools.base import ToolResult, ErrorCategory


@pytest.fixture
def delegate_tool():
    from app.tools.delegate import DelegateTool
    return DelegateTool()


async def test_delegate_requires_task(delegate_tool):
    """Delegate tool returns VALIDATION error when no task is provided."""
    result = await delegate_tool.execute(task="", role="researcher")
    assert not result.success
    assert result.error_category == ErrorCategory.VALIDATION


async def test_delegate_disabled(delegate_tool, monkeypatch):
    """Delegate tool returns PERMISSION error when delegation is disabled."""
    monkeypatch.setenv("ENABLE_DELEGATION", "false")
    from app.config import reset_config
    reset_config()
    result = await delegate_tool.execute(task="research something", role="researcher")
    assert not result.success
    assert result.error_category == ErrorCategory.PERMISSION


async def test_delegate_depth_limit(delegate_tool, monkeypatch):
    """Delegate tool respects MAX_DELEGATION_DEPTH."""
    monkeypatch.setenv("ENABLE_DELEGATION", "true")
    monkeypatch.setenv("MAX_DELEGATION_DEPTH", "0")
    from app.config import reset_config
    reset_config()
    result = await delegate_tool.execute(task="research something", role="researcher")
    assert not result.success
