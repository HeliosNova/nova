"""Tests for individual tool execute() methods with mocked subprocess/httpx.

Tests:
- DynamicTool (custom tool) uses safe env
- WebSearchTool handles httpx timeout gracefully
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.tools.base import ToolResult


# ===========================================================================
# Shell Exec — Platform-Aware Command Construction
# ===========================================================================

class TestWebSearchTimeout:
    """Verify web search handles native_search failures gracefully.

    The web_search tool now delegates to app.tools.native_search.search()
    which internally handles SearXNG/DDG/Wikipedia failover. Errors at the
    underlying transport layer raise from native_search; the tool wrapper
    surfaces them as a retriable ToolResult.
    """

    @pytest.mark.asyncio
    async def test_web_search_timeout(self):
        """native_search raising surfaces as upstream failure."""
        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        with patch("app.tools.native_search.search",
                   AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
            result = await tool.execute(query="test query")
        assert not result.success
        assert "search failed" in (result.error or "").lower()
        assert result.retriable is True

    @pytest.mark.asyncio
    async def test_web_search_connect_error(self):
        """native_search ConnectError surfaces as upstream failure."""
        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        with patch("app.tools.native_search.search",
                   AsyncMock(side_effect=httpx.ConnectError("refused"))):
            result = await tool.execute(query="test query")
        assert not result.success
        assert "search failed" in (result.error or "").lower()
        assert result.retriable is True

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self):
        """Empty query returns validation error."""
        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        result = await tool.execute(query="")
        assert not result.success
        assert "No query" in result.error
