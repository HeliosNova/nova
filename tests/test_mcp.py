"""Tests for MCP (Model Context Protocol) tool wrapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.base import ToolResult, ErrorCategory


async def test_mcp_tool_name_prefix():
    """MCP tools are prefixed with 'mcp_' for namespacing."""
    from app.tools.mcp import MCPTool
    mock_client = MagicMock()
    tool_spec = {"name": "test_tool", "description": "A test tool"}
    tool = MCPTool(client=mock_client, tool_spec=tool_spec, server_name="my_server")
    assert tool.name == "mcp_test_tool"
    assert "test tool" in tool.description.lower()


async def test_mcp_tool_execute_no_client():
    """MCPTool with dead client returns error."""
    from app.tools.mcp import MCPTool
    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(side_effect=Exception("Connection dead"))
    tool_spec = {"name": "broken", "description": "broken tool"}
    tool = MCPTool(client=mock_client, tool_spec=tool_spec)
    result = await tool.execute(action="test")
    assert not result.success


def test_mcp_error_classification():
    """MCP error classification maps known patterns correctly."""
    from app.tools.mcp import _classify_mcp_error
    assert _classify_mcp_error("Request timed out") == ErrorCategory.TRANSIENT
    assert _classify_mcp_error("Invalid parameter") == ErrorCategory.VALIDATION
    assert _classify_mcp_error("Permission denied") == ErrorCategory.PERMISSION
    assert _classify_mcp_error("Tool not found") == ErrorCategory.NOT_FOUND
    assert _classify_mcp_error("Something unexpected") == ErrorCategory.INTERNAL
