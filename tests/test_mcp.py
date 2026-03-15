"""Tests for MCP (Model Context Protocol) bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.base import ToolResult


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------


class TestMCPTool:
    """Test MCPTool wrapping and execution."""

    def _make_tool(self, client=None, spec=None):
        from app.tools.mcp import MCPTool
        spec = spec or {
            "name": "get_weather",
            "description": "Get current weather",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string"},
                },
                "required": ["city"],
            },
        }
        return MCPTool(client or MagicMock(), spec)

    def test_name_prefixed(self):
        tool = self._make_tool()
        assert tool.name == "mcp_get_weather"

    def test_description(self):
        tool = self._make_tool()
        assert tool.description == "Get current weather"

    def test_parameters_formatted(self):
        tool = self._make_tool()
        assert "city" in tool.parameters
        assert "[required]" in tool.parameters

    def test_missing_name_uses_unknown(self):
        tool = self._make_tool(spec={"description": "test"})
        assert tool.name == "mcp_unknown"

    @pytest.mark.asyncio
    async def test_execute_calls_client(self):
        mock_client = MagicMock()
        # Simulate mcp SDK CallToolResult with content list
        mock_block = MagicMock()
        mock_block.text = "Sunny, 72F"
        mock_result = MagicMock()
        mock_result.content = [mock_block]
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool = self._make_tool(client=mock_client)
        with patch("app.core.access_tiers._tier", return_value="full"):
            result = await tool.execute(city="NYC")

        mock_client.call_tool.assert_called_once_with("get_weather", {"city": "NYC"})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "Sunny" in result.output

    @pytest.mark.asyncio
    async def test_execute_handles_error_gracefully(self):
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=RuntimeError("server crashed"))

        tool = self._make_tool(client=mock_client)
        with patch("app.core.access_tiers._tier", return_value="full"):
            result = await tool.execute(city="NYC")

        assert result.success is False
        assert "server crashed" in result.error


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class TestMCPManager:
    """Test MCPManager discovery (without real MCP servers)."""

    @pytest.mark.asyncio
    async def test_nonexistent_dir_returns_zero(self, tmp_path):
        from app.tools.mcp import MCPManager
        mgr = MCPManager()
        registry = MagicMock()

        with patch("app.tools.mcp.config") as mock_config:
            mock_config.MCP_CONFIG_DIR = str(tmp_path / "does_not_exist")
            count = await mgr.discover_and_register(registry)

        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_dir_returns_zero(self, tmp_path):
        from app.tools.mcp import MCPManager
        mgr = MCPManager()
        registry = MagicMock()

        with patch("app.tools.mcp.config") as mock_config:
            mock_config.MCP_CONFIG_DIR = str(tmp_path)
            count = await mgr.discover_and_register(registry)

        assert count == 0

    @pytest.mark.asyncio
    async def test_missing_mcp_package_returns_zero(self, tmp_path):
        from app.tools.mcp import MCPManager
        mgr = MCPManager()
        registry = MagicMock()

        # Create a config file so we get past the "no configs" check
        config_file = tmp_path / "test_server.json"
        config_file.write_text('{"command": "test-server", "args": []}')

        with patch("app.tools.mcp.config") as mock_config:
            mock_config.MCP_CONFIG_DIR = str(tmp_path)
            # The import of 'mcp' will fail -> returns 0
            with patch.dict("sys.modules", {"mcp": None}):
                count = await mgr.discover_and_register(registry)

        assert count == 0
