"""MCP (Model Context Protocol) bridge — discover and register external MCP tools.

Loads MCP server configs from JSON files in MCP_CONFIG_DIR,
connects to each server, and wraps discovered tools as BaseTool instances.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import config
from app.tools.base import BaseTool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class MCPTool(BaseTool):
    """Wraps a single MCP tool as a Nova_ BaseTool."""

    def __init__(self, client, tool_spec: dict):
        self.name = f"mcp_{tool_spec.get('name', 'unknown')}"
        self.description = tool_spec.get("description", "MCP tool")
        # Format parameters from JSON schema for the system prompt
        input_schema = tool_spec.get("inputSchema", {})
        props = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            req = " [required]" if pname in required else ""
            param_parts.append(f"{pname}: {ptype}{req}")
        self.parameters = ", ".join(param_parts) if param_parts else "none"
        self._client = client
        self._raw_name = tool_spec.get("name", "unknown")

    async def execute(self, **kwargs) -> ToolResult:
        from app.core.access_tiers import _tier
        tier = _tier()
        if tier == "sandboxed":
            return ToolResult(
                output="",
                success=False,
                error="MCP tools are blocked at 'sandboxed' access tier. Set SYSTEM_ACCESS_LEVEL to 'standard' or higher.",
            )
        if tier == "standard":
            logger.warning("MCP tool '%s' executing at 'standard' tier — external tool access", self.name)
        try:
            result = await self._client.call_tool(self._raw_name, kwargs)
            # MCP results can be text or structured — normalize to string
            if hasattr(result, "content"):
                # mcp SDK returns CallToolResult with content list
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                output = "\n".join(parts)
            else:
                output = str(result)
            if config.ENABLE_INJECTION_DETECTION:
                from app.core.injection import sanitize_content
                output = sanitize_content(output, context="MCP tool output")
            return ToolResult(output=output, success=True)
        except Exception as e:
            logger.error("MCP tool '%s' failed: %s", self.name, e)
            return ToolResult(output="", success=False, error=str(e))


class MCPManager:
    """Discovers MCP configs, connects to servers, registers tools."""

    def __init__(self):
        self._sessions: list = []  # Track for cleanup

    async def discover_and_register(self, registry: ToolRegistry) -> int:
        """Scan MCP_CONFIG_DIR for *.json configs, connect, register tools.
        Returns count of MCP tools registered.
        """
        config_dir = Path(config.MCP_CONFIG_DIR)
        if not config_dir.exists():
            logger.info("MCP config dir %s does not exist, skipping", config_dir)
            return 0

        config_files = list(config_dir.glob("*.json"))
        if not config_files:
            logger.info("No MCP configs found in %s", config_dir)
            return 0

        total = 0
        for config_file in config_files:
            try:
                total += await self._load_server(config_file, registry)
            except Exception as e:
                logger.warning("Failed to load MCP config %s: %s", config_file.name, e)

        return total

    async def _load_server(self, config_file: Path, registry: ToolRegistry) -> int:
        """Load a single MCP server config and register its tools."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("mcp package not installed — skipping MCP support")
            return 0

        server_config = json.loads(config_file.read_text())
        server_name = config_file.stem

        # Support both flat format and nested "mcpServers" format
        if "mcpServers" in server_config:
            # Handle claude-style config: {"mcpServers": {"name": {"command": ..., "args": [...]}}}
            count = 0
            for name, srv_cfg in server_config["mcpServers"].items():
                count += await self._connect_and_register(name, srv_cfg, registry)
            return count
        else:
            return await self._connect_and_register(server_name, server_config, registry)

    async def _connect_and_register(
        self, name: str, srv_config: dict, registry: ToolRegistry
    ) -> int:
        """Connect to one MCP server and register its tools."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return 0

        command = srv_config.get("command", "")
        args = srv_config.get("args", [])
        env = srv_config.get("env")

        if not command:
            logger.warning("MCP server '%s' has no command, skipping", name)
            return 0

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        stdio_cm = None
        session = None
        try:
            # Open stdio connection to MCP server
            stdio_cm = stdio_client(server_params)
            read_stream, write_stream = await stdio_cm.__aenter__()

            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            self._sessions.append((stdio_cm, session))

            # List and register tools
            tools_result = await session.list_tools()
            tools = tools_result.tools if hasattr(tools_result, "tools") else []

            registered = 0
            for tool_spec in tools:
                spec_dict = {
                    "name": tool_spec.name,
                    "description": getattr(tool_spec, "description", ""),
                    "inputSchema": getattr(tool_spec, "inputSchema", {}),
                }
                mcp_tool = MCPTool(session, spec_dict)
                registry.register(mcp_tool)
                registered += 1

            logger.info("MCP server '%s': registered %d tools", name, registered)
            return registered

        except Exception as e:
            # Clean up partial resources on failure
            if session is not None:
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    pass
            if stdio_cm is not None:
                try:
                    await stdio_cm.__aexit__(None, None, None)
                except Exception:
                    pass
            logger.warning("MCP server '%s' connection failed: %s", name, e)
            return 0

    async def refresh(self, registry: ToolRegistry) -> int:
        """Re-discover MCP tools from existing sessions and register new ones.

        Note: For v1.0, MCP tools are discovered at startup. Restart to pick up
        new MCP server tools. This method re-lists tools from existing sessions only.
        """
        total = 0
        for stdio_cm, session in self._sessions:
            try:
                tools_result = await session.list_tools()
                tools = tools_result.tools if hasattr(tools_result, "tools") else []
                for tool_spec in tools:
                    name = f"mcp_{tool_spec.name}"
                    if registry.get(name) is None:
                        spec_dict = {
                            "name": tool_spec.name,
                            "description": getattr(tool_spec, "description", ""),
                            "inputSchema": getattr(tool_spec, "inputSchema", {}),
                        }
                        registry.register(MCPTool(session, spec_dict))
                        total += 1
            except Exception as e:
                logger.warning("MCP refresh failed for session: %s", e)
        if total:
            logger.info("MCP refresh: registered %d new tools", total)
        return total

    async def close(self) -> None:
        """Clean up all MCP sessions."""
        for stdio_cm, session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()
        logger.info("MCP sessions closed")
