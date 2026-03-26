"""Tests that validate exam guide alignment structural requirements.

Acts as a regression guard: if someone adds a new tool with a bare
description or missing schema, these tests catch it.
"""

from __future__ import annotations

import json

import pytest

from app.tools.base import BaseTool, ErrorCategory, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Helpers — instantiate all registered tools
# ---------------------------------------------------------------------------

def _get_all_tools() -> list[BaseTool]:
    """Import and instantiate all tool classes."""
    from app.tools.web_search import WebSearchTool
    from app.tools.calculator import CalculatorTool
    from app.tools.http_fetch import HttpFetchTool
    from app.tools.file_ops import FileOpsTool
    from app.tools.code_exec import CodeExecTool
    from app.tools.shell_exec import ShellExecTool
    from app.tools.knowledge import KnowledgeSearchTool
    from app.tools.memory_tool import MemorySearchTool
    from app.tools.browser import BrowserTool
    from app.tools.delegate import DelegateTool
    from app.tools.action_email import EmailSendTool
    from app.tools.action_calendar import CalendarTool
    from app.tools.action_reminder import ReminderTool
    from app.tools.action_webhook import WebhookTool
    from app.tools.background_task import BackgroundTaskTool
    from app.tools.monitor_tool import MonitorTool
    from app.tools.desktop import DesktopTool
    from app.tools.integration import IntegrationTool
    from app.tools.screenshot import ScreenshotTool

    return [
        WebSearchTool(),
        CalculatorTool(),
        HttpFetchTool(),
        FileOpsTool(),
        CodeExecTool(),
        ShellExecTool(),
        KnowledgeSearchTool(),
        MemorySearchTool(),
        BrowserTool(),
        DelegateTool(),
        EmailSendTool(),
        CalendarTool(),
        ReminderTool(),
        WebhookTool(),
        BackgroundTaskTool(),
        MonitorTool(),
        DesktopTool(),
        IntegrationTool(),
        ScreenshotTool(),
    ]


# ---------------------------------------------------------------------------
# Phase 1+2: Every tool has input_schema and description
# ---------------------------------------------------------------------------

class TestToolSchemas:
    """Verify every registered tool has a proper JSON Schema."""

    @pytest.fixture
    def tools(self):
        return _get_all_tools()

    def test_all_tools_have_input_schema(self, tools):
        """Every tool must have input_schema defined (not None)."""
        missing = [t.name for t in tools if t.input_schema is None]
        assert not missing, f"Tools missing input_schema: {missing}"

    def test_all_schemas_are_valid_json_schema(self, tools):
        """Every input_schema must have 'type: object' and 'properties'."""
        invalid = []
        for t in tools:
            schema = t.input_schema
            if schema is None:
                continue
            if schema.get("type") != "object":
                invalid.append(f"{t.name}: missing type=object")
            if "properties" not in schema:
                invalid.append(f"{t.name}: missing properties")
        assert not invalid, f"Invalid schemas: {invalid}"

    def test_all_schemas_have_required_array(self, tools):
        """Every input_schema should declare required params."""
        missing = []
        for t in tools:
            schema = t.input_schema
            if schema is None:
                continue
            if "required" not in schema:
                missing.append(t.name)
        assert not missing, f"Tools missing 'required' in schema: {missing}"

    def test_all_descriptions_are_substantial(self, tools):
        """Every tool description must be > 80 chars (multi-sentence)."""
        short = [(t.name, len(t.description)) for t in tools if len(t.description) < 80]
        assert not short, f"Tools with short descriptions: {short}"


# ---------------------------------------------------------------------------
# Phase 1: ToolResult + ErrorCategory
# ---------------------------------------------------------------------------

class TestErrorCategory:
    """Verify ErrorCategory enum and ToolResult integration."""

    def test_error_category_values(self):
        """All expected error categories exist."""
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.PERMISSION.value == "permission"
        assert ErrorCategory.NOT_FOUND.value == "not_found"
        assert ErrorCategory.INTERNAL.value == "internal"

    def test_tool_result_accepts_category(self):
        """ToolResult can be created with error_category."""
        r = ToolResult(
            output="", success=False, error="test",
            error_category=ErrorCategory.TRANSIENT,
        )
        assert r.error_category == ErrorCategory.TRANSIENT
        assert r.retriable is False  # Default

    def test_tool_result_category_defaults_none(self):
        """ToolResult.error_category defaults to None for backward compat."""
        r = ToolResult(output="ok", success=True)
        assert r.error_category is None


# ---------------------------------------------------------------------------
# Phase 1D: get_tool_list includes schemas for cloud providers
# ---------------------------------------------------------------------------

class TestToolListFormat:
    """Verify ToolRegistry.get_tool_list() produces cloud-ready format."""

    def test_tool_list_includes_description(self):
        """get_tool_list() entries include description."""
        from app.tools.web_search import WebSearchTool

        reg = ToolRegistry()
        reg.register(WebSearchTool())
        tools = reg.get_tool_list()
        assert len(tools) == 1
        assert "description" in tools[0]
        assert len(tools[0]["description"]) > 80

    def test_tool_list_includes_parameters_schema(self):
        """get_tool_list() entries include JSON Schema parameters."""
        from app.tools.calculator import CalculatorTool

        reg = ToolRegistry()
        reg.register(CalculatorTool())
        tools = reg.get_tool_list()
        assert "parameters" in tools[0]
        params = tools[0]["parameters"]
        assert params["type"] == "object"
        assert "expression" in params["properties"]

    def test_cloud_provider_gets_real_schema(self):
        """Anthropic/OpenAI/Google _convert_tools() will get real schemas."""
        from app.tools.http_fetch import HttpFetchTool

        reg = ToolRegistry()
        reg.register(HttpFetchTool())
        tools = reg.get_tool_list()

        # Simulate Anthropic format
        t = tools[0]
        anthropic_tool = {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        }
        assert anthropic_tool["input_schema"]["type"] == "object"
        assert "url" in anthropic_tool["input_schema"]["properties"]


# ---------------------------------------------------------------------------
# Phase 1E: execute_full returns ToolResult
# ---------------------------------------------------------------------------

class TestExecuteFull:
    """Verify ToolRegistry.execute_full() returns structured results."""

    @pytest.mark.asyncio
    async def test_execute_full_success(self):
        from app.tools.calculator import CalculatorTool

        reg = ToolRegistry()
        reg.register(CalculatorTool())
        output, result = await reg.execute_full("calculator", {"expression": "2+2"})
        assert result is not None
        assert result.success is True
        assert "4" in output

    @pytest.mark.asyncio
    async def test_execute_full_failure(self):
        from app.tools.calculator import CalculatorTool

        reg = ToolRegistry()
        reg.register(CalculatorTool())
        output, result = await reg.execute_full("calculator", {"expression": ""})
        assert result is not None
        assert result.success is False
        assert result.error_category == ErrorCategory.VALIDATION

    @pytest.mark.asyncio
    async def test_execute_full_not_found(self):
        reg = ToolRegistry()
        output, result = await reg.execute_full("nonexistent", {})
        assert result is not None
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND


# ---------------------------------------------------------------------------
# Phase 1C: trim_output
# ---------------------------------------------------------------------------

class TestTrimOutput:
    """Verify per-tool trim_output() methods."""

    def test_default_trim_short_passthrough(self):
        """Short output passes through unchanged."""
        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        short = "Hello world"
        assert tool.trim_output(short) == short

    def test_default_trim_truncates_long(self):
        """Long output is truncated with marker."""
        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        long_text = "x" * 5000
        trimmed = tool.trim_output(long_text)
        assert len(trimmed) < 5000
        assert "truncated" in trimmed.lower()

    def test_shell_tail_strategy(self):
        """Shell tool keeps tail of output."""
        from app.tools.shell_exec import ShellExecTool
        tool = ShellExecTool()
        output = "line_start\n" + ("x" * 5000) + "\nimportant_tail"
        trimmed = tool.trim_output(output)
        assert "important_tail" in trimmed
        assert "truncated" in trimmed.lower()

    def test_browser_preserves_interactive(self):
        """Browser tool preserves interactive elements section."""
        from app.tools.browser import BrowserTool
        tool = BrowserTool()
        body = "x" * 5000
        ie = "--- Interactive Elements ---\nButtons: [1] Submit"
        output = f"Page: Test\n{body}\n\n{ie}"
        trimmed = tool.trim_output(output)
        assert "Interactive Elements" in trimmed

    def test_knowledge_keeps_top_chunks(self):
        """Knowledge tool keeps top 3 chunks."""
        from app.tools.knowledge import KnowledgeSearchTool
        tool = KnowledgeSearchTool()
        chunks = "\n\n".join(
            f"[{i}] Source: doc{i}\n{'content ' * 100}" for i in range(1, 8)
        )
        trimmed = tool.trim_output(chunks)
        assert "[1]" in trimmed
        assert "[2]" in trimmed
        assert "[3]" in trimmed
        assert "truncated" in trimmed.lower() or "more results" in trimmed.lower()


# ---------------------------------------------------------------------------
# Phase 4: MCP Server
# ---------------------------------------------------------------------------

class TestMCPServer:
    """Verify MCP server structural requirements."""

    def test_mcp_tools_have_long_descriptions(self):
        """All 5 MCP tools have descriptions > 200 chars."""
        from app.mcp_server import _TOOLS
        short = [(t.name, len(t.description)) for t in _TOOLS if len(t.description) < 200]
        assert not short, f"MCP tools with short descriptions: {short}"

    def test_mcp_error_helper_format(self):
        """_mcp_error returns CallToolResult with isError=True and structured JSON."""
        from app.mcp_server import _mcp_error
        result = _mcp_error("test error", "validation", False)
        assert result.isError is True
        assert len(result.content) == 1
        data = json.loads(result.content[0].text)
        assert data["error"] == "test error"
        assert data["error_category"] == "validation"
        assert data["is_retryable"] is False

    def test_mcp_error_retryable(self):
        from app.mcp_server import _mcp_error
        result = _mcp_error("timeout", "transient", True)
        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert data["is_retryable"] is True
        assert data["error_category"] == "transient"
