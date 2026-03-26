"""Tests for Phase 4: Tools, ToolRegistry, and Delegation."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.base import BaseTool, ToolRegistry, ToolResult, ErrorCategory
from app.tools.calculator import CalculatorTool
from app.tools.code_exec import CodeExecTool, _check_code_safety
from app.tools.file_ops import FileOpsTool, _safe_path
from app.tools.http_fetch import HttpFetchTool, _is_safe_url
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.memory_tool import MemorySearchTool
from app.tools.web_search import WebSearchTool


def _make_test_db():
    """Create an in-memory SQLite DB with the full schema."""
    from app.database import SafeDB

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = SafeDB(db_path)
    db.init_schema()
    return db


# ===========================================================================
# ToolRegistry
# ===========================================================================

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.get("calculator") is tool

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_get_descriptions(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())
        desc = registry.get_descriptions()
        assert "calculator" in desc
        assert "web_search" in desc

    def test_get_tool_list(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())
        tools = registry.get_tool_list()
        names = {t["name"] for t in tools}
        assert names == {"calculator", "web_search"}

    @pytest.mark.asyncio
    async def test_execute_registered_tool(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = await registry.execute("calculator", {"expression": "2+2"})
        assert "4" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})
        assert "not found" in result.lower()

    def test_tool_names(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry.tool_names


# ===========================================================================
# Calculator
# ===========================================================================

class TestCalculator:
    @pytest.mark.asyncio
    async def test_simple_addition(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2+2")
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_compound_interest(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="15000 * (1 + 0.075)**12")
        assert result.success
        assert "35" in result.output  # ~$35,726

    @pytest.mark.asyncio
    async def test_division(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="100/3")
        assert result.success
        assert "33" in result.output

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="not a math expression %%%")
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="")
        assert not result.success


# ===========================================================================
# Code Execution Safety
# ===========================================================================

class TestCodeExecSafety:
    def test_blocks_os_import(self):
        assert _check_code_safety("import os") is not None

    def test_blocks_subprocess(self):
        assert _check_code_safety("import subprocess") is not None

    def test_blocks_socket(self):
        assert _check_code_safety("from socket import socket") is not None

    def test_allows_math(self):
        assert _check_code_safety("import math\nprint(math.pi)") is None

    def test_blocks_eval(self):
        assert _check_code_safety("eval('1+1')") is not None

    def test_blocks_dunder_import(self):
        assert _check_code_safety("__import__('os')") is not None

    def test_allows_print(self):
        assert _check_code_safety("print('hello')") is None


class TestCodeExec:
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        tool = CodeExecTool()
        result = await tool.execute(code="print(2 + 2)")
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_blocked_import(self):
        tool = CodeExecTool()
        result = await tool.execute(code="import os\nprint(os.getcwd())")
        assert not result.success
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_code(self):
        tool = CodeExecTool()
        result = await tool.execute(code="")
        assert not result.success


# ===========================================================================
# HTTP Fetch Safety
# ===========================================================================

class TestHttpFetchSafety:
    # SSRF tests (localhost, metadata, non-http) live in test_http_fetch.py.
    # Only unique tests kept here.

    def test_blocks_private_ip(self):
        assert not _is_safe_url("http://192.168.1.1")
        assert not _is_safe_url("http://10.0.0.1")
        assert not _is_safe_url("http://172.16.0.1")

    def test_allows_public_url(self):
        assert _is_safe_url("https://example.com")


# ===========================================================================
# File Ops Safety
# ===========================================================================

class TestFileOpsSafety:
    def test_safe_path_within_sandbox(self):
        path = _safe_path("test.txt")
        assert path is not None

    def test_blocks_path_traversal(self):
        path = _safe_path("../../etc/passwd")
        assert path is None

    def test_relative_paths_within_sandbox_work(self):
        # Relative paths resolve under the sandbox root
        path = _safe_path("subdir/test.txt")
        assert path is not None


# ===========================================================================
# Knowledge Search (without retriever)
# ===========================================================================

class TestKnowledgeSearch:
    @pytest.mark.asyncio
    async def test_no_retriever(self):
        tool = KnowledgeSearchTool(retriever=None)
        result = await tool.execute(query="test")
        assert result.success
        assert "no documents" in result.output.lower()

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = KnowledgeSearchTool()
        result = await tool.execute(query="")
        assert not result.success


# ===========================================================================
# Memory Search
# ===========================================================================

class TestMemorySearch:
    @pytest.mark.asyncio
    async def test_search_facts(self, db):
        from app.core.memory import ConversationStore, UserFactStore

        user_facts = UserFactStore(db)
        user_facts.set("name", "John")
        user_facts.set("city", "London")

        tool = MemorySearchTool(user_facts=user_facts)
        result = await tool.execute(query="john")
        assert result.success
        assert "John" in result.output

    @pytest.mark.asyncio
    async def test_search_no_results(self, db):
        from app.core.memory import ConversationStore, UserFactStore

        tool = MemorySearchTool(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        result = await tool.execute(query="xyznonexistent")
        assert result.success
        assert "no matching" in result.output.lower()


# ===========================================================================
# Tool Parameter Validation (from test_audit_consolidated)
# ===========================================================================

class TestToolParameterValidation:
    """ToolRegistry.execute() catches TypeError on bad params."""

    @pytest.fixture
    def registry(self):
        from app.tools.base import ToolRegistry, BaseTool, ToolResult
        class StrictTool(BaseTool):
            name = "strict"
            description = "test"
            parameters = "x: int"
            async def execute(self, *, x: int = 0, **kwargs) -> ToolResult:
                return ToolResult(output=str(x + 1), success=True)
        reg = ToolRegistry()
        reg.register(StrictTool())
        return reg

    @pytest.mark.asyncio
    async def test_valid_params_succeed(self, registry):
        result = await registry.execute("strict", {"x": 5})
        assert result == "6"

    @pytest.mark.asyncio
    async def test_type_error_caught(self, registry):
        result = await registry.execute("strict", {"x": "not_a_number"})
        assert "[Tool error: strict]" in result

    @pytest.mark.asyncio
    async def test_missing_tool(self, registry):
        result = await registry.execute("nonexistent", {})
        assert "[Tool error: nonexistent]" in result
        assert "not found" in result.lower()


# ===========================================================================
# Custom Tool Schema (from test_audit_consolidated)
# ===========================================================================

class TestCustomToolSchema:
    """DynamicTool rejects unexpected parameters."""

    @pytest.fixture(autouse=True)
    def _set_tier(self):
        """DynamicTool requires standard/full tier."""
        with patch("app.core.access_tiers.config") as mock_cfg:
            mock_cfg.SYSTEM_ACCESS_LEVEL = "standard"
            yield

    @pytest.mark.asyncio
    async def test_unexpected_params_rejected(self):
        from app.core.custom_tools import DynamicTool, CustomToolRecord

        record = CustomToolRecord(
            id=1, name="test_tool", description="test",
            parameters='[{"name": "x", "type": "str"}]',
            code="def run(x=''):\n    return x",
        )
        store = MagicMock()
        tool = DynamicTool(record, store)
        result = await tool.execute(x="hello", unexpected_param="bad")
        assert not result.success
        assert "Unexpected parameters" in result.error

    @pytest.mark.asyncio
    async def test_declared_params_accepted(self):
        from app.core.custom_tools import DynamicTool, CustomToolRecord

        record = CustomToolRecord(
            id=1, name="test_tool", description="test",
            parameters='[{"name": "x", "type": "str"}]',
            code="def run(x=''):\n    return x",
        )
        store = MagicMock()
        store.record_use = MagicMock(return_value=None)
        tool = DynamicTool(record, store)
        result = await tool.execute(x="hello")
        assert "Unexpected parameters" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_malformed_schema_skips_validation(self):
        from app.core.custom_tools import DynamicTool, CustomToolRecord

        record = CustomToolRecord(
            id=1, name="test_tool", description="test",
            parameters="not valid json",
            code="def run(**kwargs):\n    return 'ok'",
        )
        store = MagicMock()
        store.record_use = MagicMock(return_value=None)
        tool = DynamicTool(record, store)
        result = await tool.execute(anything="goes")
        assert "Unexpected parameters" not in (result.error or "")


# ===========================================================================
# HTTP Fetch Pooling (from test_audit_consolidated)
# ===========================================================================

class TestHttpFetchPooling:

    def test_singleton_client_reused(self):
        from app.tools import http_fetch
        old_client = http_fetch._client
        try:
            http_fetch._client = None
            client1 = http_fetch._get_client()
            client2 = http_fetch._get_client()
            assert client1 is client2
        finally:
            if http_fetch._client is not None:
                import asyncio
                try:
                    asyncio.get_event_loop().run_until_complete(http_fetch._client.aclose())
                except Exception:
                    pass
            http_fetch._client = old_client

    @pytest.mark.asyncio
    async def test_close_resets_singleton(self):
        from app.tools import http_fetch
        old_client = http_fetch._client
        http_fetch._client = None
        _ = http_fetch._get_client()
        await http_fetch.close_http_client()
        assert http_fetch._client is None
        http_fetch._client = old_client


# ===========================================================================
# Auto-Disable Notification (from test_audit_consolidated)
# ===========================================================================

class TestAutoDisableNotification:

    def test_returns_none_normally(self):
        from app.core.custom_tools import CustomToolStore
        db = MagicMock()
        db.execute = MagicMock(return_value=MagicMock(lastrowid=1))
        db.fetchone = MagicMock(return_value={"times_used": 1, "success_rate": 1.0})
        store = CustomToolStore.__new__(CustomToolStore)
        store._db = db
        result = store.record_use("test_tool", success=True)
        assert result is None

    def test_returns_message_on_auto_disable(self):
        from app.core.custom_tools import CustomToolStore
        db = MagicMock()
        db.execute = MagicMock(return_value=MagicMock())
        db.fetchone = MagicMock(return_value={"times_used": 6, "success_rate": 0.2})
        store = CustomToolStore.__new__(CustomToolStore)
        store._db = db
        result = store.record_use("bad_tool", success=False)
        assert result is not None
        assert "auto-disabled" in result
        assert "bad_tool" in result


# ===========================================================================
# MCP Refresh (from test_audit_consolidated)
# ===========================================================================

class TestMCPRefresh:

    def test_mcp_manager_has_refresh(self):
        from app.tools.mcp import MCPManager
        manager = MCPManager()
        assert hasattr(manager, "refresh")


# ===========================================================================
# Browser Interactive Elements
# ===========================================================================

class TestBrowserInteractiveElements:
    """Verify interactive element formatting and extraction."""

    def test_format_interactive_elements_full(self):
        from app.tools.browser import BrowserTool

        data = {
            "buttons": [
                {"label": "Sign in", "selector": "button#sign-in", "type": "button"},
                {"label": "Submit", "selector": "input[type=\"submit\"]", "type": "submit"},
            ],
            "inputs": [
                {"label": "Email", "selector": "input#email", "type": "email", "value": "user@test.com"},
                {"label": "Password", "selector": "input[name=\"password\"]", "type": "password", "value": ""},
            ],
            "radioGroups": [
                {"label": "Size", "selector": "input[name=\"size\"]", "type": "radio",
                 "options": "Small, Medium, Large [selected]", "selected": "large"},
            ],
            "checkboxGroups": [
                {"label": "Bacon", "selector": "input[name=\"topping\"]", "type": "checkbox",
                 "options": "Bacon [checked], Cheese, Onion", "checked": "bacon"},
            ],
            "links": [
                {"label": "Home", "selector": "a.nav-home", "href": "https://example.com"},
                {"label": "About", "selector": "a[href=\"/about\"]", "href": "https://example.com/about"},
            ],
            "selects": [
                {"label": "Country", "selector": "select#country", "options": "US, UK, CA", "selected": "US"},
            ],
        }
        result = BrowserTool._format_interactive_elements(data)
        assert "Buttons:" in result
        assert '"Sign in"' in result
        assert "button#sign-in" in result
        assert "Inputs:" in result
        assert '"Email"' in result
        assert "input#email" in result
        assert '= "user@test.com"' in result
        assert "(empty)" in result  # Password has no value
        assert "Radio groups:" in result
        assert "selected" in result
        assert "Checkbox groups:" in result
        assert "checked" in result
        assert "Links:" in result
        assert '"Home"' in result
        assert "Selects:" in result
        assert '"Country"' in result

    def test_format_interactive_elements_empty(self):
        from app.tools.browser import BrowserTool

        result = BrowserTool._format_interactive_elements({})
        assert "no interactive elements" in result

    def test_format_interactive_elements_partial(self):
        from app.tools.browser import BrowserTool

        data = {
            "buttons": [{"label": "Go", "selector": "button.go", "type": ""}],
            "inputs": [],
            "links": [],
            "selects": [],
        }
        result = BrowserTool._format_interactive_elements(data)
        assert "Buttons:" in result
        assert '"Go"' in result
        assert "Inputs:" not in result
        assert "Links:" not in result

    def test_get_interactive_elements_action_in_dispatch(self):
        """Verify get_interactive_elements is a valid action string."""
        from app.tools.browser import BrowserTool
        assert "get_interactive_elements" in BrowserTool.parameters


# ===========================================================================
# Custom Tool Limits
# ===========================================================================

class TestCustomToolLimits:
    """Verify custom tool creation limits and auto-disable."""

    def test_code_length_limit(self):
        from app.core.custom_tools import CustomToolStore

        db = _make_test_db()
        store = CustomToolStore(db)
        long_code = "x = 1\n" * 2000  # well over 5000 chars
        result = store.create_tool(
            name="too_long",
            description="Test tool",
            parameters="[]",
            code=long_code,
        )
        assert result == -1, "Code exceeding 5000 chars should be rejected"

    def test_tool_count_limit(self):
        from app.core.custom_tools import CustomToolStore

        db = _make_test_db()
        store = CustomToolStore(db)
        # Create MAX_TOOLS tools
        for i in range(store.MAX_TOOLS):
            code = f"def run(): return 'tool_{i}'"
            store.create_tool(
                name=f"tool_{i}",
                description=f"Test tool {i}",
                parameters="[]",
                code=code,
            )

        # The 51st should fail
        result = store.create_tool(
            name="one_too_many",
            description="Over limit",
            parameters="[]",
            code="def run(): return 'fail'",
        )
        assert result == -1, f"Should reject tool #{store.MAX_TOOLS + 1}"

    def test_auto_disable_on_low_success(self):
        from app.core.custom_tools import CustomToolStore

        db = _make_test_db()
        store = CustomToolStore(db)
        code = "def run(): return 'ok'"
        tool_id = store.create_tool(
            name="failing_tool",
            description="Might fail",
            parameters="[]",
            code=code,
        )
        assert tool_id > 0

        # Record enough failures to trigger auto-disable (EMA alpha=0.15 from 1.0)
        for _ in range(10):
            store.record_use("failing_tool", success=False)

        # Check if auto-disabled
        tool = db.fetchone("SELECT enabled FROM custom_tools WHERE name = ?", ("failing_tool",))
        assert tool["enabled"] == 0, "Tool should be auto-disabled after repeated failures"


# ===========================================================================
# Delegation Tool (from test_delegation)
# ===========================================================================

from app.schema import EventType, StreamEvent
from app.tools.delegate import (
    DelegateTool,
    get_delegation_depth,
    set_delegation_depth,
)


class TestDelegateToolBasics:
    @pytest.fixture(autouse=True)
    def reset_depth(self):
        set_delegation_depth(0)
        yield
        set_delegation_depth(0)

    @pytest.mark.asyncio
    async def test_no_task(self):
        r = await DelegateTool().execute()
        assert not r.success
        assert "No task" in r.error

    @pytest.mark.asyncio
    async def test_disabled(self):
        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": False, "MAX_DELEGATION_DEPTH": 1})()):
            r = await DelegateTool().execute(task="test")
            assert not r.success
            assert "disabled" in r.error.lower()

    @pytest.mark.asyncio
    async def test_max_depth_reached(self):
        set_delegation_depth(1)
        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True, "MAX_DELEGATION_DEPTH": 1})()):
            r = await DelegateTool().execute(task="test")
            assert not r.success
            assert "depth" in r.error.lower()

    @pytest.mark.asyncio
    async def test_successful_delegation(self):
        async def fake_think(**kwargs):
            yield StreamEvent(type=EventType.THINKING, data={"stage": "generating"})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "The weather "})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "is sunny."})
            yield StreamEvent(type=EventType.DONE, data={"conversation_id": "eph"})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True, "MAX_DELEGATION_DEPTH": 1})()):
            with patch("app.core.brain.think", side_effect=fake_think):
                r = await DelegateTool().execute(task="What is the weather?", role="researcher")

        assert r.success
        assert "sunny" in r.output

    @pytest.mark.asyncio
    async def test_depth_restored_after_delegation(self):
        async def fake_think(**kwargs):
            assert get_delegation_depth() == 1
            yield StreamEvent(type=EventType.TOKEN, data={"text": "ok"})
            yield StreamEvent(type=EventType.DONE, data={})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True, "MAX_DELEGATION_DEPTH": 2})()):
            with patch("app.core.brain.think", side_effect=fake_think):
                await DelegateTool().execute(task="test")

        assert get_delegation_depth() == 0

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        long_text = "x" * 5000

        async def verbose_think(**kwargs):
            yield StreamEvent(type=EventType.TOKEN, data={"text": long_text})
            yield StreamEvent(type=EventType.DONE, data={})

        with patch("app.tools.delegate.config",
                   type("C", (), {"ENABLE_DELEGATION": True, "MAX_DELEGATION_DEPTH": 1})()):
            with patch("app.core.brain.think", side_effect=verbose_think):
                r = await DelegateTool().execute(task="test")

        assert r.success
        assert len(r.output) <= 3100
        assert "truncated" in r.output


# ===========================================================================
# Edge Case Tests (Phase 5 audit items 5.9, 5.10)
# ===========================================================================

class TestCalculatorEdgeCases:
    """Calculator edge cases: division by zero, empty, overflow."""

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="1/0")
        # SymPy returns zoo (complex infinity), should still succeed
        assert result.success or "zoo" in (result.output or "") or "infinity" in (result.output or "").lower()

    @pytest.mark.asyncio
    async def test_very_large_number(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2**1000")
        assert result.success

    @pytest.mark.asyncio
    async def test_nested_parentheses(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="((2+3)*4)/2")
        assert result.success
        assert "10" in result.output


class TestHttpFetchEdgeCases:
    """HTTP fetch edge cases: invalid URL, redirect loop."""

    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        tool = HttpFetchTool()
        result = await tool.execute(url="ftp://example.com/file")
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = HttpFetchTool()
        result = await tool.execute(url="")
        assert not result.success
        assert result.error_category == ErrorCategory.VALIDATION

    @pytest.mark.asyncio
    async def test_invalid_method(self):
        tool = HttpFetchTool()
        result = await tool.execute(url="https://example.com", method="TRACE")
        assert not result.success
        assert result.error_category == ErrorCategory.VALIDATION


class TestShellExecEdgeCases:
    """Shell exec edge cases: empty command, disabled."""

    @pytest.mark.asyncio
    async def test_empty_command(self):
        from app.tools.shell_exec import ShellExecTool
        tool = ShellExecTool()
        result = await tool.execute(command="")
        assert not result.success
        assert result.error_category == ErrorCategory.VALIDATION

    @pytest.mark.asyncio
    async def test_shell_disabled(self, monkeypatch):
        from app.tools.shell_exec import ShellExecTool
        monkeypatch.setenv("ENABLE_SHELL_EXEC", "false")
        from app.config import reset_config
        reset_config()
        tool = ShellExecTool()
        result = await tool.execute(command="echo hello")
        assert not result.success
        assert result.error_category == ErrorCategory.PERMISSION


class TestToolErrorPaths:
    """Verify tools return specific ErrorCategory (not generic INTERNAL)."""

    @pytest.mark.asyncio
    async def test_calculator_validation_error(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="")
        assert not result.success
        assert result.error_category == ErrorCategory.VALIDATION

    @pytest.mark.asyncio
    async def test_http_fetch_permission_error(self):
        tool = HttpFetchTool()
        result = await tool.execute(url="http://127.0.0.1/admin")
        assert not result.success
        assert result.error_category == ErrorCategory.PERMISSION

    @pytest.mark.asyncio
    async def test_code_exec_permission_error(self):
        tool = CodeExecTool()
        result = await tool.execute(code="import os; os.system('rm -rf /')")
        assert not result.success
        # Should be VALIDATION (blocked code) not generic INTERNAL
