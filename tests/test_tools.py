"""Tests for Phase 4: Tools, ToolRegistry, and Delegation."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.base import BaseTool, ToolRegistry, ToolResult, ErrorCategory
from app.tools.calculator import CalculatorTool
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

# ===========================================================================
# MCP Refresh (from test_audit_consolidated)
# ===========================================================================

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

class TestCalculatorOutputFidelity:
    """Regression tests for the 2026-06-10 audit failure: the tool returned
    4339.00000000000 (dead int-formatting branch - SymPy Float.__eq__ vs int
    is structural, always False) and the 9B mis-copied the digit tail into
    answers (4329/4325). Integer results must render clean; natural-language
    words must be rejected, not parsed into letter-product algebra.
    """

    @pytest.mark.asyncio
    async def test_integer_result_has_no_decimal_tail(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="47*89+156")
        assert result.success
        assert result.output == "47*89+156 = 4339"

    @pytest.mark.asyncio
    async def test_natural_language_rejected(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="Calculate 47 * 89 + 156")
        assert not result.success
        assert "pure math" in (result.error or "").lower() or "words" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_symbolic_math_still_works(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="integrate(x**2, x)")
        assert result.success

    @pytest.mark.asyncio
    async def test_math_function_words_allowed(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="sqrt(144) + floor(2.7)")
        assert result.success
        assert "14" in result.output  # 12 + 2

    @pytest.mark.asyncio
    async def test_huge_int_stays_scientific(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2**1000")
        assert result.success
        # 15-digit Float precision must not masquerade as 300 exact digits
        assert "e+" in result.output.lower()

    @pytest.mark.asyncio
    async def test_non_integer_unchanged(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="100/3")
        assert result.success
        assert "33.3" in result.output
