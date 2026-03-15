"""Tests for Phase 4: Tools and ToolRegistry."""

from __future__ import annotations

import pytest

from app.tools.base import BaseTool, ToolRegistry, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.code_exec import CodeExecTool, _check_code_safety
from app.tools.file_ops import FileOpsTool, _safe_path
from app.tools.http_fetch import HttpFetchTool, _is_safe_url
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.memory_tool import MemorySearchTool
from app.tools.web_search import WebSearchTool


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
    def test_blocks_localhost(self):
        assert not _is_safe_url("http://localhost:8080")

    def test_blocks_private_ip(self):
        assert not _is_safe_url("http://192.168.1.1")
        assert not _is_safe_url("http://10.0.0.1")
        assert not _is_safe_url("http://172.16.0.1")

    def test_blocks_metadata(self):
        assert not _is_safe_url("http://169.254.169.254/latest/meta-data/")

    def test_allows_public_url(self):
        assert _is_safe_url("https://example.com")

    def test_blocks_non_http(self):
        assert not _is_safe_url("file:///etc/passwd")
        assert not _is_safe_url("ftp://example.com")


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
