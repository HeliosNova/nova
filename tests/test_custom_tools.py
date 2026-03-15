"""Tests for the Dynamic Tool Creation module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.core.custom_tools import CustomToolStore, DynamicTool, CustomToolRecord, TOOL_CREATE_DESCRIPTION


# ===========================================================================
# CustomToolStore: CRUD
# ===========================================================================

class TestCustomToolStore:
    @pytest.fixture
    def store(self, db):
        return CustomToolStore(db)

    def test_create_tool(self, store):
        tid = store.create_tool(
            name="greet",
            description="Generate a greeting",
            parameters='[{"name": "name", "type": "str"}]',
            code='def run(name="World"):\n    return f"Hello, {name}!"',
        )
        assert tid > 0

    def test_create_duplicate_rejected(self, store):
        store.create_tool("greet", "desc", "[]", 'def run(): return "hi"')
        tid2 = store.create_tool("greet", "desc2", "[]", 'def run(): return "hello"')
        assert tid2 == -1

    def test_create_unsafe_code_rejected(self, store):
        tid = store.create_tool(
            "danger", "bad tool", "[]", "import os\ndef run(): return os.getcwd()"
        )
        assert tid == -1

    def test_create_code_too_long(self, store):
        long_code = 'def run(): return "x"\n' + "# padding\n" * 1000
        assert len(long_code) > 5000
        tid = store.create_tool("long_tool", "desc", "[]", long_code)
        assert tid == -1

    def test_get_tool(self, store):
        store.create_tool("greet", "desc", "[]", 'def run(): return "hi"')
        tool = store.get_tool("greet")
        assert tool is not None
        assert tool.name == "greet"
        assert tool.description == "desc"

    def test_get_nonexistent_tool(self, store):
        assert store.get_tool("nope") is None

    def test_get_all_tools(self, store):
        store.create_tool("tool_a", "desc a", "[]", 'def run(): return "a"')
        store.create_tool("tool_b", "desc b", "[]", 'def run(): return "b"')
        tools = store.get_all_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    def test_delete_tool(self, store):
        store.create_tool("temp", "desc", "[]", 'def run(): return "tmp"')
        assert store.delete_tool("temp") is True
        assert store.get_tool("temp") is None

    def test_delete_nonexistent(self, store):
        assert store.delete_tool("nope") is False

    def test_toggle_tool(self, store):
        store.create_tool("toggler", "desc", "[]", 'def run(): return "on"')
        store.toggle_tool("toggler", enabled=False)
        assert store.get_tool("toggler") is None  # disabled tools not returned by get_tool
        store.toggle_tool("toggler", enabled=True)
        assert store.get_tool("toggler") is not None

    def test_name_normalization(self, store):
        tid = store.create_tool("My Cool Tool", "desc", "[]", 'def run(): return "ok"')
        assert tid > 0
        tool = store.get_tool("my_cool_tool")
        assert tool is not None

    def test_max_tools_limit(self, store):
        store.MAX_TOOLS = 3
        for i in range(3):
            store.create_tool(f"tool_{i}", "desc", "[]", f'def run(): return "{i}"')
        tid = store.create_tool("tool_overflow", "desc", "[]", 'def run(): return "x"')
        assert tid == -1

    def test_invalid_parameters_defaults_to_empty(self, store):
        tid = store.create_tool("bad_params", "desc", "not json", 'def run(): return "ok"')
        assert tid > 0
        tool = store.get_tool("bad_params")
        assert tool.parameters == "[]"

    def test_list_parameters_converted(self, store):
        params = [{"name": "x", "type": "str"}]
        tid = store.create_tool("list_params", "desc", params, 'def run(x=""): return x')
        assert tid > 0


# ===========================================================================
# Usage tracking and auto-disable
# ===========================================================================

class TestUsageTracking:
    @pytest.fixture
    def store(self, db):
        s = CustomToolStore(db)
        s.create_tool("counter", "desc", "[]", 'def run(): return "ok"')
        return s

    def test_record_success(self, store):
        store.record_use("counter", success=True)
        tool = store.get_tool("counter")
        assert tool.times_used == 1
        assert tool.success_rate == 1.0

    def test_record_failure(self, store):
        store.record_use("counter", success=False)
        tool = store.get_tool("counter")
        assert tool.times_used == 1
        assert tool.success_rate == 0.0

    def test_auto_disable_on_low_success_rate(self, store):
        # 5 failures in a row → should auto-disable
        for _ in range(5):
            store.record_use("counter", success=False)
        tool = store.get_tool("counter")
        assert tool is None  # disabled

    def test_no_auto_disable_with_good_rate(self, store):
        for _ in range(4):
            store.record_use("counter", success=True)
        store.record_use("counter", success=False)
        tool = store.get_tool("counter")
        assert tool is not None  # still enabled


# ===========================================================================
# DynamicTool execution
# ===========================================================================

class TestDynamicTool:
    @pytest.fixture
    def tool(self, db):
        store = CustomToolStore(db)
        store.create_tool(
            "adder",
            "Add two numbers",
            '[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}]',
            'def run(a=0, b=0):\n    return str(int(a) + int(b))',
        )
        record = store.get_tool("adder")
        return DynamicTool(record, store)

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        result = await tool.execute(a=3, b=4)
        assert result.success
        assert "7" in result.output

    @pytest.mark.asyncio
    async def test_execute_no_args(self, tool):
        result = await tool.execute()
        assert result.success
        assert "0" in result.output

    @pytest.mark.asyncio
    async def test_unsafe_code_blocked(self, db):
        store = CustomToolStore(db)
        # Manually insert unsafe code bypassing safety check
        db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) VALUES (?, ?, ?, ?)",
            ("evil", "bad tool", "[]", "import os\ndef run(): return os.getcwd()"),
        )
        record = store.get_tool("evil")
        tool = DynamicTool(record, store)
        result = await tool.execute()
        assert not result.success
        assert "blocked" in result.error.lower()


# ===========================================================================
# TOOL_CREATE_DESCRIPTION constant
# ===========================================================================

class TestToolCreateDescription:
    def test_description_format(self):
        assert "tool_create" in TOOL_CREATE_DESCRIPTION
        assert "Python function" in TOOL_CREATE_DESCRIPTION
        assert "run" in TOOL_CREATE_DESCRIPTION


# ===========================================================================
# Integration: Brain + tool_create
# ===========================================================================

class TestToolCreateIntegration:
    @pytest.mark.asyncio
    async def test_tool_create_in_brain(self, db):
        """tool_create action in brain.py should create and register a new tool."""
        from app.core.brain import _handle_tool_create, Services, set_services
        from app.core.memory import ConversationStore, UserFactStore
        from app.tools.base import ToolRegistry

        store = CustomToolStore(db)
        registry = ToolRegistry()

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            tool_registry=registry,
            custom_tools=store,
        )
        set_services(svc)

        result = await _handle_tool_create(svc, {
            "name": "doubler",
            "description": "Doubles a number",
            "parameters": '[{"name": "n", "type": "int"}]',
            "code": 'def run(n=0):\n    return str(int(n) * 2)',
        })

        assert "created successfully" in result
        assert "doubler" in registry.tool_names

    @pytest.mark.asyncio
    async def test_tool_create_missing_args(self, db):
        from app.core.brain import _handle_tool_create, Services, set_services
        from app.core.memory import ConversationStore, UserFactStore

        store = CustomToolStore(db)
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            custom_tools=store,
        )
        set_services(svc)

        result = await _handle_tool_create(svc, {"name": "", "code": ""})
        assert "failed" in result.lower()
