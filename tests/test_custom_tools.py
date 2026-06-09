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

    def test_create_with_imports_allowed(self, store):
        # Sovereign owner directive (2026-04): no policy gates on Nova's
        # custom tools. Anything that parses as valid Python is accepted.
        tid = store.create_tool(
            "ok_with_import",
            "tool that uses os",
            "[]",
            "import os\ndef run(): return os.getcwd()",
        )
        assert tid > 0

    def test_create_invalid_syntax_rejected(self, store):
        # The only remaining gate is syntactic correctness — code that
        # doesn't parse can't be stored.
        tid = store.create_tool(
            "broken", "bad tool", "[]", "def run(:\n    return 'broken'"
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
        from unittest.mock import patch
        with patch.object(type(store), "MAX_TOOLS", new=property(lambda self: 3)):
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
        # EMA (alpha=0.15): 0.15*0.0 + 0.85*1.0 = 0.85
        assert tool.success_rate == pytest.approx(0.85, abs=0.01)

    def test_auto_disable_on_low_success_rate(self, store):
        # EMA needs more failures to drop below 0.3 threshold
        for _ in range(8):
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
    @pytest.fixture(autouse=True)
    def _set_tier(self):
        """DynamicTool requires standard/full tier."""
        with patch("app.core.access_tiers.config") as mock_cfg:
            mock_cfg.SYSTEM_ACCESS_LEVEL = "standard"
            yield

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
    async def test_safe_code_with_imports_runs(self, db):
        # Sovereign owner directive — no policy gates. Anything that parses runs.
        store = CustomToolStore(db)
        db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) VALUES (?, ?, ?, ?)",
            ("ctypes_ok", "uses ctypes", "[]", "import ctypes\ndef run(): return str(type(ctypes))"),
        )
        record = store.get_tool("ctypes_ok")
        tool = DynamicTool(record, store)
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"
        assert "module" in result.output


# ===========================================================================
# Sandbox hardening — dynamic tools are ALWAYS sandboxed
# ===========================================================================

class TestDynamicToolSandbox:
    """Sovereign-mode tests: dynamic tools have NO policy gates.

    Owner directive (2026-04): 'access tier should always be unblocked',
    'no policy gates on Nova'. These tests pin the unrestricted behavior;
    they MUST FAIL if anyone re-introduces sandbox restrictions.
    """

    @pytest.fixture(autouse=True)
    def _set_full_tier(self):
        with patch("app.core.access_tiers.config") as mock_cfg:
            mock_cfg.SYSTEM_ACCESS_LEVEL = "full"
            yield

    def _make_tool(self, db, name, code):
        store = CustomToolStore(db)
        db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) VALUES (?, ?, ?, ?)",
            (name, f"test tool {name}", "[]", code),
        )
        record = store.get_tool(name)
        return DynamicTool(record, store)

    @pytest.mark.asyncio
    async def test_socket_import_runs(self, db):
        tool = self._make_tool(
            db, "net_tool",
            "import socket\ndef run(): return socket.gethostname()",
        )
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"

    @pytest.mark.asyncio
    async def test_os_import_runs(self, db):
        tool = self._make_tool(db, "os_tool", "import os\ndef run(): return os.getcwd()")
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"

    @pytest.mark.asyncio
    async def test_subprocess_import_runs(self, db):
        tool = self._make_tool(
            db, "sub_tool",
            "import subprocess\ndef run(): return 'ok'",
        )
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"

    @pytest.mark.asyncio
    async def test_http_libs_import(self, db):
        # Test only modules importable from a `python -I` subprocess (which
        # ignores user site-packages). On the host, httpx may be installed via
        # pip --user but won't load under -I; in the container it's present
        # via system site-packages so it works there. Probe explicitly.
        import subprocess, sys
        candidates = ["http", "urllib"]
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-c", "import httpx"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                candidates.append("httpx")
        except Exception:
            pass
        for mod in candidates:
            tool = self._make_tool(
                db, f"http_{mod}",
                f"import {mod}\ndef run(): return '{mod} ok'",
            )
            result = await tool.execute()
            assert result.success, f"import {mod} unexpectedly failed: {result.error}"

    @pytest.mark.asyncio
    async def test_safe_code_still_works_at_full_tier(self, db):
        tool = self._make_tool(
            db, "math_tool",
            'import math\ndef run(): return str(math.pi)'
        )
        result = await tool.execute()
        assert result.success
        assert "3.14" in result.output

    @pytest.mark.asyncio
    async def test_runtime_network_works(self, db):
        tool = self._make_tool(
            db, "real_net",
            'def run():\n    import socket\n    return "OK: " + socket.gethostname()'
        )
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"
        assert "OK" in result.output

    @pytest.mark.asyncio
    async def test_eval_runs(self, db):
        tool = self._make_tool(
            db, "eval_tool",
            'def run(): return str(eval("1+1"))'
        )
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"
        assert "2" in result.output

    @pytest.mark.asyncio
    async def test_exec_runs(self, db):
        tool = self._make_tool(
            db, "exec_tool",
            'def run():\n    ns = {}\n    exec("x=1+2", ns)\n    return str(ns["x"])'
        )
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"
        assert "3" in result.output

    @pytest.mark.asyncio
    async def test_pickle_import_runs(self, db):
        tool = self._make_tool(
            db, "pickle_tool",
            'import pickle\ndef run(): return "ok"'
        )
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"

    @pytest.mark.asyncio
    async def test_create_tool_accepts_imports(self, db):
        store = CustomToolStore(db)
        tid = store.create_tool(
            "os_reader", "reads os info", "[]",
            "import os\ndef run(): return os.getcwd()"
        )
        assert tid > 0


class TestDynamicToolSandboxNoneTier:
    """At 'none' tier the same unrestricted behavior applies."""

    @pytest.fixture(autouse=True)
    def _set_none_tier(self):
        with patch("app.core.access_tiers.config") as mock_cfg:
            mock_cfg.SYSTEM_ACCESS_LEVEL = "none"
            yield

    def _make_tool(self, db, name, code):
        store = CustomToolStore(db)
        db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) VALUES (?, ?, ?, ?)",
            (name, f"test {name}", "[]", code),
        )
        record = store.get_tool(name)
        return DynamicTool(record, store)

    @pytest.mark.asyncio
    async def test_os_runs_at_none_tier(self, db):
        tool = self._make_tool(db, "os_none", "import os\ndef run(): return os.getcwd()")
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"

    @pytest.mark.asyncio
    async def test_subprocess_runs_at_none_tier(self, db):
        tool = self._make_tool(db, "sub_none", "import subprocess\ndef run(): return 'ok'")
        result = await tool.execute()
        assert result.success, f"unexpected failure: {result.error}"


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


# ===========================================================================
# _validate_proposal AST blocked-import gate (#20)
# ===========================================================================

class TestValidateProposalImportGate:
    """LLM-generated tool proposals must be rejected at validation time if
    they reference an import on the current tier's block-list. Without this
    they get persisted to the custom_tools registry as valid; only the
    runtime sandbox blocks the actual execution, but by then the rotten
    tool has aged/promoted infrastructure built around it."""

    @staticmethod
    def _proposal(code: str) -> dict:
        return {
            "name": "my_tool",
            "description": "a tool that does a thing — over 10 chars",
            "code": code,
            "parameters": [],
        }

    def test_clean_code_accepted(self, monkeypatch):
        # Clean stdlib-math proposal with no blocked imports — must pass.
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "sandboxed")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "import math\n"
            "def run(x: float = 1.0) -> str:\n"
            "    return str(math.sqrt(x))\n"
        ))
        assert ok, f"expected accept, got: {reason}"

    def test_blocks_top_level_os_import(self, monkeypatch):
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "sandboxed")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "import os\n"
            "def run() -> str:\n"
            "    return os.getcwd()\n"
        ))
        assert not ok
        assert "blocked import" in reason and "os" in reason

    def test_blocks_from_import(self, monkeypatch):
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "sandboxed")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "from subprocess import check_output\n"
            "def run() -> str:\n"
            "    return check_output(['ls']).decode()\n"
        ))
        assert not ok
        assert "subprocess" in reason

    def test_blocks_submodule_import(self, monkeypatch):
        # `from urllib.request import urlopen` — top-level package is `urllib`.
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "sandboxed")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "from urllib.request import urlopen\n"
            "def run(u: str = 'x') -> str:\n"
            "    return urlopen(u).read().decode()\n"
        ))
        assert not ok
        assert "urllib" in reason

    def test_blocks_dynamic_import_via_dunder(self, monkeypatch):
        # __import__("os") is a builtin bypass for `import os`. Must still trip.
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "sandboxed")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "def run() -> str:\n"
            "    os = __import__('os')\n"
            "    return os.getcwd()\n"
        ))
        assert not ok
        assert "__import__" in reason and "os" in reason

    def test_none_tier_allows_blocked_imports(self, monkeypatch):
        # When the operator explicitly sets tier="none" the gate must
        # release — that's the documented opt-out.
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "none")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "import os\n"
            "def run() -> str:\n"
            "    return os.getcwd()\n"
        ))
        assert ok, f"tier=none should release: {reason}"

    def test_standard_tier_allows_pathlib(self, monkeypatch):
        # pathlib is blocked at sandboxed but allowed at standard. Verify
        # the gate consults the tier correctly.
        from app.core import auto_tools, access_tiers
        monkeypatch.setattr(access_tiers, "_tier", lambda: "standard")
        ok, reason = auto_tools._validate_proposal(self._proposal(
            "from pathlib import Path\n"
            "def run() -> str:\n"
            "    return str(Path('.').resolve())\n"
        ))
        assert ok, f"standard should allow pathlib: {reason}"
