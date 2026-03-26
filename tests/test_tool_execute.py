"""Tests for individual tool execute() methods with mocked subprocess/httpx.

Tests:
- ShellExecTool uses cmd /c on Windows, sh -c on Linux
- CodeExecTool uses safe env (no API keys leaked)
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

class TestShellExecPlatform:
    """Verify shell_exec constructs correct command based on platform."""

    @pytest.mark.asyncio
    async def test_shell_exec_windows(self, monkeypatch):
        """Mock sys.platform as 'win32', verify command uses cmd /c."""
        monkeypatch.setenv("ENABLE_SHELL_EXEC", "true")
        from app.config import reset_config
        reset_config()

        captured_args = {}

        def mock_subprocess_run(cmd, **kwargs):
            captured_args["cmd"] = cmd
            captured_args["kwargs"] = kwargs
            result = MagicMock()
            result.stdout = "output"
            result.stderr = ""
            result.returncode = 0
            return result

        with patch("app.core.platform.IS_WINDOWS", True), \
             patch("app.core.platform.sys") as mock_sys, \
             patch("app.tools.shell_exec.get_shell_command") as mock_get_cmd, \
             patch("app.tools.shell_exec.subprocess.run", side_effect=mock_subprocess_run), \
             patch("app.tools.shell_exec._check_command_safety", return_value=None):
            mock_sys.platform = "win32"
            mock_get_cmd.return_value = ["cmd", "/c", "dir"]

            from app.tools.shell_exec import ShellExecTool
            tool = ShellExecTool()
            result = await tool.execute(command="dir")

            assert result.success
            mock_get_cmd.assert_called_once_with("dir")
            assert captured_args["cmd"] == ["cmd", "/c", "dir"]

    @pytest.mark.asyncio
    async def test_shell_exec_linux(self, monkeypatch):
        """Mock sys.platform as 'linux', verify command uses sh -c."""
        monkeypatch.setenv("ENABLE_SHELL_EXEC", "true")
        from app.config import reset_config
        reset_config()

        captured_args = {}

        def mock_subprocess_run(cmd, **kwargs):
            captured_args["cmd"] = cmd
            captured_args["kwargs"] = kwargs
            result = MagicMock()
            result.stdout = "output"
            result.stderr = ""
            result.returncode = 0
            return result

        with patch("app.core.platform.IS_WINDOWS", False), \
             patch("app.core.platform.sys") as mock_sys, \
             patch("app.tools.shell_exec.get_shell_command") as mock_get_cmd, \
             patch("app.tools.shell_exec.subprocess.run", side_effect=mock_subprocess_run), \
             patch("app.tools.shell_exec._check_command_safety", return_value=None):
            mock_sys.platform = "linux"
            mock_get_cmd.return_value = ["sh", "-c", "ls -la"]

            from app.tools.shell_exec import ShellExecTool
            tool = ShellExecTool()
            result = await tool.execute(command="ls -la")

            assert result.success
            mock_get_cmd.assert_called_once_with("ls -la")
            assert captured_args["cmd"] == ["sh", "-c", "ls -la"]

    def test_get_shell_command_windows(self):
        """Directly test get_shell_command with IS_WINDOWS=True."""
        with patch("app.core.platform.IS_WINDOWS", True):
            from app.core.platform import get_shell_command
            # Force re-evaluation by calling with patched constant
            # Since the function reads IS_WINDOWS at call time
            result = get_shell_command("echo hello")
            assert result == ["cmd", "/c", "echo hello"]

    def test_get_shell_command_linux(self):
        """Directly test get_shell_command with IS_WINDOWS=False."""
        with patch("app.core.platform.IS_WINDOWS", False):
            from app.core.platform import get_shell_command
            result = get_shell_command("echo hello")
            assert result == ["sh", "-c", "echo hello"]


# ===========================================================================
# Code Exec — Safe Environment
# ===========================================================================

class TestCodeExecSafeEnv:
    """Verify code execution uses safe env with no API keys."""

    @pytest.mark.asyncio
    async def test_code_exec_safe_env(self, monkeypatch):
        """Verify code execution passes safe env to subprocess.run (no API keys)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")

        captured_env = {}

        original_run = subprocess.run

        def mock_subprocess_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            result = MagicMock()
            result.stdout = "4\n"
            result.stderr = ""
            result.returncode = 0
            return result

        with patch("app.tools.code_exec.subprocess.run", side_effect=mock_subprocess_run):
            from app.tools.code_exec import CodeExecTool
            tool = CodeExecTool()
            result = await tool.execute(code="print(2 + 2)")

        assert result.success
        # Safe env should not contain API keys
        assert "OPENAI_API_KEY" not in captured_env
        assert "ANTHROPIC_API_KEY" not in captured_env
        # But should contain basic PATH
        assert "PATH" in captured_env

    @pytest.mark.asyncio
    async def test_code_exec_safe_env_has_lang(self):
        """Safe env should include LANG for consistent encoding."""
        from app.core.platform import get_safe_env
        env = get_safe_env()
        assert "LANG" in env
        assert env["LANG"] == "C.UTF-8"


# ===========================================================================
# Custom Tool — Safe Environment
# ===========================================================================

class TestCustomToolSafeEnv:
    """Verify custom tool execution uses safe env (no API key leakage)."""

    @pytest.fixture(autouse=True)
    def _set_tier(self):
        """DynamicTool requires standard/full tier."""
        with patch("app.core.access_tiers.config") as mock_cfg:
            mock_cfg.SYSTEM_ACCESS_LEVEL = "standard"
            yield

    @pytest.mark.asyncio
    async def test_custom_tool_safe_env(self, monkeypatch):
        """Verify DynamicTool passes safe env to subprocess.run."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")

        from app.core.custom_tools import DynamicTool, CustomToolRecord

        record = CustomToolRecord(
            id=1, name="test_tool", description="test",
            parameters='[{"name": "x", "type": "str"}]',
            code="def run(x=''):\n    return x",
        )
        store = MagicMock()
        store.record_use = MagicMock(return_value=None)
        tool = DynamicTool(record, store)

        captured_env = {}

        def mock_subprocess_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            result = MagicMock()
            result.stdout = "hello\n"
            result.stderr = ""
            result.returncode = 0
            return result

        with patch("app.core.custom_tools.subprocess.run", side_effect=mock_subprocess_run):
            result = await tool.execute(x="hello")

        assert result.success
        # Safe env should not contain API keys
        assert "OPENAI_API_KEY" not in captured_env
        assert "PATH" in captured_env

    @pytest.mark.asyncio
    async def test_custom_tool_records_usage(self):
        """DynamicTool records usage via store.record_use on success."""
        from app.core.custom_tools import DynamicTool, CustomToolRecord

        record = CustomToolRecord(
            id=1, name="tracked_tool", description="test",
            parameters='[]',
            code="def run():\n    return 'ok'",
        )
        store = MagicMock()
        store.record_use = MagicMock(return_value=None)
        tool = DynamicTool(record, store)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "ok\n"
            result.stderr = ""
            result.returncode = 0
            return result

        with patch("app.core.custom_tools.subprocess.run", side_effect=mock_subprocess_run):
            result = await tool.execute()

        assert result.success
        store.record_use.assert_called_once_with("tracked_tool", success=True)


# ===========================================================================
# Web Search — Timeout Handling
# ===========================================================================

class TestWebSearchTimeout:
    """Verify web search handles httpx timeout gracefully."""

    @pytest.mark.asyncio
    async def test_web_search_timeout(self):
        """Mock httpx to timeout, verify appropriate error returned."""
        from app.tools.web_search import WebSearchTool

        tool = WebSearchTool()

        # Mock httpx.AsyncClient to raise TimeoutException
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.web_search.httpx.AsyncClient", return_value=mock_client_instance):
            result = await tool.execute(query="test query")

        assert not result.success
        assert "Search failed" in result.error
        assert result.retriable is True

    @pytest.mark.asyncio
    async def test_web_search_connect_error(self):
        """Mock httpx to raise ConnectError, verify appropriate error returned."""
        from app.tools.web_search import WebSearchTool

        tool = WebSearchTool()

        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.web_search.httpx.AsyncClient", return_value=mock_client_instance):
            result = await tool.execute(query="test query")

        assert not result.success
        assert "Search failed" in result.error
        assert result.retriable is True

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self):
        """Empty query returns validation error."""
        from app.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        result = await tool.execute(query="")
        assert not result.success
        assert "No query" in result.error
