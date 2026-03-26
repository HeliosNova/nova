"""Tests for platform detection and safe environment helpers."""

from __future__ import annotations

import os
import sys

from app.core.platform import get_safe_env, get_safe_cwd, get_shell_command, IS_WINDOWS


def test_is_windows_flag():
    """IS_WINDOWS matches sys.platform."""
    assert IS_WINDOWS == (sys.platform == "win32")


def test_safe_env_has_path():
    """Safe env always includes PATH."""
    env = get_safe_env()
    assert "PATH" in env
    assert len(env["PATH"]) > 0


def test_safe_env_no_secrets():
    """Safe env never includes sensitive variables."""
    os.environ["SECRET_KEY"] = "top-secret"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    try:
        env = get_safe_env()
        assert "SECRET_KEY" not in env
        assert "OPENAI_API_KEY" not in env
    finally:
        del os.environ["SECRET_KEY"]
        del os.environ["OPENAI_API_KEY"]


def test_safe_env_has_lang():
    """Safe env includes LANG for consistent locale handling."""
    env = get_safe_env()
    assert env.get("LANG") == "C.UTF-8"


def test_safe_cwd_returns_string():
    """get_safe_cwd returns a non-empty string."""
    cwd = get_safe_cwd()
    assert isinstance(cwd, str)
    assert len(cwd) > 0


def test_shell_command_returns_list():
    """get_shell_command returns a list suitable for subprocess."""
    cmd = get_shell_command("echo hello")
    assert isinstance(cmd, list)
    assert len(cmd) >= 2
    assert "echo hello" in cmd


def test_shell_command_windows_vs_unix():
    """Verify shell command structure matches platform."""
    cmd = get_shell_command("ls -la")
    if IS_WINDOWS:
        assert cmd[0] == "cmd"
        assert cmd[1] == "/c"
    else:
        assert cmd[0] == "sh"
        assert cmd[1] == "-c"
