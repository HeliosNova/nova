"""Tests for sub-agent isolation — tool whitelist mechanism."""

import pytest

from app.core.access_tiers import (
    CURIOSITY_TOOLS,
    MAINTENANCE_TOOLS,
    get_tool_whitelist,
    is_tool_allowed,
    set_tool_whitelist,
)


# ---------------------------------------------------------------------------
# Tool whitelist
# ---------------------------------------------------------------------------


def test_no_whitelist_allows_all():
    set_tool_whitelist(None)
    assert is_tool_allowed("web_search")
    assert is_tool_allowed("shell_exec")
    assert is_tool_allowed("anything")


def test_whitelist_restricts():
    set_tool_whitelist(MAINTENANCE_TOOLS)
    try:
        assert is_tool_allowed("memory_search")
        assert is_tool_allowed("knowledge_search")
        assert is_tool_allowed("context_detail")
        assert is_tool_allowed("calculator")
        assert not is_tool_allowed("web_search")
        assert not is_tool_allowed("shell_exec")
        assert not is_tool_allowed("file_ops")
        assert not is_tool_allowed("code_exec")
    finally:
        set_tool_whitelist(None)


def test_curiosity_tools_include_web():
    set_tool_whitelist(CURIOSITY_TOOLS)
    try:
        assert is_tool_allowed("web_search")
        assert is_tool_allowed("http_fetch")
        assert is_tool_allowed("memory_search")
        assert not is_tool_allowed("shell_exec")
        assert not is_tool_allowed("file_ops")
    finally:
        set_tool_whitelist(None)


def test_whitelist_clears():
    set_tool_whitelist(MAINTENANCE_TOOLS)
    assert not is_tool_allowed("web_search")
    set_tool_whitelist(None)
    assert is_tool_allowed("web_search")


def test_get_whitelist():
    set_tool_whitelist(None)
    assert get_tool_whitelist() is None

    set_tool_whitelist(MAINTENANCE_TOOLS)
    wl = get_tool_whitelist()
    assert wl == MAINTENANCE_TOOLS
    set_tool_whitelist(None)


def test_whitelist_from_list():
    set_tool_whitelist(["web_search", "calculator"])
    try:
        assert is_tool_allowed("web_search")
        assert is_tool_allowed("calculator")
        assert not is_tool_allowed("shell_exec")
    finally:
        set_tool_whitelist(None)


# ---------------------------------------------------------------------------
# Predefined sets
# ---------------------------------------------------------------------------


def test_maintenance_tools_content():
    assert "memory_search" in MAINTENANCE_TOOLS
    assert "knowledge_search" in MAINTENANCE_TOOLS
    assert "context_detail" in MAINTENANCE_TOOLS
    assert "calculator" in MAINTENANCE_TOOLS
    assert "web_search" not in MAINTENANCE_TOOLS


def test_curiosity_tools_superset():
    assert MAINTENANCE_TOOLS.issubset(CURIOSITY_TOOLS)
    assert "web_search" in CURIOSITY_TOOLS
    assert "http_fetch" in CURIOSITY_TOOLS
