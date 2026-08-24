"""Auto-generated tools run a forced-sandbox AST screen (audit 2026-08-23).

auto_tools/tool_triggers/agent_loop generate tool code from capability gaps that
can be steered by injected web content, and run it in-process at full tier where
socket is not blocked. create_tool(..., screen_network=True) applies the
comprehensive `_DYNAMIC_TOOL_BLOCKED_*` sets (disabled for owner tools): network/
subprocess/FFI/import-machinery imports, exec/introspection builtins, and the
dunder-attribute escape. Owner-driven creation keeps the syntax-only directive.
"""

import pytest

from app.core.custom_tools import _generated_tool_network_violation


@pytest.mark.parametrize(
    "code,expected",
    [
        ("import socket\ndef run():\n    return 1", "socket"),
        ("from urllib.request import urlopen\ndef run():\n    return urlopen('http://x')", "urllib"),
        ("import httpx\ndef run():\n    return httpx.get('http://x')", "httpx"),
        ("import subprocess\ndef run():\n    return subprocess.run(['ls'])", "subprocess"),
        ("import ctypes\ndef run():\n    return 1", "ctypes"),
        # bypass classes the comprehensive screen must also close:
        ("import os\ndef run():\n    return os.system('curl http://x')", "os"),
        ("import importlib\ndef run():\n    return importlib.import_module('socket')", "importlib"),
        ("def run():\n    m = __import__('socket')\n    return m", "socket"),
    ],
)
def test_flags_blocked_imports(code, expected):
    assert _generated_tool_network_violation(code) == expected


@pytest.mark.parametrize(
    "code",
    [
        "def run():\n    return getattr(x, 'read')",                 # exec/introspection builtin
        "def run():\n    return eval('1+1')",
        "def run():\n    return ().__class__.__mro__[1].__subclasses__()",  # object-graph walk
    ],
)
def test_flags_exec_and_dunder_escape(code):
    # These are the in-process bypasses (no obvious import); the screen must
    # still reject them — the exact token varies, so assert "rejected".
    assert _generated_tool_network_violation(code) is not None


@pytest.mark.parametrize(
    "code",
    [
        "import math, json\ndef run(x):\n    return math.sqrt(x)",
        "import re, statistics\nfrom collections import Counter\ndef run(xs):\n    return Counter(xs)",
        "def run(a, b):\n    return a + b",
    ],
)
def test_allows_pure_computation(code):
    assert _generated_tool_network_violation(code) is None
