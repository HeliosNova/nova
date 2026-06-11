"""Tests for explicit-tool-request detection (app/core/brain._EXPLICIT_TOOL_REQUEST_RE).

Must fire when the user explicitly asks Nova to run code / use the calculator,
and must NOT fire on implicit math (where self-computing the right answer is fine).
"""
from __future__ import annotations

import pytest

from app.core.brain import _EXPLICIT_TOOL_REQUEST_RE as RE


@pytest.mark.parametrize("q", [
    "Write and run Python code to list all prime numbers less than 20",
    "run this python code and tell me the output",
    "Please execute the script and report the result",
    "Use the calculator to find 15% of 840",
    "use a calculator for this",
    "call code_exec to compute the factorial of 10",
])
def test_fires_on_explicit_tool_request(q):
    assert RE.search(q) is not None, f"should detect explicit tool request: {q!r}"


@pytest.mark.parametrize("q", [
    "Calculate 15 percent of 840",
    "What is the square root of 256?",
    "What is 17 multiplied by 23?",
    "How do I write cleaner Python functions?",   # 'write' + 'python' but no 'run'
    "Tell me about prime numbers",
    "Who leads Anthropic?",
])
def test_does_not_fire_on_implicit_or_unrelated(q):
    assert RE.search(q) is None, f"should NOT force a tool for: {q!r}"
