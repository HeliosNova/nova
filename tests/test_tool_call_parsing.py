"""Text tool-call extraction — JSON forms plus the Python-style `name(arg=val)`
fallback the model sometimes emits as plain text.

Bug (2026-06-20): the 9B base would answer "What is 17 multiplied by 23?" with the
literal text `calculator(expression="17*23")` — a Python-call syntax the JSON
extractor missed, so no tool ran and the raw call shipped as the answer (no 391).
The fallback now recognizes `name(arg=val)` for registered tools, precision-gated
to content that STARTS with the call so prose isn't misread.
"""
from __future__ import annotations

from app.core.llm import _extract_tool_calls

TOOLS = [{"name": "calculator"}, {"name": "web_search"}, {"name": "code_exec"}]


def _calls(content):
    return [(c.tool, c.args) for c in _extract_tool_calls(content, TOOLS)]


# --- Python-style paren fallback (the bug) ---

def test_paren_calculator():
    assert _calls('calculator(expression="17*23")') == [("calculator", {"expression": "17*23"})]


def test_paren_multi_kwarg_with_int():
    assert _calls("web_search(query='latest news', max_results=3)") == [
        ("web_search", {"query": "latest news", "max_results": 3})
    ]


def test_paren_case_insensitive():
    assert _calls('Calculator(expression="2+2")') == [("calculator", {"expression": "2+2"})]


def test_paren_nested_parens_in_value():
    assert _calls('calculator(expression="(3+4)*2")') == [("calculator", {"expression": "(3+4)*2"})]


def test_code_exec_paren():
    assert _calls('code_exec(code="print(1+1)")') == [("code_exec", {"code": "print(1+1)"})]


# --- precision guards (must NOT misread as a tool call) ---

def test_prose_mentioning_tool_not_parsed():
    assert _calls('I used the calculator(expression="x") earlier to check.') == []


def test_unknown_tool_not_parsed():
    assert _calls("foobar(x=1)") == []


def test_plain_answer_not_parsed():
    assert _calls("The answer is 42 and that's final.") == []


# --- JSON forms still work (regression) ---

def test_json_object_still_parsed():
    assert _calls('{"tool": "calculator", "args": {"expression": "2+2"}}') == [
        ("calculator", {"expression": "2+2"})
    ]


def test_json_array_still_parsed():
    out = _calls('[{"tool":"calculator","args":{"expression":"1+1"}}]')
    assert out == [("calculator", {"expression": "1+1"})]
