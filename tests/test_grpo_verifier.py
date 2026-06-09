"""Tests for app.core.grpo_verifier — deterministic GRPO rollout scoring."""

from __future__ import annotations

from app.core import grpo_verifier as v


# ---- is_replayable ------------------------------------------------------

def test_is_replayable_known_types():
    assert v.is_replayable("math_correct") is True
    assert v.is_replayable("json_valid") is True
    assert v.is_replayable("schema_match") is True


def test_is_replayable_unknown_returns_false():
    assert v.is_replayable("tool_correct") is False
    assert v.is_replayable("claim_grounded") is False
    assert v.is_replayable("not_a_signal") is False


def test_is_replayable_code_opt_in():
    assert v.is_replayable("code_passes_tests") is False
    assert v.is_replayable("code_passes_tests", allow_code=True) is True


# ---- verify_math --------------------------------------------------------

def test_verify_math_correct_addition():
    assert v.verify_math("compute 137 + 41", "The answer is 178.") == 1.0


def test_verify_math_correct_multiplication():
    assert v.verify_math("calculate 12 * 5", "60") == 1.0


def test_verify_math_wrong_answer():
    assert v.verify_math("calculate 12 * 5", "55") == 0.0


def test_verify_math_no_expression_in_query():
    # No arithmetic shape in query — unverifiable, return None
    assert v.verify_math("what is python?", "a language") is None


def test_verify_math_within_tolerance():
    assert v.verify_math("calculate 1.0 / 3", "0.3333") == 1.0


def test_verify_math_handles_thousand_separator_in_response():
    assert v.verify_math("compute 1000 * 5", "5,000") == 1.0


def test_verify_math_division_by_zero_safe():
    # The expression "5 / 0" must not crash
    out = v.verify_math("compute 5 / 0", "infinity")
    # eval returns None → unverifiable
    assert out is None


def test_verify_math_pow_overflow_protected():
    # Don't allow giant exponents to blow up
    out = v.verify_math("compute 2 ** 99999", "blah")
    assert out is None


def test_verify_math_word_form_operators():
    assert v.verify_math("compute 12 plus 5", "17") == 1.0
    assert v.verify_math("compute 12 times 5", "60") == 1.0


# ---- verify_json --------------------------------------------------------

def test_verify_json_clean_object():
    assert v.verify_json("q", '{"tool": "calc"}') == 1.0


def test_verify_json_extracts_embedded():
    raw = 'Sure, here it is: {"foo": 1, "bar": [1,2]}'
    assert v.verify_json("q", raw) == 1.0


def test_verify_json_invalid():
    assert v.verify_json("q", "not json at all") == 0.0


def test_verify_json_empty():
    assert v.verify_json("q", "") == 0.0


def test_verify_json_partial():
    # Unbalanced braces — fails
    assert v.verify_json("q", '{"foo": 1') == 0.0


# ---- verify_schema ------------------------------------------------------

def test_verify_schema_valid_tool_call():
    resp = '{"tool": "calculator", "args": {"expression": "1+1"}}'
    assert v.verify_schema("q", resp) == 1.0


def test_verify_schema_missing_args_field():
    resp = '{"tool": "calculator"}'
    assert v.verify_schema("q", resp) == 0.0


def test_verify_schema_evidence_target_match():
    resp = '{"tool": "calculator", "args": {}}'
    assert v.verify_schema("q", resp, evidence="tool=calculator step=1") == 1.0


def test_verify_schema_evidence_target_mismatch():
    resp = '{"tool": "calculator", "args": {}}'
    out = v.verify_schema("q", resp, evidence="tool=web_search step=1")
    assert out == 0.5  # right shape, wrong target


def test_verify_schema_no_tool_key():
    assert v.verify_schema("q", "plain text") == 0.0


# ---- top-level verify dispatch -----------------------------------------

def test_verify_dispatches_to_math():
    assert v.verify("math_correct", "1 + 1", "2") == 1.0


def test_verify_dispatches_to_json():
    assert v.verify("json_valid", "q", '{"a": 1}') == 1.0


def test_verify_unknown_type_returns_none():
    assert v.verify("not_a_real_type", "q", "r") is None


def test_verify_tool_correct_not_replayable():
    # tool_correct is not in the replayable map — falls through to None
    assert v.verify("tool_correct", "q", "r") is None
