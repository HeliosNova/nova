"""Deterministic calculator-transcription guard (model program item 2b).

Live failure 2026-08-24: [MATH TOOL ROUTING] correctly forced the calculator
("847 * 293 - 1200 = 246971") but the 9B then WROTE DIFFERENT DIGITS in its
answer ("= **235,671**"). Tool routing is pointless if the read-back mangles
the number, and digit transcription is exactly what small models fumble.

The guard is deterministic: if a calculator result "E = N" exists for this
turn and the answer claims "E = M" (same normalized expression, different
number), M is a transcription error BY CONSTRUCTION — substitute N. Numbers
not tied to a calculator expression are never touched.
"""

from app.core.brain import _fix_calculator_transcriptions


def _tr(expr: str, value: str):
    # Shape mirrors gen.tool_results entries in brain.py (key is "output").
    return {"tool": "calculator", "output": f"{expr} = {value}"}


class TestTranscriptionGuard:
    def test_wrong_digits_after_same_expression_are_replaced(self):
        answer = "847 * 293 - 1200 = **235,671**"
        fixed = _fix_calculator_transcriptions(answer, [_tr("847 * 293 - 1200", "246971")])
        assert "246971" in fixed or "246,971" in fixed
        assert "235,671" not in fixed and "235671" not in fixed

    def test_correct_answer_untouched(self):
        answer = "847 * 293 - 1200 = 246971"
        fixed = _fix_calculator_transcriptions(answer, [_tr("847 * 293 - 1200", "246971")])
        assert fixed == answer

    def test_comma_formatted_correct_answer_untouched(self):
        answer = "The result is 847 * 293 - 1200 = 246,971."
        fixed = _fix_calculator_transcriptions(answer, [_tr("847 * 293 - 1200", "246971")])
        assert fixed == answer

    def test_whitespace_variations_still_match(self):
        answer = "So 847*293 - 1200 = 999999 apples."
        fixed = _fix_calculator_transcriptions(answer, [_tr("847 * 293 - 1200", "246971")])
        assert "246971" in fixed and "999999" not in fixed

    def test_unrelated_numbers_never_touched(self):
        answer = "In 2026 the answer to 847 * 293 - 1200 = 246971 was computed in 3 seconds."
        fixed = _fix_calculator_transcriptions(answer, [_tr("847 * 293 - 1200", "246971")])
        assert "2026" in fixed and "3 seconds" in fixed

    def test_no_calculator_results_is_noop(self):
        answer = "1 + 1 = 3 (intentionally wrong, no tool ran)"
        assert _fix_calculator_transcriptions(answer, []) == answer

    def test_non_calculator_tools_ignored(self):
        answer = "847 * 293 - 1200 = 999"
        results = [{"tool": "web_search", "output": "847 * 293 - 1200 = 246971"}]
        assert _fix_calculator_transcriptions(answer, results) == answer

    def test_multiple_expressions_each_guarded(self):
        answer = "First 2 + 2 = 5 and then 10 / 4 = 2.5 exactly."
        results = [_tr("2 + 2", "4"), _tr("10 / 4", "2.5")]
        fixed = _fix_calculator_transcriptions(answer, results)
        assert "2 + 2 = 4" in fixed and "10 / 4 = 2.5" in fixed

    def test_decimal_result_wrong_transcription_fixed(self):
        answer = "10 / 4 = 2.4"
        fixed = _fix_calculator_transcriptions(answer, [_tr("10 / 4", "2.5")])
        assert "2.5" in fixed and "2.4" not in fixed

    def test_empty_answer_noop(self):
        assert _fix_calculator_transcriptions("", [_tr("1+1", "2")]) == ""
