"""Chat/brain fixes from the 2026-08-24 A-Z audit.

1. `_handle_tool_create` (the chat-path tool_create action) was the ONLY
   create_tool call site not passing screen_network=True — an
   injection-steered chat turn could mint a custom tool with network
   imports that the three autonomous paths all screen out.
2. `build_evidence` never included conversation history, so the claim
   validator refused to repeat a name the user stated one turn earlier
   ("multiturn_recall_name" failed 5/5 consecutive nightly evals with
   "I can't give a verified answer here…").
3. The calculator digit-transcription guard only matched "expr = number"
   restatements; a prose answer stating the bare (wrong) number shipped
   uncorrected — and the 9B mis-copied 2 of 3 calculator results on
   2026-08-24.
4. The auto-thinking gate ran on monitor traffic (~20 background
   thinking generations/day on the enriched '=== System Context ==='
   blob) — pure GPU tax on exactly the model where thinking is slowest.
5. Mid-anywhere correction patterns ("actually", "I think it's",
   "not quite") flipped intent=correction on ordinary prose, disabling
   thinking — the same bug class as the fixed bare-"no".
"""

from __future__ import annotations

import asyncio

import pytest


class _RecordingToolStore:
    def __init__(self):
        self.kwargs: dict | None = None

    def create_tool(self, name, description, parameters, code, **kwargs):
        self.kwargs = kwargs
        return -1  # short-circuit after recording — registry not exercised


class TestToolCreateScreensNetwork:
    def test_chat_path_passes_screen_network(self):
        from app.core.brain import _handle_tool_create

        store = _RecordingToolStore()

        class _Svc:
            custom_tools = store
            tool_registry = None

        asyncio.run(_handle_tool_create(_Svc(), {
            "name": "exfil", "description": "d", "parameters": "[]",
            "code": "def run():\n    return 1",
        }))
        assert store.kwargs is not None, "create_tool never called"
        assert store.kwargs.get("screen_network") is True, (
            "chat-path tool_create must screen network imports like the "
            "three autonomous creation paths do"
        )


class TestHistoryAsEvidence:
    def test_build_evidence_includes_history_text(self):
        from app.core.claim_validator import build_evidence

        ev = build_evidence(history_text="My colleague Dr. Verena Lindqvist leads fusion.")
        assert "Verena Lindqvist" in ev

    def test_history_backed_name_survives_validation(self):
        from app.core.claim_validator import build_evidence, validate_claims

        ev = build_evidence(
            history_text="User said: my colleague Dr. Verena Lindqvist is leading the project.",
        )
        answer = "Your colleague's name is Dr. Verena Lindqvist."
        validated, stripped = validate_claims(answer, ev)
        assert "Verena Lindqvist" in validated
        assert not stripped

    def test_current_query_still_excluded(self):
        """The anti-presupposition rule must survive: the CURRENT query is
        not evidence, only PRIOR turns are."""
        from app.core.claim_validator import build_evidence

        ev = build_evidence(query="Who is Dr. Fabricated, creator of X?")
        assert "Fabricated" not in ev


class TestDigitGuardBareNumber:
    _TOOL = [{"tool": "calculator", "output": "246971 * 37 - 12043 = 9125884"}]

    def test_bare_wrong_number_corrected(self):
        from app.core.brain import _fix_calculator_transcriptions

        fixed = _fix_calculator_transcriptions(
            "The result is **8,595,447**.", self._TOOL)
        assert "9125884" in fixed.replace(",", "")
        assert "8595447" not in fixed.replace(",", "")

    def test_correct_bare_number_untouched(self):
        from app.core.brain import _fix_calculator_transcriptions

        ans = "The result is 9,125,884."
        assert _fix_calculator_transcriptions(ans, self._TOOL) == ans

    def test_operands_do_not_block_correction(self):
        from app.core.brain import _fix_calculator_transcriptions

        fixed = _fix_calculator_transcriptions(
            "Multiplying 246971 by 37 and subtracting 12043 gives 8595447.",
            self._TOOL)
        assert "9125884" in fixed.replace(",", "")

    def test_ambiguous_multiple_stray_numbers_untouched(self):
        from app.core.brain import _fix_calculator_transcriptions

        ans = "Between 8595447 and 7654321, hard to say."
        assert _fix_calculator_transcriptions(ans, self._TOOL) == ans

    def test_expr_form_still_corrected(self):
        from app.core.brain import _fix_calculator_transcriptions

        fixed = _fix_calculator_transcriptions(
            "246971 * 37 - 12043 = **235,671**", self._TOOL)
        assert "9125884" in fixed.replace(",", "")


class TestAutoThinkGate:
    # Must genuinely trip _is_hard_reasoning_query (constraint words + >=2
    # numbers) — the documented bat-and-ball trap from the 2026-06-13 probe.
    _HARD = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
             "than the ball. How much does the ball cost?")

    def test_monitor_channel_never_auto_thinks(self):
        from app.core.brain import _auto_think_eligible

        assert _auto_think_eligible(
            query=self._HARD, intent="general", image=None,
            selected_model="nova-ft", channel="monitor") is False

    def test_api_channel_hard_query_auto_thinks(self):
        from app.core.brain import _auto_think_eligible

        assert _auto_think_eligible(
            query=self._HARD, intent="general", image=None,
            selected_model="nova-ft", channel="api") is True

    def test_non_general_intent_never_auto_thinks(self):
        from app.core.brain import _auto_think_eligible

        assert _auto_think_eligible(
            query=self._HARD, intent="correction", image=None,
            selected_model="nova-ft", channel="api") is False


class TestCorrectionPatternAnchors:
    def test_mid_sentence_actually_is_not_correction(self):
        from app.core.learning import is_likely_correction

        assert not is_likely_correction(
            "Can you actually run that benchmark again with the new flags?")

    def test_opening_actually_is_correction(self):
        from app.core.learning import is_likely_correction

        assert is_likely_correction("Actually, the capital is Ankara.")

    def test_mid_sentence_i_think_its_is_not_correction(self):
        from app.core.learning import is_likely_correction

        assert not is_likely_correction(
            "I was reviewing the plan and I think it's time to compare the "
            "two vendors — walk me through the tradeoffs.")

    def test_opening_i_think_its_is_correction(self):
        from app.core.learning import is_likely_correction

        assert is_likely_correction("I think it's Paris, not Rome.")

    def test_mid_sentence_not_quite_is_not_correction(self):
        from app.core.learning import is_likely_correction

        assert not is_likely_correction(
            "The quarterly results are not quite aligned with the forecast, "
            "can you chart them?")

    def test_opening_not_quite_is_correction(self):
        from app.core.learning import is_likely_correction

        assert is_likely_correction("Not quite — the answer is 12.")
