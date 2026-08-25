"""Correction intent must not hijack reasoning queries (model program 2026-08-24).

Live root cause of the reasoning-eval 0.714 plateau: the correction pre-filter
pattern `no,?\\s` matched a bare mid-sentence "no " — so a first-turn REASONING
query like "... Answer yes or no and explain why." classified as intent=
"correction", which disables the hard-query auto-thinking gate in
_run_generation_loop (it requires intent == "general"). The model then
pattern-matched the intuitive wrong answer (live: answered "No" to the
A-north-of-B / B-east-of-C spatial question; thinking-enabled answers "Yes").

Two independent guards, both required:
1. learning.py: a bare "no" only signals a correction at the START of the
   message ("No, ..."), not mid-sentence ("yes or no").
2. brain_routing._classify_intent: "correction" requires a prior assistant
   answer in THIS conversation — a first-turn message has nothing to correct.
"""

import pytest

from app.core.brain_routing import _classify_intent
from app.core.learning import is_likely_correction

SPATIAL_EVAL_QUERY = (
    "A is directly north of B. B is directly east of C. "
    "Is A northeast of C? Answer yes or no and explain why."
)


class TestBareNoPattern:
    def test_mid_sentence_no_is_not_a_correction_signal(self):
        assert is_likely_correction(SPATIAL_EVAL_QUERY) is False

    def test_yes_or_no_phrasing_is_not_a_correction_signal(self):
        assert is_likely_correction(
            "Is the statement true? Reply yes or no and justify."
        ) is False

    def test_message_starting_with_no_still_matches(self):
        assert is_likely_correction("No, the capital of Australia is Canberra") is True

    def test_message_starting_with_no_without_comma_still_matches(self):
        assert is_likely_correction("no that answer was wrong") is True

    def test_actually_anchored_to_message_opening(self):
        # 2026-08-25: "actually" anchored like bare-"no" — mid-sentence it is
        # ordinary English and the anywhere-match disabled auto-thinking.
        assert is_likely_correction("Actually, the answer is 42") is True
        assert is_likely_correction("Well, actually it launched in 2019") is True
        assert is_likely_correction("Can you actually run that benchmark?") is False

    def test_thats_wrong_still_matches_anywhere(self):
        assert is_likely_correction("Hmm, that's wrong — it launched in 2019") is True


class TestCorrectionRequiresPriorAnswer:
    @pytest.mark.asyncio
    async def test_first_turn_correction_phrasing_classifies_general(self):
        # Nothing has been said yet — there is nothing to correct.
        assert await _classify_intent("Actually, it's 42", has_prior_answer=False) == "general"

    @pytest.mark.asyncio
    async def test_correction_with_prior_answer_still_classifies_correction(self):
        assert await _classify_intent("Actually, it's 42", has_prior_answer=True) == "correction"

    @pytest.mark.asyncio
    async def test_default_keeps_legacy_behavior(self):
        # Callers that don't pass the flag (existing tests, tools) see no change.
        assert await _classify_intent("Actually, it's 42") == "correction"

    @pytest.mark.asyncio
    async def test_spatial_eval_query_first_turn_is_general(self):
        assert await _classify_intent(SPATIAL_EVAL_QUERY, has_prior_answer=False) == "general"

    @pytest.mark.asyncio
    async def test_greeting_unaffected_by_flag(self):
        assert await _classify_intent("Hello", has_prior_answer=False) == "greeting"
