"""Recall/precision guard for is_likely_correction (the cheap pre-filter that
gates correction learning). Widened 2026-06-09 after measuring 38% recall on
natural correction phrasings; these lock in the gain without precision loss.
"""
from __future__ import annotations

import pytest

from app.core.learning import is_likely_correction as f

# Creatively-phrased corrections the OLD filter missed — must now be caught.
CREATIVE_CORRECTIONS = [
    "Hmm, that doesn't sound right.",
    "I do not think so.",
    "Wasn't it Vue, not React?",
    "You forgot about the leap year.",
    "It is the other way around.",
    "That is a myth.",
    "You are mixing it up with something else.",
    "Last I checked it was 256.",
    "I think it's actually 16.",
    "That is only half the story.",
]

# Must still be caught (the explicit ones).
EXPLICIT_CORRECTIONS = [
    "No, that is wrong, it is Canberra.",
    "Actually the correct answer is 42.",
    "You got that wrong.",
]

# Must NOT fire — includes adversarial near-misses of the new patterns.
NON_CORRECTIONS = [
    "Thanks, that is helpful!",
    "What is the capital of France?",
    "I think it is a great idea, thanks!",   # 'I think it is' (not 'it's')
    "Tell me a myth from Greek mythology.",  # 'a myth' but no 'that's a myth'
    "I checked the weather, looks sunny.",   # 'I checked' but no 'last I checked'
    "Summarize the other document too.",
    "Explain quantum computing to a 10-year-old.",
]


@pytest.mark.parametrize("text", CREATIVE_CORRECTIONS + EXPLICIT_CORRECTIONS)
def test_corrections_detected(text):
    assert f(text) is True, f"should detect correction: {text!r}"


@pytest.mark.parametrize("text", NON_CORRECTIONS)
def test_non_corrections_ignored(text):
    assert f(text) is False, f"should NOT flag as correction: {text!r}"


def test_recall_threshold():
    caught = sum(f(t) for t in CREATIVE_CORRECTIONS)
    assert caught >= 9, f"creative-correction recall regressed: {caught}/10"
