"""Tests for the grounding-honesty caveat (app/core/brain._maybe_unverified_caveat).

The caveat must fire ONLY when the exact critique-flagged answer is still what's
shipping (under-fire, never double-signal), and be idempotent.
"""
from __future__ import annotations

from app.core.brain import _maybe_unverified_caveat, _UNVERIFIED_CAVEAT


FLAGGED = "The summit happened on March 3, 2026 with 12 attendees."


def test_caveat_appended_when_flagged_text_survives():
    out = _maybe_unverified_caveat(FLAGGED, FLAGGED)
    assert out != FLAGGED
    assert "couldn't confirm some specifics" in out
    assert out.startswith(FLAGGED)


def test_no_caveat_when_not_flagged():
    # flagged_text is None -> critique never flagged -> no caveat
    assert _maybe_unverified_caveat(FLAGGED, None) == FLAGGED


def test_no_caveat_when_content_changed_since_flag():
    # A later rewrite/regeneration/footer changed final_content -> suppress (the
    # flagged text is no longer what's shipping).
    rewritten = "A summit occurred in early March 2026; attendance is unconfirmed."
    assert _maybe_unverified_caveat(rewritten, FLAGGED) == rewritten


def test_idempotent_no_double_append():
    once = _maybe_unverified_caveat(FLAGGED, FLAGGED)
    twice = _maybe_unverified_caveat(once, once)
    # Already carries the marker -> not appended again.
    assert twice.count("couldn't confirm some specifics") == 1


def test_caveat_text_is_italic_footer_style():
    # Consistent with the existing confidence-footer markup (_(...)_).
    assert _UNVERIFIED_CAVEAT.strip().startswith("_(")
    assert _UNVERIFIED_CAVEAT.strip().endswith(")_")
