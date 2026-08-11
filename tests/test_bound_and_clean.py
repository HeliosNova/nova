"""Deterministic digest bound: guarantee complete-sentence ending + length cap.

The final backstop against generation-truncation (mid-word endings when a pass
hits max_tokens) AND length overrun (model ignoring the soft char bound and
busting the storage cap). 2026-07-09.
"""

from __future__ import annotations

from app.monitors.deep_research import _bound_and_clean


class TestEndingCleanup:
    def test_midword_truncation_trimmed_to_last_sentence(self):
        t = ("**Lead:** The reactor came online (x.com). A second firm confirmed "
             "the result (y.com). The market reacted sharply to the news that allows")
        out = _bound_and_clean(t)
        assert out.rstrip().endswith("(y.com).")
        assert "that allows" not in out

    def test_already_clean_untouched(self):
        t = "**Lead:** Complete sentence one. Complete sentence two (z.com)."
        assert _bound_and_clean(t) == t

    def test_ends_with_citation_paren_is_clean(self):
        t = "The deal closed for $4B (reuters.com)"
        assert _bound_and_clean(t) == t

    def test_ends_with_question_or_exclamation(self):
        assert _bound_and_clean("Is this the turning point?") == "Is this the turning point?"


class TestLengthBound:
    def test_over_limit_trimmed_to_sentence_under_cap(self):
        body = ("This is a full sentence with enough words to matter. " * 300)  # ~15k chars
        out = _bound_and_clean(body, max_chars=10800)
        assert len(out) <= 10800
        assert out.rstrip().endswith(".")

    def test_under_limit_length_preserved(self):
        t = "Short and complete. Two sentences here."
        assert _bound_and_clean(t, max_chars=10800) == t

    def test_fits_storage_cap(self):
        body = ("Dense clause, with citations (a.com; b.com), and numbers 42.5%. " * 400)
        out = _bound_and_clean(body, max_chars=10800)
        assert len(out) < 12000    # under _RESULT_CAP


class TestSafety:
    def test_empty(self):
        assert _bound_and_clean("") == ""

    def test_no_boundary_left_as_is(self):
        t = "singlelongtokenwithnopunctuationatall" * 5
        out = _bound_and_clean(t)
        assert out  # doesn't crash or return empty
