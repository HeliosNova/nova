"""A guard and a prompt written against each other (2026-09-04).

Owner-reported live: monitor citations reading "(deep analysis)". The stripper
added on 2026-09-01 required a COLON, so it removed "(deep analysis: the Fed
pivot)" but not "(deep analysis)". The same change told the synthesis prompt
never to write "(deep analysis: …)" — naming the exact spelling. The model
obeyed the letter and switched to the form the stripper could not see.

Measured on stored digests: carrying the artifact went from about 10% before
that deploy to 34% on 09-02, 45% on 09-03 and 55% on 09-04. Across three days
of digests the widened stripper removes 352 pseudo-citations from 55 of 157.

The rule is now semantic rather than a spelling: a short parenthetical that
points at "analysis" and carries no domain token is the digest citing its own
thinking. One that names a source is a real attribution and survives.
"""
from __future__ import annotations

import inspect

import pytest

from app.monitors.deep_research import _strip_pseudo_citations


@pytest.mark.parametrize("text", [
    "Growth is slowing (deep analysis: the Fed pivot).",
    "Growth is slowing (deep analysis).",
    "Growth is slowing (analysis).",
    "Growth is slowing (our analysis).",
    "Growth is slowing (internal analysis of the curve).",
    "Growth is slowing (per deep analyses).",
])
def test_every_spelling_of_self_citation_is_removed(text):
    out, n = _strip_pseudo_citations(text)
    assert n == 1, f"not stripped: {text}"
    assert "analys" not in out
    assert out.startswith("Growth is slowing")


@pytest.mark.parametrize("text", [
    "The Fed cut rates (reuters.com).",
    "Per the analysis by reuters.com, growth slowed (reuters.com).",
    "Growth is slowing (analysis by ft.com).",
])
def test_real_attributions_survive(text):
    out, n = _strip_pseudo_citations(text)
    assert n == 0 and out == text


def test_prose_mentioning_analysis_is_untouched():
    text = ("A long parenthetical that merely discusses analysis of policy at great "
            "length and should be left alone because it is prose, not a citation slot.")
    out, n = _strip_pseudo_citations(text)
    assert n == 0 and out == text


def test_multiple_artifacts_in_one_digest_are_all_counted():
    text = ("Rates fell (deep analysis). Growth slowed (analysis). "
            "The Fed cut (reuters.com).")
    out, n = _strip_pseudo_citations(text)
    assert n == 2
    assert "(reuters.com)" in out and "analys" not in out


def test_empty_and_none_are_safe():
    assert _strip_pseudo_citations("") == ("", 0)
    assert _strip_pseudo_citations(None) == (None, 0)


def test_the_prompt_no_longer_names_the_spelling_to_avoid():
    """Naming the exact forbidden string taught the model which form to use."""
    from app.monitors import deep_research as dr
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "(deep analysis: …)" not in src, \
        "do not hand the model an exact string to route around"
    assert "never point at" in src, "the instruction must be semantic"
