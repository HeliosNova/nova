"""Regression tests for the 2026-08-16 digest cosmetic-cleanup fixes in
``deep_research._tidy_citations``.

Two artifacts leaked to Discord in the qwen3.8 Space digest (result 7192):
  - a PLACEHOLDER article title — the model wrote ``titled "new article"`` when it
    never got the real title;
  - a dropped token inside italics left a ``*Artemis *`` trailing-space artifact
    (the model wrote ``*Artemis II*`` and the ``II`` was lost).

Both are stripped/normalized without harming real titles, bold labels, bullets,
arithmetic, or the pre-existing citation cleanup.
"""
from __future__ import annotations

from app.monitors.deep_research import _tidy_citations


def test_placeholder_title_clause_stripped():
    s = ('This discovery has been published in a new article titled "new article" '
         'within *Frontiers in Astronomy* (science.nasa.gov).')
    out = _tidy_citations(s)
    assert 'titled "new article"' not in out
    assert '"new article"' not in out
    # the real content survives, cleanly
    assert "published in a new article within" in out
    assert "Frontiers in Astronomy" in out
    assert "science.nasa.gov" in out


def test_placeholder_title_variants_stripped():
    for bad in ('titled "the study"', "titled 'a paper'", 'entitled "new report"',
                'called "this analysis"', 'titled "untitled"'):
        s = f"The work was {bad} in the journal."
        assert bad not in _tidy_citations(s), bad


def test_real_titles_preserved():
    keep = [
        'a study titled "The Immune Response to X" today',
        'a paper titled "A New Analysis of Reconnection" (nasa.gov)',
        'the book titled "Article One of the Constitution"',
    ]
    for s in keep:
        assert _tidy_citations(s) == s, s


def test_emphasis_trailing_space_collapsed():
    assert _tidy_citations("NASA’s *Artemis * mission") == "NASA’s *Artemis* mission"
    assert _tidy_citations("from *Artemis * (if launched)") == "from *Artemis* (if launched)"
    # multiple occurrences mid-line (a line-initial '*' is left alone as a possible bullet)
    assert _tidy_citations("the *Artemis * and *Orion * capsules") == \
        "the *Artemis* and *Orion* capsules"


def test_emphasis_fix_leaves_markdown_intact():
    keep = [
        "* **Lead Development:** the thing happened *fast* today.",  # bullet + bold + clean italic
        "**Secondary Developments**",                                # double-asterisk bold
        "the *Rosalind Franklin* rover launches",                    # clean italic, no trailing space
        "compute 3 * 4 * 5 = 60",                                    # arithmetic, not emphasis
        "* Something *emph* here",                                   # bullet then inline italic
    ]
    for s in keep:
        assert _tidy_citations(s) == s, s


def test_citation_dollar_strip_unregressed():
    # the pre-existing behavior must still hold after adding steps 3 & 4
    assert _tidy_citations("($quiverquant.com$)") == "(quiverquant.com)"


def test_empty_and_plain_text_passthrough():
    assert _tidy_citations("") == ""
    assert _tidy_citations("A normal sentence with no artifacts.") == \
        "A normal sentence with no artifacts."
