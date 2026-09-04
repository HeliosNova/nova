"""Navigation menus stopped being evidence (2026-09-04).

The entail gate scores a claim against the best-matching WINDOW of a page, and a
site's section menu can win that competition: measured live, a claim about rate
expectations was scored against "Video / Big Business / So Expensive / View From
Above / Small Business / Authorized Account" — CNBC's nav — and another against
a "FREE ACCOUNT" masthead. Two of seventeen misses on one digest had chrome as
their entire evidence window.

_scrub_chrome was a substring blocklist ("subscribe now", "cookie settings"),
which is the name-the-forbidden-string pattern that keeps losing here: it needs a
new entry per site, forever. These two rules are structural instead.

The false positive is known and accepted: four consecutive short unpunctuated
lines with no digits are dropped whether they are a menu or a bare list of
company names. That costs a little evidence; leaving a menu in costs a dropped
sentence, which is worse.
"""
from __future__ import annotations

from app.monitors.deep_research import _scrub_chrome

NL = chr(10)
FACT = "The Federal Reserve held rates steady at 4 percent this week."


def test_a_section_menu_stops_being_evidence():
    body = NL.join(["Video", "Big Business", "So Expensive", "View From Above",
                    "Small Business", "Authorized Account", FACT])
    out = _scrub_chrome(body)
    assert out == FACT
    assert "So Expensive" not in out


def test_a_single_short_heading_survives():
    """One short line is a heading. A run of them is a menu."""
    body = NL.join(["Markets", FACT])
    assert _scrub_chrome(body) == NL.join(["Markets", FACT])


def test_three_short_lines_are_not_yet_a_menu():
    """Four, not three: a byline block or pull quote runs to three."""
    body = NL.join(["Markets", "By A Reporter", "New York", FACT])
    out = _scrub_chrome(body)
    assert "By A Reporter" in out and FACT in out


def test_digits_mark_a_run_as_content_not_chrome():
    """A column of short numeric lines is a table or a ticker list."""
    body = NL.join(["Apple 240", "Microsoft 410", "Nvidia 178", "Broadcom 355"])
    assert _scrub_chrome(body) == body


def test_punctuated_short_sentences_are_prose():
    body = NL.join(["The Fed held rates steady.", "Officials signalled patience.",
                    "Markets rallied.", "Traders were unmoved."])
    assert _scrub_chrome(body) == body


def test_a_short_all_caps_line_is_a_control_not_prose():
    body = NL.join(["FREE ACCOUNT", "The stock market looked unusually tranquil."])
    assert _scrub_chrome(body) == "The stock market looked unusually tranquil."


def test_an_all_caps_sentence_long_enough_to_be_prose_survives():
    """The rule is for controls, not for shouting."""
    loud = "BREAKING: THE CENTRAL BANK HAS RAISED ITS POLICY RATE AGAIN TODAY."
    assert _scrub_chrome(loud) == loud


def test_the_old_blocklist_still_applies():
    body = NL.join(["Advertisement", "Subscribe now for full access", FACT])
    assert _scrub_chrome(body) == FACT
