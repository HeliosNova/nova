"""When a FORECAST line does not parse, the warning has to show the line (2026-09-07).

44 "FORECAST line present but not stored - mint format drift?" warnings in the
persisted log, 43 of them from the Storyline Tracker, 9 on 2026-09-06 alone
against 28 mints: roughly a quarter of the storyline forecasts are lost, and
not one of the 44 warnings says what the line looked like. The model's raw
output is not stored anywhere (the summary is cut at CHANGED and the STATE and
FORECAST lines are stripped from it), so the drift has been undiagnosable for
eleven days. The 300-token ceiling was the obvious suspect and it is refuted:
zero truncations at 300 in the same log.

So: the line itself, in the warning. Instrument first, widen the parser when
the shapes are on record.
"""
from __future__ import annotations

from app.core.forecasts import forecast_line_excerpt


def test_the_excerpt_is_the_forecast_line_itself():
    out = ("The story moved on.\n\nCHANGED: talks resumed\n"
           "STATE: ceasefire | holding\n"
           "FORECAST: Brent will trade above $90 through December 2026 | resolves 2026-12-31 | 0.7 confidence.")
    assert forecast_line_excerpt(out) == (
        "FORECAST: Brent will trade above $90 through December 2026 | resolves 2026-12-31 | 0.7 confidence.")


def test_the_excerpt_survives_bold_and_leading_space():
    out = "text\n  **FORECAST:** the vote passes | resolves 2026-11-05 | 0.6"
    assert forecast_line_excerpt(out) == "**FORECAST:** the vote passes | resolves 2026-11-05 | 0.6"


def test_the_excerpt_is_bounded_and_single_line():
    out = "FORECAST: " + "x" * 600 + "\nnext line"
    ex = forecast_line_excerpt(out)
    assert len(ex) <= 240 and "\n" not in ex


def test_no_line_means_an_empty_excerpt():
    assert forecast_line_excerpt("nothing here") == ""
    assert forecast_line_excerpt("") == ""
