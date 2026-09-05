"""A canary whose threshold cannot be reached is not a canary (2026-09-04).

The digest health check warned above a 75% entail drop-rate — but it summed
EVERY rotated log file with no window at all, while the digest lengths beside it
used 7 days. Measured on the live logs: 12,975 checked and 6,775 dropped, a
lifetime 52% over 13,000 samples. A single catastrophic day moves that by
fractions of a point, so 75% was unreachable and the check could not fire on the
thing it exists to watch. Daily rates over the same period ran 49-57%, which is
the signal the lifetime mean was averaging away.

Two fixes: window the scan to match the rest of the check, and compare TODAY
against the trailing week, because a rate that JUMPS is what a gate or prompt
change does and an absolute bar on a weekly mean will not see it.
"""
from __future__ import annotations

import pytest

from app.monitors.health_checks import (
    DROP_RATE_STEP,
    _digest_health_verdict,
    entail_gate_totals,
)

LINE = "{date} 12:00:00,000 [INFO] app.monitors.deep_research []: " \
       "[entail-gate] topic: {checked} checked, 3 unsupported → {dropped} dropped\n"
HEALTHY = ([8000] * 40, 0)


def _log(tmp_path, days):
    """days: {date -> (checked, dropped)}"""
    p = tmp_path / "nova-app.log"
    p.write_text("".join(LINE.format(date=d, checked=c, dropped=dr)
                         for d, (c, dr) in days.items()), encoding="utf-8")
    return str(tmp_path / "nova-app.log*")


def test_lines_outside_the_window_are_not_counted(tmp_path):
    glob = _log(tmp_path, {"2026-09-04": (100, 50), "2026-06-01": (900, 800)})
    checked, dropped = entail_gate_totals(7, glob, today="2026-09-05")
    assert (checked, dropped) == (100, 50), "the old lifetime sum is back"


def test_a_wider_window_picks_the_older_lines_back_up(tmp_path):
    glob = _log(tmp_path, {"2026-09-04": (100, 50), "2026-06-01": (900, 800)})
    checked, _ = entail_gate_totals(3650, glob, today="2026-09-05")
    assert checked == 1000


def test_a_line_without_a_date_is_skipped_not_guessed(tmp_path):
    p = tmp_path / "nova-app.log"
    p.write_text("[entail-gate] topic: 500 checked, 1 unsupported → 400 dropped\n",
                 encoding="utf-8")
    assert entail_gate_totals(7, str(p), today="2026-09-05") == (0, 0)


def test_a_missing_log_is_not_an_error(tmp_path):
    assert entail_gate_totals(7, str(tmp_path / "nope*"), today="2026-09-05") == (0, 0)


def test_a_jump_today_is_caught_even_though_the_week_looks_fine():
    """The failure mode a weekly mean hides: a gate change lands and today's
    rate leaps while the trailing average barely twitches."""
    status, summary = _digest_health_verdict(
        *HEALTHY, checked=8000, dropped=4240,          # week at 53%
        recent_rate=0.53 + DROP_RATE_STEP + 0.01)
    assert status == "warning"
    assert "jumped" in summary and "today" in summary


def test_ordinary_daily_spread_does_not_trip_it():
    """Measured spread was 49-57%, so a couple of points must stay quiet."""
    status, _ = _digest_health_verdict(
        *HEALTHY, checked=8000, dropped=4240, recent_rate=0.57)
    assert status == "info"


def test_a_thin_day_is_not_evidence():
    """Under 100 checked, the caller passes None rather than a noisy ratio."""
    status, _ = _digest_health_verdict(*HEALTHY, checked=8000, dropped=4240,
                                       recent_rate=None)
    assert status == "info"


def test_a_thin_week_cannot_raise_the_step_warning():
    status, _ = _digest_health_verdict(*HEALTHY, checked=150, dropped=80,
                                       recent_rate=0.99)
    assert status == "info", "too few samples to call a step"


@pytest.mark.parametrize("avg,link,expect", [
    (1500, 0, "error"),      # substance collapsed
    (3000, 0, "warning"),    # thinning
    (8000, 5, "error"),      # link-only share over 10%
])
def test_the_original_verdicts_still_hold(avg, link, expect):
    status, _ = _digest_health_verdict([avg] * 40, link, 8000, 4240, recent_rate=0.53)
    assert status == expect
