"""A track record only describes the process that produced it (2026-09-04).

On 2026-09-04 every one of Nova's 103 resolved forecasts had been minted under
the old rules — 30-day clamp, resolver searching the raw claim with no date
filter, judge never told its window. The 0.60 hit rate against 0.75 stated
confidence was therefore a verdict on code that no longer runs, and it was
being fed back into the mint prompts as if it described current behaviour.
Calibration now scores within a scoring regime, so a change to how forecasts
are made or graded is measurable instead of averaged into history.
"""
from __future__ import annotations

import pytest

from app.core.forecasts import (
    REGIME,
    REGIME_LEGACY,
    accuracy,
    calibration,
    create_forecast,
)


def _resolved(db, conf, status, *, regime=None, created="2026-09-03 12:00:00"):
    db.execute(
        "INSERT INTO forecasts (claim, storyline_key, confidence, resolves_at, status, "
        "created_at, regime) VALUES (?, 'k', ?, datetime('now','-1 day'), ?, ?, ?)",
        (f"a claim about something {conf}{status}{created}", conf, status, created, regime))


def test_new_forecasts_are_stamped_with_the_current_regime(db):
    fid = create_forecast(db, "Nvidia ships Rubin to three Azure regions", days=30, confidence=0.6)
    assert fid
    assert db.fetchone("SELECT regime FROM forecasts WHERE id = ?", (fid,))["regime"] == REGIME


def test_legacy_rows_are_excluded_from_the_current_record(db):
    # old regime: overconfident and wrong
    for _ in range(4):
        _resolved(db, 0.9, "miss", created="2026-08-20 12:00:00")
    # current regime: well calibrated
    _resolved(db, 0.6, "hit", regime=REGIME)
    _resolved(db, 0.6, "miss", regime=REGIME)

    now = accuracy(db)
    assert now["resolved"] == 2 and now["rate"] == 0.5 and now["regime"] == REGIME

    old = accuracy(db, regime=REGIME_LEGACY)
    assert old["resolved"] == 4 and old["rate"] == 0.0

    pooled = accuracy(db, regime=None)
    assert pooled["resolved"] == 6, "pooling stays available, it is just not the default"


def test_calibration_is_scoped_too(db):
    for _ in range(6):
        _resolved(db, 0.9, "miss", created="2026-08-20 12:00:00")
    _resolved(db, 0.6, "hit", regime=REGIME)
    _resolved(db, 0.6, "hit", regime=REGIME)

    cur = calibration(db, min_n=1)
    assert cur["n"] == 2 and cur["hit_rate"] == 1.0
    assert cur["gap"] < 0, "current regime is underconfident, not overconfident"

    legacy = calibration(db, min_n=1, regime=REGIME_LEGACY)
    assert legacy["n"] == 6 and legacy["gap"] > 0.5, "the old regime was badly overconfident"


def test_unstamped_rows_are_dated_into_the_right_regime(db):
    """No backfill required: NULL regime is resolved by created_at."""
    _resolved(db, 0.8, "hit", regime=None, created="2026-08-01 00:00:00")   # legacy
    _resolved(db, 0.8, "hit", regime=None, created="2026-09-03 00:00:00")   # current
    assert accuracy(db)["resolved"] == 1
    assert accuracy(db, regime=REGIME_LEGACY)["resolved"] == 1


def test_the_column_and_index_exist_in_a_fresh_schema(db):
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(forecasts)")}
    assert "regime" in cols
    idx = {r["name"] for r in db.fetchall("PRAGMA index_list(forecasts)")}
    assert any("regime" in n for n in idx)


def test_the_prompt_feedback_uses_the_scoped_record(db):
    """global_calibration_note must not teach from the retired regime."""
    from app.core.forecasts import global_calibration_note
    for _ in range(30):
        _resolved(db, 0.95, "miss", created="2026-08-10 12:00:00")
    assert global_calibration_note(db, min_n=20) is None, \
        "30 legacy outcomes must not become advice about how Nova forecasts now"
