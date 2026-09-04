"""Is a stated confidence worth anything? (2026-09-04)

"Brier 0.25 = coin flip" was the yardstick, and it is the wrong one. The
baseline that matters is predicting the record's OWN base rate for every claim,
which scores hit_rate * (1 - hit_rate). Measured over Nova's 105 resolved
forecasts: stated confidence scored 0.2541, always saying 0.60 scored 0.2400.
The numbers were worse than ignoring the claim. Fitting a one-parameter
shrinkage on those same outcomes reached 0.2425 leave-one-out - still short, so
recalibration cannot rescue it either (scripts/forecast_shrinkage.py).

Skill states that in one number. Above zero the confidence carried information;
at or below zero it did not.

The second half of this file guards a defect the regime stamp introduced the
same day: scoping the calibration record to the CURRENT regime meant that the
moment the estimator changed, the mint prompt lost its feedback until 20 new
forecasts resolved - months at these horizons.
"""
from __future__ import annotations

import pytest

from app.core import forecasts as fc


class _DB:
    """Just enough store for calibration()."""

    def __init__(self, rows):
        self._rows = rows

    def fetchall(self, _q, _a=()):
        return self._rows


def _rows(pairs, regime=fc.REGIME):
    return [{"confidence": c, "status": "hit" if o else "miss", "regime": regime}
            for c, o in pairs]


def test_confidence_that_tracks_outcomes_scores_positive_skill():
    """Confident-and-right, unsure-and-wrong: that is information."""
    pairs = [(0.9, 1)] * 20 + [(0.1, 0)] * 20
    cal = fc.calibration(_DB(_rows(pairs)), regime=None)
    assert cal["skill"] > 0.5, cal


def test_a_flat_overconfident_record_scores_no_skill():
    """Nova's actual shape: everything stated at 0.75, 60% of it happens."""
    pairs = [(0.75, 1)] * 60 + [(0.75, 0)] * 40
    cal = fc.calibration(_DB(_rows(pairs)), regime=None)
    assert cal["hit_rate"] == pytest.approx(0.6)
    assert cal["skill"] <= 0, "a constant confidence cannot beat the base rate"
    assert cal["base_brier"] == pytest.approx(0.24, abs=0.001)


def test_skill_is_reported_alongside_the_old_numbers():
    """Brier and gap keep working; skill is added, not substituted."""
    cal = fc.calibration(_DB(_rows([(0.8, 1), (0.8, 0)])), regime=None)
    assert {"n", "hit_rate", "mean_conf", "gap", "brier",
            "base_brier", "skill"} <= set(cal)


def test_an_all_hit_record_cannot_divide_by_zero():
    """base = p(1-p) is zero when everything resolved the same way."""
    cal = fc.calibration(_DB(_rows([(0.9, 1)] * 5)), regime=None)
    assert cal["skill"] is None


def test_the_note_falls_back_when_the_current_regime_is_too_young(monkeypatch):
    """The defect the regime stamp introduced: a fresh estimator has no record,
    and silence is not the honest answer when a legacy one exists."""
    legacy = _rows([(0.75, 1)] * 60 + [(0.75, 0)] * 40, regime=fc.REGIME_LEGACY)

    calls = {"n": 0}
    real = fc.calibration

    def _cal(db, **kw):
        calls["n"] += 1
        if kw.get("regime", fc.REGIME) is not None:
            return None                      # current regime: too few resolved
        return real(db, **kw)

    monkeypatch.setattr(fc, "calibration", _cal)
    monkeypatch.setattr(fc, "calibration_buckets", lambda db, **kw: [(0.8, 0.62, 40)])
    note = fc.global_calibration_note(_DB(legacy), min_n=20)
    assert note and "RETIRED estimator" in note, note
    assert calls["n"] == 2, "it must try the current regime first"
    # the retired regime's NUMBERS must not become advice — only its lesson
    assert "delivered hit rate" not in note
    assert "belongs at that number" not in note


def test_the_note_says_plainly_when_there_is_no_edge(monkeypatch):
    monkeypatch.setattr(fc, "calibration_buckets", lambda db, **kw: [(0.8, 0.62, 40)])
    rows = _rows([(0.75, 1)] * 60 + [(0.75, 0)] * 40)
    note = fc.global_calibration_note(_DB(rows), min_n=20)
    assert "NO measured edge" in note
    assert "skill" in note


def test_the_note_stays_quiet_about_edge_when_there_is_one(monkeypatch):
    monkeypatch.setattr(fc, "calibration_buckets", lambda db, **kw: [(0.9, 0.95, 20)])
    rows = _rows([(0.9, 1)] * 20 + [(0.1, 0)] * 20)
    note = fc.global_calibration_note(_DB(rows), min_n=20)
    assert "NO measured edge" not in note
