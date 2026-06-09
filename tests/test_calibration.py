"""Tests for Conformal Abstention thresholds (app.core.calibration)."""

from __future__ import annotations

import pytest

from app.core import calibration
from app.database import get_db, _instances


@pytest.fixture(autouse=True)
def _reset_calib_cache():
    calibration.invalidate()
    yield
    calibration.invalidate()


@pytest.fixture
def reflexion_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "calib.db"))
    monkeypatch.setenv("ENABLE_CONFORMAL_ABSTENTION", "true")
    from app.config import reset_config
    reset_config()
    _instances.clear()
    db = get_db()
    db.init_schema()
    yield db
    db.close()
    _instances.clear()


def _seed_reflexions(db, scores):
    for s in scores:
        db.execute(
            "INSERT INTO reflexions (task_summary, reflection, outcome, quality_score, created_at) "
            "VALUES ('t', 'r', 'success', ?, CURRENT_TIMESTAMP)",
            (float(s),),
        )


def test_defaults_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_CONFORMAL_ABSTENTION", "false")
    from app.config import reset_config
    reset_config()
    calibration.invalidate()
    thr = calibration.get_thresholds()
    assert thr.is_default is True
    assert thr.tier1_low == 0.40
    assert thr.tier1_high == 0.60
    assert thr.tier2 == 0.50
    assert thr.tier3 == 0.60


def test_defaults_when_too_few_reflexions(reflexion_db):
    _seed_reflexions(reflexion_db, [0.5, 0.6, 0.4])  # 3 < min 30
    thr = calibration.get_thresholds()
    assert thr.is_default is True
    assert thr.n_samples == 3


def test_defaults_when_distribution_too_tight(reflexion_db):
    # 50 reflexions, all between 0.69 and 0.71 — std < 0.05
    _seed_reflexions(reflexion_db, [0.7] * 50)
    thr = calibration.get_thresholds()
    assert thr.is_default is True


def test_thresholds_computed_from_real_distribution(reflexion_db):
    # Seed a varied distribution n=50
    scores = [i / 49.0 for i in range(50)]  # 0.0, ..., 1.0
    _seed_reflexions(reflexion_db, scores)
    thr = calibration.get_thresholds()
    assert thr.is_default is False
    assert thr.n_samples == 50
    # Ordering invariants
    assert 0.0 <= thr.tier1_low <= 1.0
    assert thr.tier1_high > thr.tier1_low
    assert 0.0 <= thr.tier2 <= 1.0
    assert 0.0 <= thr.tier3 <= 1.0


def test_cache_returns_same_object(reflexion_db):
    # Two calls within TTL — same FooterThresholds (cache hit)
    _seed_reflexions(reflexion_db, [i / 49.0 for i in range(50)])
    a = calibration.get_thresholds()
    b = calibration.get_thresholds()
    assert a is b


def test_invalidate_forces_recompute(reflexion_db):
    _seed_reflexions(reflexion_db, [0.5] * 50)  # tight, so default
    a = calibration.get_thresholds()
    assert a.is_default is True
    # Add varied scores
    _seed_reflexions(reflexion_db, [i / 49.0 for i in range(50)])
    calibration.invalidate()
    b = calibration.get_thresholds()
    assert b.is_default is False


def test_percentile_helper_edges():
    # _percentile is private but stable behavior
    assert calibration._percentile([], 0.5) == 0.0
    assert calibration._percentile([0.7], 0.5) == 0.7
    assert calibration._percentile([0.0, 1.0], 0.5) == 0.5
