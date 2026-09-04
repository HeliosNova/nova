"""Forecast confidence comes from several samples, not one sentence (2026-09-04).

Nova took a single verbalized `0.x` out of one generation — the weakest
estimator in the literature — and its legacy record is 15 points overconfident
(hit 0.60 against a stated 0.75). Confidence is now the mean of that stated
number and k independently sampled probabilities, and the spread is stored so
"does disagreement predict error" can be answered from data later.

The regime string bumps with the change, which is the whole point of having
one: the next record is attributable to this estimator rather than averaged
into the old one.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core import forecasts as fc

LINE = ("Nvidia ships Rubin to three Azure regions | resolves 2026-12-31 | 0.8")


def _samples(*values):
    """Fake the model returning a given probability on each call."""
    return AsyncMock(side_effect=[f'{{"probability": {v}}}' for v in values])


@pytest.mark.asyncio
async def test_confidence_is_the_mean_of_stated_and_sampled(db):
    with patch.object(fc.llm, "invoke_nothink", _samples(0.4, 0.4, 0.4)):
        fid = await fc.parse_and_store_forecast_ensembled(db, f"FORECAST: {LINE}", k=3)
    row = db.fetchone("SELECT confidence, conf_spread, regime FROM forecasts WHERE id = ?", (fid,))
    # stated 0.8 plus three samples of 0.4 -> (0.8 + 1.2) / 4
    assert row["confidence"] == pytest.approx(0.5, abs=0.01)
    assert row["conf_spread"] == pytest.approx(0.0, abs=0.01)
    assert row["regime"] == fc.REGIME


@pytest.mark.asyncio
async def test_spread_records_disagreement(db):
    with patch.object(fc.llm, "invoke_nothink", _samples(0.2, 0.5, 0.9)):
        fid = await fc.parse_and_store_forecast_ensembled(db, f"FORECAST: {LINE}", k=3)
    row = db.fetchone("SELECT conf_spread FROM forecasts WHERE id = ?", (fid,))
    assert row["conf_spread"] == pytest.approx(0.7, abs=0.01)


@pytest.mark.asyncio
async def test_an_unreachable_model_keeps_the_stated_confidence(db):
    """Sampling may only improve an estimate; it must never block a mint."""
    with patch.object(fc.llm, "invoke_nothink", AsyncMock(side_effect=RuntimeError("ollama down"))):
        fid = await fc.parse_and_store_forecast_ensembled(db, f"FORECAST: {LINE}", k=3)
    assert fid
    row = db.fetchone("SELECT confidence, conf_spread FROM forecasts WHERE id = ?", (fid,))
    assert row["confidence"] == pytest.approx(0.8, abs=0.01)
    assert row["conf_spread"] is None


@pytest.mark.asyncio
async def test_garbage_samples_are_ignored_not_averaged(db):
    with patch.object(fc.llm, "invoke_nothink",
                      AsyncMock(side_effect=['{"probability": 1.7}', "not json",
                                             '{"probability": 0.6}'])):
        fid = await fc.parse_and_store_forecast_ensembled(db, f"FORECAST: {LINE}", k=3)
    row = db.fetchone("SELECT confidence FROM forecasts WHERE id = ?", (fid,))
    # only 0.6 survives validation: (0.8 + 0.6*3) / 4
    assert row["confidence"] == pytest.approx(0.65, abs=0.01)


@pytest.mark.asyncio
async def test_no_forecast_line_mints_nothing_and_costs_no_samples(db):
    calls = AsyncMock()
    with patch.object(fc.llm, "invoke_nothink", calls):
        assert await fc.parse_and_store_forecast_ensembled(db, "FORECAST: none") is None
    calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_claim_and_its_date_still_come_from_the_line(db):
    with patch.object(fc.llm, "invoke_nothink", _samples(0.5, 0.5, 0.5)):
        fid = await fc.parse_and_store_forecast_ensembled(db, f"FORECAST: {LINE}", k=3)
    row = db.fetchone("SELECT claim, resolves_at FROM forecasts WHERE id = ?", (fid,))
    assert row["claim"].startswith("Nvidia ships Rubin")
    assert row["resolves_at"].startswith("2026-12-31")


def test_the_regime_bumped_and_keeps_its_history():
    assert fc.REGIME == "2026-09-04-ensembled"
    assert fc.REGIME in fc.REGIME_HISTORY and fc.REGIME_LEGACY in fc.REGIME_HISTORY
    assert fc.REGIME_HISTORY[0] == fc.REGIME, "newest first"


def test_the_spread_column_exists(db):
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(forecasts)")}
    assert "conf_spread" in cols and "regime" in cols
