"""Forecasting from zero (2026-09-22): a validator at mint, a criterion the
judge is bound to, a new regime.

The record was reset to zero the same day. Before it: 969 forecasts, 94 hit /
50 miss with no skill over the base rate, 48 unresolvable, 17 restatements,
and a judge left to decide for itself what "settles" a free-text claim.
OpenForecaster's largest lever was a validator that kept ~7% of candidate
questions: one binary outcome, an explicit resolution criterion, a consistent
deadline, no leakage. That validator now sits in front of every mint.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core import forecasts


def _validator(accept=True, claim=None, criterion="Settles TRUE if the Fed's statement on "
               "2026-10-29 keeps the target range at 4.00-4.25%; FALSE otherwise", reason="ok"):
    payload = {"accept": accept, "reason": reason, "criterion": criterion}
    if claim is not None:
        payload["claim"] = claim
    return json.dumps(payload)


LINE = ("STATE: holding\nFORECAST: The Fed holds its policy rate at the October 2026 meeting "
        "| resolves 2026-10-30 | 0.7 confidence")


@pytest.mark.asyncio
async def test_an_accepted_forecast_stores_the_criterion_under_the_new_regime(db):
    calls = []

    async def fake(messages, **kw):
        calls.append(messages[0]["content"])
        if "Decide whether it is a usable forecast" in messages[0]["content"]:
            return _validator()
        return json.dumps({"probability": 0.7})

    with patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(side_effect=fake)):
        fid = await forecasts.parse_and_store_forecast_ensembled(db, LINE, source_monitor="t")
    row = db.fetchone("SELECT claim, criterion, regime, status FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "open"
    assert row["criterion"].startswith("Settles TRUE if")
    assert row["regime"] == forecasts.REGIME == "2026-09-22-criterion"
    # the confidence samples were shown the criterion, not just the claim
    conf_prompts = [c for c in calls if "Estimate the probability" in c]
    assert conf_prompts and all("RESOLUTION CRITERION" in c for c in conf_prompts)


@pytest.mark.asyncio
async def test_a_rejected_candidate_mints_nothing(db):
    async def fake(messages, **kw):
        if "Decide whether it is a usable forecast" in messages[0]["content"]:
            return _validator(accept=False, reason="restates announced guidance")
        return json.dumps({"probability": 0.7})

    with patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(side_effect=fake)):
        fid = await forecasts.parse_and_store_forecast_ensembled(db, LINE, source_monitor="t")
    # A refusal is not "nothing parsed": the callers' format-drift warning keys
    # on None, and a validator refusal must not count as a parser loss.
    assert fid == forecasts.REJECTED and not fid and fid is not None
    assert db.fetchone("SELECT count(*) AS c FROM forecasts")["c"] == 0


@pytest.mark.asyncio
async def test_an_unreachable_validator_mints_unvalidated_rather_than_losing_the_forecast(db):
    async def fake(messages, **kw):
        if "Decide whether it is a usable forecast" in messages[0]["content"]:
            raise RuntimeError("model busy")
        return json.dumps({"probability": 0.7})

    with patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(side_effect=fake)):
        fid = await forecasts.parse_and_store_forecast_ensembled(db, LINE, source_monitor="t")
    row = db.fetchone("SELECT criterion, status FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "open" and (row["criterion"] or "") == ""


@pytest.mark.asyncio
async def test_a_rewrite_that_invents_or_loses_text_is_discarded_for_the_original(db):
    async def fake(messages, **kw):
        if "Decide whether it is a usable forecast" in messages[0]["content"]:
            return _validator(claim="x")                    # lost the claim
        return json.dumps({"probability": 0.7})

    with patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(side_effect=fake)):
        fid = await forecasts.parse_and_store_forecast_ensembled(db, LINE, source_monitor="t")
    row = db.fetchone("SELECT claim FROM forecasts WHERE id=?", (fid,))
    assert "Fed holds its policy rate" in row["claim"]


@pytest.mark.asyncio
async def test_the_judge_is_bound_to_the_criterion(db):
    fid = forecasts.create_forecast(db, "The merger closes by the end of the month", days=1,
                                    confidence=0.7,
                                    criterion="Settles TRUE if an SEC 8-K reports completion by the date")
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour'), "
               "created_at = datetime('now','-20 days') WHERE id=?", (fid,))
    seen = {}

    async def fake(messages, **kw):
        seen["prompt"] = messages[0]["content"]
        return json.dumps({"verdict": "unresolvable", "evidence_date": None, "reason": "no filing"})

    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value="- something [reuters.com]")), \
         patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(side_effect=fake)), \
         patch("app.core.forecasts.llm.extract_json_object",
               side_effect=lambda raw: json.loads(raw)):
        await forecasts.resolve_due(db)
    assert "RESOLUTION CRITERION" in seen["prompt"]
    assert "SEC 8-K" in seen["prompt"]


def test_regime_history_leads_with_the_new_regime():
    assert forecasts.REGIME_HISTORY[0] == forecasts.REGIME == "2026-09-22-criterion"
    assert "2026-09-04-ensembled" in forecasts.REGIME_HISTORY
