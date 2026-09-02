"""Forecasting discipline (audit 2026-09-01).

Measured before the change: Brier 0.253 vs 0.250 coin flip; 506/723 forecasts
clamped to exactly 30 days while 110 open ones carried a later in-claim
deadline; 8/99 resolutions cited evidence that predated the forecast; guidance
("Micron guided for $50B") graded as a hit; 15 clusters of near-duplicate open
claims; the calibration note reached 4 of 29 dossier families.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import forecasts


NOW = datetime(2026, 9, 1, 12, 0, 0)


# --- deadlines inside the claim --------------------------------------------

@pytest.mark.parametrize("claim, expected", [
    ("PTC will win FDA accelerated approval for ST-920 by Q4 2026", datetime(2026, 12, 31)),
    ("India-UK bilateral trade will exceed $56 billion in 2027", datetime(2027, 12, 31)),
    ("The 30-year yield will remain above 5% through November 2026", datetime(2026, 11, 30)),
    ("The deal will close by March 15, 2027", datetime(2027, 3, 15)),
    ("Launch will slip to mid-2027", datetime(2027, 6, 30)),
    ("Bitcoin will break above $65,000 within 14 days", NOW + timedelta(days=14)),
    ("Omnichain audits will finish by Q1 2027", datetime(2027, 3, 31)),
    ("Chang'e-7 will launch on or before December 5", datetime(2026, 12, 5)),
])
def test_claim_deadline_parses_explicit_deadlines(claim, expected):
    assert forecasts.claim_deadline(claim, now=NOW) == expected


def test_claim_deadline_ignores_past_context_dates():
    # July 2026 is context (already past at NOW); the only deadline is 14 days.
    got = forecasts.claim_deadline("Bitcoin will retest $65,000 within 14 days as it did in July 2026", now=NOW)
    assert got == NOW + timedelta(days=14)


def test_claim_deadline_none_without_a_date():
    assert forecasts.claim_deadline("Brent crude will exceed $90 per barrel", now=NOW) is None


# --- minting ----------------------------------------------------------------

def _resolves(db, fid) -> datetime:
    row = db.fetchone("SELECT resolves_at FROM forecasts WHERE id=?", (fid,))
    return datetime.strptime(row["resolves_at"][:19], "%Y-%m-%d %H:%M:%S")


def test_parse_accepts_explicit_resolution_date(db):
    fid = forecasts.parse_and_store_forecast(
        db, "CHANGED: x\nFORECAST: The widget ships to customers | resolves 2026-12-15 | 0.7 confidence")
    assert fid
    assert _resolves(db, fid).date() == datetime(2026, 12, 15).date()


def test_parse_legacy_days_no_longer_clamped_to_30(db):
    fid = forecasts.parse_and_store_forecast(db, "FORECAST: The consortium publishes its charter | 90 days | 0.6")
    assert fid
    delta = _resolves(db, fid) - datetime.utcnow()
    assert 88 <= delta.days <= 91


def test_horizon_clamped_to_365_days(db):
    fid = forecasts.create_forecast(db, "Fusion power plant reaches net gain commercially", days=900, confidence=0.5)
    delta = _resolves(db, fid) - datetime.utcnow()
    assert 363 <= delta.days <= 366


def test_in_claim_deadline_extends_stored_horizon(db):
    fid = forecasts.parse_and_store_forecast(
        db, "FORECAST: PTC Therapeutics wins FDA accelerated approval for ST-920 by Q4 2026 | 30 days | 0.65")
    assert _resolves(db, fid).date() == datetime(2026, 12, 31).date()


def test_near_duplicate_open_claim_is_restated_not_minted(db):
    a = forecasts.create_forecast(
        db, "The Federal Reserve will announce a 25-basis-point rate hike at its September meeting",
        days=30, confidence=0.7, storyline_key="dossier:finance")
    b = forecasts.create_forecast(
        db, "The Federal Reserve will announce a 25 basis point rate hike at the September meeting",
        days=30, confidence=0.8, storyline_key="dossier:finance")
    rows = {r["id"]: dict(r) for r in db.fetchall("SELECT * FROM forecasts")}
    assert rows[a]["status"] == "open"
    assert rows[b]["status"] == "restated" and rows[b]["resolution"] == f"restates #{a}"
    assert forecasts.list_due(db) == []
    # a different bet in the same family is still minted
    c = forecasts.create_forecast(
        db, "The 30-year Treasury yield will close above 5.2% on at least five sessions in October",
        days=30, confidence=0.6, storyline_key="dossier:finance")
    assert rows.get(c) is None and db.fetchone("SELECT status FROM forecasts WHERE id=?", (c,))["status"] == "open"


# --- evidence filtering -----------------------------------------------------

def _res(url, title, date="", snippet="s"):
    return SimpleNamespace(url=url, title=title, published_date=date, snippet=snippet)


def test_filter_drops_pre_forecast_and_junk_evidence():
    created = "2026-08-15 10:00:00"
    results = [
        _res("https://reuters.com/a", "After", "2026-08-20"),
        _res("https://reuters.com/b", "Before", "2025-12-09"),
        _res("https://reuters.com/c", "Undated"),
        _res("https://reuters.com/d", "Relative", "3 days ago"),
    ]
    with patch("app.core.source_authority.authority", return_value=0.9):
        kept, old, junk = forecasts._filter_evidence(results, created)
    assert [r.title for r, _ in kept] == ["After", "Undated", "Relative"]
    assert old == 1 and junk == 0
    with patch("app.core.source_authority.authority", return_value=0.1):
        kept, old, junk = forecasts._filter_evidence(results[:1], created)
    assert kept == [] and junk == 1


# --- grading ----------------------------------------------------------------

def _due(db, claim="The merger closes by the end of the month", created_days_ago=20):
    fid = forecasts.create_forecast(db, claim, days=1, confidence=0.7)
    db.execute("UPDATE forecasts SET resolves_at = datetime('now','-1 hour'), "
               "created_at = datetime('now', ?) WHERE id=?", (f"-{created_days_ago} days", fid))
    return fid


@pytest.mark.asyncio
async def test_hit_with_pre_window_evidence_date_is_rejected(db):
    fid = _due(db)
    evidence = "- The merger closed (2026-06-01) [reuters.com]: closed last quarter"
    verdict = {"verdict": "hit", "evidence_date": "2026-06-01", "reason": "closed"}
    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value=evidence)), \
         patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(return_value="{}")), \
         patch("app.core.forecasts.llm.extract_json_object", return_value=verdict):
        await forecasts.resolve_due(db)
    row = db.fetchone("SELECT status, attempts FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "open" and row["attempts"] == 1


@pytest.mark.asyncio
async def test_hit_without_date_when_dated_evidence_exists_is_rejected(db):
    fid = _due(db)
    evidence = "- The merger closed (2026-08-28) [reuters.com]: closed"
    verdict = {"verdict": "hit", "evidence_date": None, "reason": "closed"}
    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value=evidence)), \
         patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(return_value="{}")), \
         patch("app.core.forecasts.llm.extract_json_object", return_value=verdict):
        await forecasts.resolve_due(db)
    assert db.fetchone("SELECT status FROM forecasts WHERE id=?", (fid,))["status"] == "open"


@pytest.mark.asyncio
async def test_hit_with_in_window_date_is_accepted_and_stamped(db):
    fid = _due(db)
    evidence = "- The merger closed (2026-08-28) [reuters.com]: closed"
    verdict = {"verdict": "hit", "evidence_date": "2026-08-28", "reason": "closed on the 28th"}
    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value=evidence)), \
         patch("app.core.forecasts.llm.invoke_nothink", AsyncMock(return_value="{}")), \
         patch("app.core.forecasts.llm.extract_json_object", return_value=verdict):
        await forecasts.resolve_due(db)
    row = db.fetchone("SELECT status, resolution FROM forecasts WHERE id=?", (fid,))
    assert row["status"] == "hit" and row["resolution"].startswith("[2026-08-28]")


@pytest.mark.asyncio
async def test_resolver_prompt_carries_window_and_outcome_rule(db):
    fid = _due(db)
    captured = {}

    async def _fake(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        captured.update(kwargs)
        return '{"verdict": "unresolvable", "evidence_date": null, "reason": "n/a"}'

    with patch("app.core.forecasts._gather_evidence", AsyncMock(return_value="- x (undated) [a.com]")), \
         patch("app.core.forecasts.llm.invoke_nothink", _fake):
        await forecasts.resolve_due(db)
    p = captured["prompt"]
    assert "it resolves on" in p and "are NOT outcomes" in p and "dated before" in p
    assert captured.get("json_schema") and captured.get("max_tokens", 0) >= 200
    assert db.fetchone("SELECT status FROM forecasts WHERE id=?", (fid,))["status"] == "unresolvable"


# --- global calibration -----------------------------------------------------

def test_global_calibration_note_needs_a_sample_then_reports_buckets(db):
    assert forecasts.global_calibration_note(db) is None
    i = 0
    for conf, hits, n in ((0.6, 5, 10), (0.8, 6, 10), (0.9, 8, 10)):
        for k in range(n):
            i += 1
            fid = forecasts.create_forecast(db, f"distinct claim number {i} about topic {i * 7}",
                                            days=5, confidence=conf, storyline_key=f"k{i}")
            db.execute("UPDATE forecasts SET status=? WHERE id=?", ("hit" if k < hits else "miss", fid))
    note = forecasts.global_calibration_note(db)
    assert note and "CALIBRATION RECORD" in note
    assert "0.6->50%" in note and "0.8->60%" in note and "0.9->80%" in note
    assert "Brier" in note


def test_mint_prompts_ask_for_a_resolution_date():
    from app.core import dossiers, storylines
    assert "resolves YYYY-MM-DD" in dossiers._UPDATE_PROMPT
    assert "resolves YYYY-MM-DD" in dossiers._WORLD_PROMPT
    assert "resolves YYYY-MM-DD" in storylines._UPDATE_PROMPT
    assert "only a realized outcome" in storylines._UPDATE_PROMPT
