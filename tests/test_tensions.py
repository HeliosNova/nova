"""C1 (plan 2026-10-09): resolve a dossier tension from Nova's own spans,
admitted only when a second judge agrees."""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core import tensions as tn
from app.core.tensions import Tension


def _seed(db, title, dkey, bullets, digests):
    body = "## Current understanding\n\nText.\n\n## Key facts\n\n" + "\n".join(f"* {b}" for b in bullets) + "\n"
    db.execute("INSERT INTO dossiers (kind, dkey, title, body) VALUES ('domain', ?, ?, ?)", (dkey, title, body))
    mid = db.execute("INSERT INTO monitors (name, check_type, check_config, schedule_seconds) VALUES (?, 'query', '{}', 3600)",
                     (f"Domain Study: {title}",)).lastrowid
    for date, text in digests:
        db.execute("INSERT INTO monitor_results (monitor_id, status, value, created_at) VALUES (?, 'ok', ?, ?)",
                   (mid, text + " " * 420, date))


PAD = "Markets stayed calm through the session as traders weighed the data. "


@pytest.fixture
def seeded(db):
    _seed(db, "Finance", "finance",
          ["Headline inflation ran at 3.4% in August (reuters.com)"],
          [("2026-09-02 08:00:00", PAD + "Headline inflation ran at 3.4% in August, the agency said. " + PAD)])
    _seed(db, "Economics", "economics",
          ["Headline inflation eased to 2.9% in September (bbc.com)"],
          [("2026-10-05 08:00:00", PAD + "Headline inflation eased to 2.9% in September, revised data show. " + PAD)])
    return db


def _tension():
    return Tension(title_a="Finance", dkey_a="finance", val_a=3.4, head_a="Headline inflation ran at 3.4% in August",
                   title_b="Economics", dkey_b="economics", val_b=2.9, head_b="Headline inflation eased to 2.9% in September",
                   unit="%", text="Finance says 3.4% but Economics says 2.9%")


def test_locate_spans_finds_percent_and_word_unit_figures():
    digests = [("2026-10-01", "The 10-year yield closed at 5.501% on Monday. Volume was 2 billion shares. Nothing else.")]
    assert tn.locate_spans(digests, 5.501, "%")[0]["sentence"].startswith("The 10-year yield closed at 5.501%")
    assert "2 billion shares" in tn.locate_spans(digests, 2, "billion")[0]["sentence"]
    assert tn.locate_spans(digests, 7, "%") == []


@pytest.mark.asyncio
async def test_different_quantities_resolve_both_without_rewrite(seeded):
    judge = json.dumps({"verdict": "both", "reasoning": "10-year vs 30-year yields", "winner_date": None})
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=[judge, "Yes — different instruments."])):
        res = await tn.resolve_tension(seeded, _tension(), judge_model="j", verifier_model="v")
    assert res["verdict"] == "both" and res["admitted"] is True
    assert tn.apply_resolution(seeded, _tension(), res) is False
    row = seeded.fetchone("SELECT verdict, verifier_agrees, judge_model FROM tension_resolutions")
    assert (row["verdict"], row["verifier_agrees"], row["judge_model"]) == ("both", 1, "j")


@pytest.mark.asyncio
async def test_same_quantity_resolves_to_the_later_span_and_rewrites_the_loser(seeded):
    judge = json.dumps({"verdict": "b", "reasoning": "September revision is newer", "winner_date": "2026-10-05"})
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=[judge, "yes, the later digest"])):
        res = await tn.resolve_tension(seeded, _tension(), judge_model="j", verifier_model="v")
    assert res["verdict"] == "b" and res["admitted"] is True
    assert "3.4%" in res["figure_a"] and "2.9%" in res["figure_b"]
    assert tn.apply_resolution(seeded, _tension(), res) is True
    body = seeded.fetchone("SELECT body FROM dossiers WHERE dkey = 'finance'")["body"]
    assert "superseded: Economics records 2.9%" in body and "as of 2026-10-05" in body
    untouched = seeded.fetchone("SELECT body FROM dossiers WHERE dkey = 'economics'")["body"]
    assert "superseded" not in untouched
    rev = seeded.fetchone("SELECT revised, dkey FROM belief_revisions")
    assert rev["dkey"] == "finance" and "superseded by Economics" in rev["revised"]
    # idempotent: a second application does not stack notes
    assert tn.apply_resolution(seeded, _tension(), res) is False


@pytest.mark.asyncio
async def test_verifier_disagreement_leaves_the_tension_open(seeded):
    judge = json.dumps({"verdict": "a", "reasoning": "August is the headline", "winner_date": "2026-09-02"})
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=[judge, "No. The later digest says revised."])):
        res = await tn.resolve_tension(seeded, _tension(), judge_model="j", verifier_model="v")
    assert res["verdict"] == "a" and res["verifier_agrees"] is False and res["admitted"] is False
    assert tn.apply_resolution(seeded, _tension(), res) is False
    row = seeded.fetchone("SELECT verifier_agrees, reasoning FROM tension_resolutions")
    assert row["verifier_agrees"] == 0 and "verifier:" in row["reasoning"]


@pytest.mark.asyncio
async def test_judge_failure_is_stored_as_unresolvable(seeded):
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=RuntimeError("busy"))):
        res = await tn.resolve_tension(seeded, _tension(), judge_model="j", verifier_model="v")
    assert res["verdict"] == "unresolvable" and res["admitted"] is False


def test_consolidation_reports_resolved_or_open_tensions():
    from app.core import dossiers
    src = inspect.getsource(dossiers.consolidate_dossiers)
    assert "Tension resolved:" in src and "Tension open:" in src
    assert "resolve_tension(" in src and "apply_resolution" in src


def test_structured_tensions_feed_the_formatter(db):
    from app.core.dossiers import _numeric_tension_records, _numeric_tensions
    _seed(db, "Finance", "finance", ["Headline inflation ran at 3.4% in August (reuters.com)"], [])
    _seed(db, "Economics", "economics", ["Headline inflation eased to 2.9% in September (bbc.com)"], [])
    recs = _numeric_tension_records(db)
    assert len(recs) == 1 and {recs[0]["title_a"], recs[0]["title_b"]} == {"Finance", "Economics"}
    assert recs[0]["unit"] == "%" and {recs[0]["val_a"], recs[0]["val_b"]} == {3.4, 2.9}
    assert recs[0]["dkey_a"] in ("finance", "economics")
    assert _numeric_tensions(db) == [recs[0]["text"]]


@pytest.mark.asyncio
async def test_no_spans_on_either_side_is_unresolvable_without_a_judge_call(db):
    _seed(db, "Finance", "finance", ["Headline inflation ran at 3.4% in August (reuters.com)"], [])
    _seed(db, "Economics", "economics", ["Headline inflation eased to 2.9% in September (bbc.com)"], [])
    judge = AsyncMock(side_effect=RuntimeError("must not be called"))
    with patch("app.core.llm.invoke_nothink", judge):
        res = await tn.resolve_tension(db, _tension(), judge_model="j", verifier_model="v")
    assert res["verdict"] == "unresolvable" and res["admitted"] is False
    assert judge.await_count == 0
    assert "no stored digest sentence" in db.fetchone("SELECT reasoning FROM tension_resolutions")["reasoning"]

