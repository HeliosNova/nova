"""C2 (plan 2026-10-09): answerability triage before curiosity research.

Measured 2026-10-09: 3 of 5 long-memory probes failed because the loop
web-searched fictional entities, and the backlog's worst items burn three
research passes each before failing. Two yes/no probes on the resident model
("settleable from public sources now?", "would a search surface a dated
source?") are recorded on the item first; a no/no item is deferred with reason
not_answerable and its attempt is not burned; the pick prefers yes/yes.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.curiosity import CuriosityQueue
from app.monitors.heartbeat_loop import HeartbeatLoop


@pytest.fixture
def q(db):
    return CuriosityQueue(db)


def test_probe_columns_exist_and_record(q):
    cols = {r[1] for r in q._db.fetchall("PRAGMA table_info(curiosity_queue)")}
    assert {"answerable", "evidence_ok", "probe_at"} <= cols
    iid = q.add("What did the ECB decide at its October 2026 meeting?", source="dossier_open_question", urgency=0.6)
    q.record_probe(iid, answerable=True, evidence_ok=False)
    row = q._db.fetchone("SELECT answerable, evidence_ok, probe_at FROM curiosity_queue WHERE id = ?", (iid,))
    assert (row["answerable"], row["evidence_ok"]) == (1, 0) and row["probe_at"]


def test_defer_keeps_the_attempt_and_leaves_the_queue(q):
    iid = q.add("Who leads the Aetherion Guild?", source="dossier_open_question", urgency=0.6)
    q.defer(iid, "not_answerable")
    row = q._db.fetchone("SELECT status, attempts, resolution FROM curiosity_queue WHERE id = ?", (iid,))
    assert row["status"] == "deferred" and row["attempts"] == 0
    assert "not_answerable" in row["resolution"]
    assert q.get_next() is None


def test_pick_prefers_probed_yes_yes_over_higher_urgency(q):
    hi = q.add("Unprobed but urgent question about markets", source="dossier_open_question", urgency=0.7)
    lo = q.add("Probed yes/yes question about the Fed decision", source="dossier_open_question", urgency=0.55)
    q.record_probe(lo, answerable=True, evidence_ok=True)
    assert q.get_next().id == lo
    q.record_probe(hi, answerable=True, evidence_ok=True)
    assert q.get_next().id == hi   # both yes/yes: urgency decides again


def _loop():
    loop = HeartbeatLoop.__new__(HeartbeatLoop)
    return loop


@pytest.mark.asyncio
async def test_no_no_is_deferred_without_an_attempt(q):
    iid = q.add("Who leads the Aetherion Guild?", source="dossier_open_question", urgency=0.6)
    item = q.get_next()
    svc = MagicMock(); svc.curiosity = q
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=["No.", "no"])):
        verdict = await _loop()._triage_curiosity(svc, item)
    assert verdict and verdict.startswith("CURIOSITY DEFERRED") and "not_answerable" in verdict
    row = q._db.fetchone("SELECT status, attempts, answerable, evidence_ok FROM curiosity_queue WHERE id = ?", (iid,))
    assert row["status"] == "deferred" and row["attempts"] == 0
    assert (row["answerable"], row["evidence_ok"]) == (0, 0)


@pytest.mark.asyncio
async def test_yes_yes_proceeds_and_is_recorded_once(q):
    iid = q.add("What did the ECB decide in October 2026?", source="dossier_open_question", urgency=0.6)
    item = q.get_next()
    svc = MagicMock(); svc.curiosity = q
    calls = AsyncMock(side_effect=["Yes", "yes."])
    with patch("app.core.llm.invoke_nothink", calls):
        assert await _loop()._triage_curiosity(svc, item) is None
    row = q._db.fetchone("SELECT answerable, evidence_ok FROM curiosity_queue WHERE id = ?", (iid,))
    assert (row["answerable"], row["evidence_ok"]) == (1, 1)
    # already probed: no second round of calls
    item = q.get_next()
    with patch("app.core.llm.invoke_nothink", calls):
        assert await _loop()._triage_curiosity(svc, item) is None
    assert calls.await_count == 2


@pytest.mark.asyncio
async def test_probe_failure_never_blocks_research(q):
    q.add("What did the ECB decide in October 2026?", source="dossier_open_question", urgency=0.6)
    item = q.get_next()
    svc = MagicMock(); svc.curiosity = q
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=RuntimeError("busy"))):
        assert await _loop()._triage_curiosity(svc, item) is None


def test_triage_runs_before_the_path_split():
    src = inspect.getsource(HeartbeatLoop._research_one_curiosity)
    assert src.index("_triage_curiosity(") < src.index("_EVIDENCE_FIRST_SOURCES")
