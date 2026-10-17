"""Curiosity attempt accounting (2026-10-17).

Measured 10-09..10-14: 39 Curiosity Research runs, 79 outcome lines, 9
resolved. 40 of the failures were closure checks on 18 questions, three passes
each, where the judge's objection was that the public record lacks the datum
("does not provide the specific number", "lacks the necessary data"); six
pending items asked for what nobody publishes (patch status, yield rate,
internal allocation). Two changes: an unpublished-shaped item is parked at pick
time before any probe, and an evidence-first answer the judge rejects for a
missing datum parks the question with the answer attached instead of searching
the same web twice more.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.curiosity import CuriosityQueue
import app.monitors.heartbeat_loop as hb


def _loop():
    return object.__new__(hb.HeartbeatLoop)


@pytest.fixture
def q(db):
    return CuriosityQueue(db)


# --- c1: unpublished-shaped backlog items ----------------------------------------
@pytest.mark.asyncio
async def test_an_unpublished_shaped_question_is_parked_before_any_probe(q):
    iid = q.add("Startups and VC: What is the exact internal allocation split between early-stage "
                "and growth equity within the fund?", source="dossier_open_question", urgency=0.6)
    item = q.get_next()
    svc = MagicMock(); svc.curiosity = q
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=AssertionError("no probe may run"))):
        verdict = await _loop()._triage_curiosity(svc, item)
    assert verdict and "reason=unpublished" in verdict
    row = q._db.fetchone("SELECT status, attempts, resolution FROM curiosity_queue WHERE id = ?", (iid,))
    assert row["status"] == "deferred" and row["attempts"] == 0 and "unpublished" in row["resolution"]
    assert q.get_next() is None


@pytest.mark.asyncio
async def test_a_public_process_question_proceeds_to_the_probes(q):
    iid = q.add("Semiconductors: What is the current status of the Remote Access Security Act in the "
                "U.S. Senate?", source="dossier_open_question", urgency=0.6)
    item = q.get_next()
    svc = MagicMock(); svc.curiosity = q
    with patch("app.core.llm.invoke_nothink", AsyncMock(side_effect=["yes", "yes"])):
        assert await _loop()._triage_curiosity(svc, item) is None
    row = q._db.fetchone("SELECT status, answerable FROM curiosity_queue WHERE id = ?", (iid,))
    assert row["status"] == "pending" and row["answerable"] == 1


# --- c2: what the verdict was about ---------------------------------------------
@pytest.mark.parametrize("reason", [
    "The answer explicitly states that it does not provide the specific number of new grant awards issued since the drop.",
    "The response explicitly states that it lacks the necessary data to provide an answer.",
    "The answer explicitly states that it does not contain information regarding the specific $1 billion reallocation.",
    "No specific figures are given for the current transit volume.",
])
def test_the_judge_reasons_seen_live_read_as_unsettled(reason):
    assert hb.HeartbeatLoop._UNSETTLED_REASON_RE.search(reason)


@pytest.mark.parametrize("reason", [
    "The response fails to resolve the contradiction between the two figures.",
    "The answer discusses a different company than the one asked about.",
    "The proposed answer is a list of links with prose about an unrelated event.",
])
def test_other_rejections_are_not_unsettled(reason):
    assert not hb.HeartbeatLoop._UNSETTLED_REASON_RE.search(reason)


GOOD = ("The Federal Open Market Committee has 12 voting members in 2026, of whom seven are "
        "governors and five are reserve bank presidents rotating annually. The current split is "
        "six hawkish to six dovish after the August meeting (reuters.com).")


@pytest.mark.asyncio
async def test_the_closure_check_records_whether_the_record_was_the_problem(monkeypatch):
    from app.core import llm as llm_mod
    loop = _loop()

    async def _judge_no(*_a, **_k):
        return ('{"answers": false, "reason": "The answer explicitly states that it does not '
                'provide the specific number requested."}')
    monkeypatch.setattr(llm_mod, "invoke_nothink", _judge_no)
    assert await loop._curiosity_closure_check("FOMC split", GOOD) is False
    assert loop._last_closure_unsettled is True

    async def _judge_other(*_a, **_k):
        return '{"answers": false, "reason": "The answer is about a different committee."}'
    monkeypatch.setattr(llm_mod, "invoke_nothink", _judge_other)
    assert await loop._curiosity_closure_check("FOMC split", GOOD) is False
    assert loop._last_closure_unsettled is False

    async def _judge_yes(*_a, **_k):
        return '{"answers": true, "reason": "ok"}'
    monkeypatch.setattr(llm_mod, "invoke_nothink", _judge_yes)
    assert await loop._curiosity_closure_check("FOMC split", GOOD) is True
    assert loop._last_closure_unsettled is False


@pytest.mark.asyncio
async def test_a_deflection_counts_as_unsettled():
    loop = _loop()
    answer = ("The evidence does not settle this. No data available on the specific yield rate; "
              "the sources describe capacity plans only (reuters.com). ") * 2
    assert await loop._curiosity_closure_check("TSMC yield", answer) is False
    assert loop._last_closure_unsettled is True


# --- c3: the evidence-first path parks instead of requeueing -----------------------
def test_the_evidence_first_path_parks_an_unsettled_question_before_the_requeue():
    src = inspect.getsource(hb.HeartbeatLoop._research_one_curiosity)
    i_park = src.index('reason=unsettled"')          # the c3 block, not reason=unsettled_no_findings
    i_requeue = src.index("closure check failed — requeued")
    assert i_park < i_requeue
    block = src[:i_park]
    assert 'grounding.get("evidence_first")' in block and "_last_closure_unsettled" in block
    assert "svc.curiosity.defer" in block
