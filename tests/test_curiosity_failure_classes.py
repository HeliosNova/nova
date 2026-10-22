"""Curiosity failure classes from the 2026-10-21 diagnostic of the closure
judge's own reasons (the loop resolved 18% at 40 attempted):

1. TRUNCATED questions — minted text cut mid-sentence ("...What is the exact
   breakdown of") can never be answered; the judge rejected two as "truncated
   and incomplete". Guarded at mint: a long question with no terminator is cut.
2. FALSE-PREMISE questions — built on a hallucinated digest/dossier claim
   ("Anthropic has not filed for a $2 trillion IPO", "OpenAI has not submitted
   a Navier-Stokes…"). They can never resolve; the judge names it. Parked
   TERMINALLY instead of burning three research passes.
3. TRUNCATED answers — research_question's 500-token cap cut 3-6 sentence
   answers mid-sentence; raised to 800.
"""

from __future__ import annotations

import inspect
import pytest

from app.core.curiosity import CuriosityQueue
import app.monitors.heartbeat_loop as hb
from app.monitors import deep_research as dr


@pytest.fixture
def q(db):
    return CuriosityQueue(db)


# --- 1. truncated-question guard -------------------------------------------
def test_a_long_question_with_no_terminator_is_refused_at_mint(q):
    cut = ("Finance: What is the exact scope of the Critical Infrastructure Defense Program "
           "contracts as disclosed in its latest regulatory filing and what is the exact breakdown of")
    assert len(cut) > 100 and not cut.endswith("?")
    assert q.add(cut, source="dossier_open_question", urgency=0.6) == -1


def test_a_complete_question_still_mints(q):
    ok = ("Finance: What is the exact scope of the Critical Infrastructure Defense Program "
          "contracts as disclosed in the latest regulatory filing, and who are the counterparties?")
    assert q.add(ok, source="dossier_open_question", urgency=0.6) > 0


def test_short_topics_are_not_subject_to_the_terminator_guard(q):
    # The guard targets long cut questions; a short topic without punctuation is fine.
    assert q.add("Fed rate decision October", source="dossier_open_question", urgency=0.6) > 0


# --- 2. false-premise terminal park -----------------------------------------
@pytest.mark.parametrize("reason", [
    "The premise of the question is factually incorrect; OpenAI has not submitted a Navier-Stokes proof.",
    "The premise of the question is factually incorrect; Anthropic has not filed for a $2 trillion IPO.",
    "The premise of the question is factually incorrect; Iran does not have 'control' over the strait.",
    "The question's premise is false: no such filing exists.",
])
def test_judge_reasons_seen_live_read_as_false_premise(reason):
    assert hb.HeartbeatLoop._FALSE_PREMISE_RE.search(reason)


@pytest.mark.parametrize("reason", [
    "The answer explicitly states that it does not provide the specific number requested.",
    "The response fails to provide a specific duration or quantitative estimate.",
    "The answer describes the release strategy but fails to explicitly state the date.",
])
def test_missing_datum_reasons_are_NOT_false_premise(reason):
    assert not hb.HeartbeatLoop._FALSE_PREMISE_RE.search(reason)


GOOD = ("The Federal Open Market Committee has 12 voting members in 2026, seven governors and "
        "five reserve bank presidents rotating annually (reuters.com).")


@pytest.mark.asyncio
async def test_the_closure_check_flags_a_false_premise(monkeypatch):
    from app.core import llm as llm_mod
    loop = object.__new__(hb.HeartbeatLoop)

    async def _judge_premise(*_a, **_k):
        return ('{"answers": false, "reason": "The premise of the question is factually incorrect; '
                'Anthropic has not filed for a $2 trillion IPO."}')
    monkeypatch.setattr(llm_mod, "invoke_nothink", _judge_premise)
    assert await loop._curiosity_closure_check("Anthropic IPO timeline", GOOD) is False
    assert loop._last_closure_false_premise is True
    assert loop._last_closure_unsettled is False      # it is NOT a missing-datum miss

    async def _judge_datum(*_a, **_k):
        return '{"answers": false, "reason": "The answer does not provide the specific figure."}'
    monkeypatch.setattr(llm_mod, "invoke_nothink", _judge_datum)
    assert await loop._curiosity_closure_check("some figure", GOOD) is False
    assert loop._last_closure_false_premise is False
    assert loop._last_closure_unsettled is True


def test_false_premise_parks_before_unsettled_and_before_the_requeue():
    src = inspect.getsource(hb.HeartbeatLoop._research_one_curiosity)
    i_fp = src.index("reason=false_premise")
    i_uns = src.index('reason=unsettled"')
    i_req = src.index("closure check failed — requeued")
    assert i_fp < i_uns < i_req
    assert "svc.curiosity.defer" in src[:i_fp]


# --- 3. answer token cap -----------------------------------------------------
def test_research_answer_cap_is_800():
    src = inspect.getsource(dr.research_question)
    assert "max_tokens=800" in src and "max_tokens=500" not in src
