"""Half of curiosity's failures had no recorded reason (2026-09-06).

The closure check is two stages: a cheap heuristic (length, then deflection
phrases) and an LLM judge. The judge logs why it said no. Stage 1 logged
nothing at all — and of 340 closure failures since 2026-08-20, 169 came from
the judge and the other 171 came from stage 1. **Half the failures of the loop
that answers Nova's own questions were undiagnosable.**

That is why this instruments and changes nothing else. Several markers are
phrases a good answer legitimately contains while hedging one sub-part
("unclear from", "uncertain about"), so a research answer that settles the
question and admits one gap is killed before the judge ever reads it — but
whether that is actually happening is a measurement, not an argument, and the
measurement did not exist. Now it does.

These tests pin the RECORD, not the verdict. The verdicts are unchanged.
"""
from __future__ import annotations

import logging

import pytest

import app.monitors.heartbeat_loop as hb

LOGGER = "app.monitors.heartbeat_loop"
GOOD = ("The Federal Open Market Committee has 12 voting members in 2026, of whom "
        "seven are governors and five are reserve bank presidents rotating annually. "
        "The current split is six hawkish to six dovish after the August meeting.")


def _loop():
    return object.__new__(hb.HeartbeatLoop)


@pytest.mark.asyncio
async def test_a_too_short_answer_says_so(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ok = await _loop()._curiosity_closure_check("Some topic", "too short")
    assert ok is False
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "stage-1 reject" in msg and "too short" in msg
    assert "Some topic" in msg, "the topic must be named or the line is unusable"


@pytest.mark.asyncio
async def test_a_deflection_names_the_phrase_that_killed_it(caplog):
    """Which marker fired is the whole point — one of them may be over-broad."""
    answer = ("The committee met in August. The exact rotation for next year is "
              "unclear from the sources available at this time, though the current "
              "split is six to six and the vote was unanimous on the rate hold.")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ok = await _loop()._curiosity_closure_check("FOMC split", answer)
    assert ok is False
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "unclear from" in msg, "the matched marker must be named"
    assert "stage-1 reject" in msg


@pytest.mark.asyncio
async def test_the_answer_length_is_recorded_with_the_rejection():
    """A deflection at 200 chars and one at 3,000 are different problems."""
    import inspect
    src = inspect.getsource(hb.HeartbeatLoop._curiosity_closure_check)
    assert "len(result)" in src, "the rejection must carry how long the answer was"


@pytest.mark.asyncio
async def test_a_clean_answer_still_reaches_the_judge(monkeypatch, caplog):
    """Instrumentation must not have changed who gets through stage 1."""
    seen = {}

    async def _judge(_msgs, **kw):
        seen["asked"] = True
        return '{"answers": true, "reason": "concrete"}'

    from app.core import llm as llm_mod
    monkeypatch.setattr(llm_mod, "invoke_nothink", _judge)
    monkeypatch.setattr(llm_mod, "extract_json_object",
                        lambda _raw: {"answers": True, "reason": "concrete"})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ok = await _loop()._curiosity_closure_check("FOMC split", GOOD)
    assert ok is True
    assert seen.get("asked"), "a clean answer must reach the judge"
    assert "stage-1 reject" not in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_judge_failure_is_not_a_verdict(monkeypatch):
    """This asserted `is True` until 2026-09-07, and the change is the point.

    "A broken judge must not requeue forever" was the reason for defaulting
    to resolve. MAX_ATTEMPTS already bounds the loop, and the default had a
    cost the reason never weighed: on 2026-09-07 the judge timed out behind a
    saturated GPU nine times in one day and each time banked whatever text it
    had been handed - "The model is busy right now" - as a resolution, then
    sent it to the owner. None means "could not judge"; the caller defers the
    item without burning its attempt (tests/test_curiosity_busy_model_is_not_an_answer.py).
    """
    from app.core import llm as llm_mod

    async def _boom(*_a, **_k):
        raise RuntimeError("model down")

    monkeypatch.setattr(llm_mod, "invoke_nothink", _boom)
    assert await _loop()._curiosity_closure_check("FOMC split", GOOD) is None
