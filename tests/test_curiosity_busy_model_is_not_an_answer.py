"""'The model is busy right now' is not knowledge (2026-09-07).

Nine questions were RESOLVED between 11:47 and 20:48 UTC with the resolution

    The model is busy right now (a background research job is using the
    GPU). Please try again in a moment.

That string is brain.think()'s friendly answer for a timed-out generation
(LLMUnavailableError with 'timeout' in it - the provider is up, the GPU is
saturated by a digest). It arrives as a normal answer with an ERROR event
carrying code 'llm_busy'. The curiosity path's model-down guard knew only the
'I can't reach the language model' prefix, so the busy text went to the
closure judge - which ran on the same saturated model, timed out, and
"defaulted to resolve" - and was banked, offered to the lesson extractor, and
sent to the owner as "I looked into...". Ten follow-ups went out that day; 197
timed-out requests sat behind the post-restart catch-up wave.

Two rules follow. A generation the model never made is not an attempt, so
it burns nothing and resolves nothing - matched by the event code, not the
wording. And a judge that could not answer has not said yes: the item is
deferred, not resolved.
"""
from __future__ import annotations

import pytest

import app.core.brain as brain
import app.monitors.heartbeat_loop as hb
from app.core.curiosity import CuriosityQueue

BUSY = ("The model is busy right now (a background research job is using the GPU). "
        "Please try again in a moment.")
ANSWER = ("The Bank of Japan held its policy rate at 0.75 percent on September 4, 2026. "
          "Ten-year JGB yields rose to 1.9 percent after the decision (reuters.com).")
TOPIC = "Finance: What is the Bank of Japan's current policy rate"


@pytest.fixture
def db(tmp_path):
    from app.database import SafeDB
    d = SafeDB(str(tmp_path / "c.db"))
    d.init_schema()
    d.execute(
        "CREATE TABLE IF NOT EXISTS curiosity_queue ("
        "id INTEGER PRIMARY KEY, topic TEXT, source TEXT, urgency REAL, "
        "status TEXT, attempts INT DEFAULT 0, resolution TEXT, "
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP, resolved_at TEXT)")
    d.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
              "VALUES (?, 'agent_failure', 0.6, 'pending', 0)", (TOPIC,))
    yield d
    d.close()


class _Svc:
    def __init__(self, queue):
        self.curiosity = queue
        self.kg = None
        self.learning = None


def _loop(monkeypatch, queue, answer, meta, judge=True):
    lp = object.__new__(hb.HeartbeatLoop)
    judged: list[str] = []
    sent: list[str] = []

    async def _meta(_self, query):
        return answer, dict(meta)

    monkeypatch.setattr(hb.HeartbeatLoop, "_think_query_meta", _meta, raising=False)

    async def _closure(_self, topic, result):
        judged.append(topic)
        if isinstance(judge, Exception):
            raise judge
        return judge

    monkeypatch.setattr(hb.HeartbeatLoop, "_curiosity_closure_check", _closure)

    async def _follow(_self, topic, findings):
        sent.append(topic)

    monkeypatch.setattr(hb.HeartbeatLoop, "_send_curiosity_followup", _follow)
    monkeypatch.setattr(hb, "get_services", lambda: _Svc(queue), raising=False)
    monkeypatch.setattr(brain, "get_services", lambda: _Svc(queue))
    import app.tools.native_search as ns
    monkeypatch.setattr(ns, "search_health", lambda: 1.0)
    return lp, judged, sent


def _row(db):
    return db.fetchone("SELECT status, attempts, resolution FROM curiosity_queue")


# --- the busy answer -----------------------------------------------------------

@pytest.mark.asyncio
async def test_the_busy_message_is_not_researched_and_burns_nothing(monkeypatch, db):
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue, BUSY, {})

    out = await lp._research_one_curiosity(_Svc(queue))

    assert "LLM unavailable" in out, out
    row = _row(db)
    assert row["status"] == "pending" and row["attempts"] == 0 and not row["resolution"]
    assert judged == [] and sent == []


@pytest.mark.asyncio
async def test_the_event_code_is_what_is_matched_not_the_wording(monkeypatch, db):
    """The friendly text may be reworded; the code is the contract."""
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue,
                             "Sorry, everything is tied up at the moment - try again shortly.",
                             {"code": "llm_busy"})

    out = await lp._research_one_curiosity(_Svc(queue))

    assert "LLM unavailable" in out, out
    row = _row(db)
    assert row["status"] == "pending" and row["attempts"] == 0 and not row["resolution"]
    assert judged == [] and sent == []


# --- the judge that could not answer ---------------------------------------

@pytest.mark.asyncio
async def test_a_judge_that_failed_has_not_said_yes(monkeypatch, db):
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue, ANSWER, {}, judge=None)

    out = await lp._research_one_curiosity(_Svc(queue))

    assert out.startswith("CURIOSITY DEFERRED"), out
    assert "judge" in out
    row = _row(db)
    assert row["status"] == "pending" and row["attempts"] == 0, dict(row)
    assert not row["resolution"]
    assert sent == []


@pytest.mark.asyncio
async def test_the_closure_check_reports_a_failed_judge_as_none(monkeypatch):
    """True/False are verdicts; None is 'could not judge' — a different thing."""
    from app.core import llm as llm_mod

    async def _boom(*a, **k):
        raise TimeoutError("Request timed out after 3 retries")

    monkeypatch.setattr(llm_mod, "invoke_nothink", _boom)
    lp = object.__new__(hb.HeartbeatLoop)
    verdict = await lp._curiosity_closure_check(TOPIC, ANSWER)
    assert verdict is None


# --- the code reaches the caller -----------------------------------------------

@pytest.mark.asyncio
async def test_think_query_meta_carries_the_error_code(monkeypatch):
    from unittest.mock import MagicMock
    from app.schema import EventType, StreamEvent

    async def _fake_think(**kwargs):
        yield StreamEvent(type=EventType.TOKEN, data={"text": BUSY})
        yield StreamEvent(type=EventType.ERROR, data={"message": BUSY, "code": "llm_busy"})

    monkeypatch.setattr(brain, "think", _fake_think)
    loop = hb.HeartbeatLoop.__new__(hb.HeartbeatLoop)
    loop.store = MagicMock()
    loop.store.list_all.return_value = []
    loop.store.get_recent_results.return_value = []
    monkeypatch.setattr(hb, "_recent_conversation_context", lambda *a, **k: "", raising=False)

    text, meta = await loop._think_query_meta("Research question: anything")

    assert text == BUSY
    assert meta.get("code") == "llm_busy"
