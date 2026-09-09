"""A research answer the evidence does not support is not knowledge (2026-09-07).

Live, 09:02:57 UTC: the curiosity answer on personalized mRNA cancer
immunotherapy scored p=0.016-0.040 on EVERY claim against the evidence its own
web searches had retrieved ("Nature Reviews Drug Discovery, September 2026",
"EGFR G12C/G12D" - G12C is a KRAS mutation). The chat entailment gate would
have stripped all five sentences; its over-strip guard kept the original
instead, which is the right call for a person reading a chat answer and the
wrong one for an unattended loop. The closure judge, which only ever sees the
text, accepted it. The answer was banked as a resolution, offered to the lesson
extractor, and sent to the owner's channels as "I looked into...".

In the persisted log the guard fired 70 times behind Curiosity Research.

The gate now reports what it did alongside the text it returns, think() carries
that verdict on the DONE event, and the curiosity path treats "the guard kept a
majority-unsupported answer" as a closure failure: requeued, never resolved,
never sent. The chat behaviour is unchanged - the guard still keeps the text.
"""
from __future__ import annotations

import pytest

import app.core.brain as brain
import app.monitors.heartbeat_loop as hb
from app.config import config
from app.core.curiosity import CuriosityQueue

# ---------------------------------------------------------------------------
# the gate reports its verdict
# ---------------------------------------------------------------------------

EVIDENCE = [{"tool": "web_search", "output": (
    "The Bank of Japan held its policy rate at 0.75 percent on September 4, 2026, "
    "citing weak consumption. Governor Ueda said further increases depend on wage data. "
    "Ten-year JGB yields rose to 1.9 percent after the decision, the highest since 2008, "
    "and the yen weakened past 150 to the dollar in afternoon trading in Tokyo. "
    "Analysts at Nomura expect one more increase before the end of the fiscal year."
)}]

SUPPORTED = ("The Bank of Japan held its policy rate at 0.75 percent on September 4, 2026. "
             "Ten-year JGB yields rose to 1.9 percent after the decision. "
             "The yen weakened past 150 to the dollar in afternoon trading.")

UNSUPPORTED = ("The Bank of Japan raised its policy rate to 1.25 percent on September 4, 2026. "
               "Ten-year JGB yields fell to 0.4 percent after the surprise decision. "
               "The yen strengthened to 120 to the dollar in afternoon trading.")


class _Resp:
    def __init__(self, results):
        self._r = results

    def raise_for_status(self):
        return None

    def json(self):
        return {"results": self._r}


def _client(verdict):
    """A sidecar that says `verdict` for every pair, or raises when verdict is None."""
    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            if verdict is None:
                raise ConnectionError("sidecar down")
            return _Resp([{"supported": verdict, "prob": 0.95 if verdict else 0.02}
                          for _ in json["pairs"]])
    return _C


@pytest.fixture
def minicheck_on():
    config.update(ENABLE_MINICHECK=True)
    yield
    config.update(ENABLE_MINICHECK=False)


@pytest.mark.asyncio
async def test_the_guard_keeps_the_text_and_says_so(minicheck_on, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client(False))
    text, verdict = await brain._entail_gate_chat_verdict(UNSUPPORTED, EVIDENCE)
    assert text == UNSUPPORTED, "chat behaviour is unchanged: the guard keeps the original"
    assert verdict["guard"] is True
    assert verdict["checked"] == 3 and verdict["unsupported"] == 3
    assert verdict["inert"] is False


@pytest.mark.asyncio
async def test_a_supported_answer_is_reported_clean(minicheck_on, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client(True))
    text, verdict = await brain._entail_gate_chat_verdict(SUPPORTED, EVIDENCE)
    assert text == SUPPORTED
    assert verdict["guard"] is False and verdict["unsupported"] == 0
    assert verdict["checked"] == 3


@pytest.mark.asyncio
async def test_a_dead_sidecar_is_inert_not_a_verdict(minicheck_on, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client(None))
    text, verdict = await brain._entail_gate_chat_verdict(UNSUPPORTED, EVIDENCE)
    assert text == UNSUPPORTED
    assert verdict["inert"] is True and verdict["guard"] is False


@pytest.mark.asyncio
async def test_the_string_form_still_returns_only_text(minicheck_on, monkeypatch):
    """Every existing caller reads a string; that contract stays."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client(False))
    assert await brain._entail_gate_chat(UNSUPPORTED, EVIDENCE) == UNSUPPORTED


# ---------------------------------------------------------------------------
# the curiosity loop acts on it
# ---------------------------------------------------------------------------
# These seed source='agent_failure': the think() path. Since 2026-09-09 dossier
# questions go through the evidence-first chain instead
# (tests/test_curiosity_evidence_first.py); the guard/inert verdicts tested
# here come from think()'s chat gate and apply to the sources that still use it.

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
    yield d
    d.close()


class _Svc:
    def __init__(self, queue):
        self.curiosity = queue
        self.kg = None
        self.learning = None


def _loop(monkeypatch, queue, answer, meta):
    lp = object.__new__(hb.HeartbeatLoop)
    judged: list[str] = []
    sent: list[str] = []

    async def _meta(_self, query):
        return answer, dict(meta)

    async def _text(_self, query):
        return answer

    monkeypatch.setattr(hb.HeartbeatLoop, "_think_query_meta", _meta, raising=False)
    monkeypatch.setattr(hb.HeartbeatLoop, "_think_query", _text)

    async def _closure(_self, topic, result):
        judged.append(topic)
        return True    # the judge, reading only the text, would have accepted it

    monkeypatch.setattr(hb.HeartbeatLoop, "_curiosity_closure_check", _closure)

    async def _follow(_self, topic, findings):
        sent.append(topic)

    monkeypatch.setattr(hb.HeartbeatLoop, "_send_curiosity_followup", _follow)
    monkeypatch.setattr(hb, "get_services", lambda: _Svc(queue), raising=False)
    monkeypatch.setattr(brain, "get_services", lambda: _Svc(queue))
    import app.tools.native_search as ns
    monkeypatch.setattr(ns, "search_health", lambda: 1.0)
    return lp, judged, sent


TOPIC = "Science: What is the current response rate of personalized mRNA vaccines in NSCLC"


@pytest.mark.asyncio
async def test_a_guarded_answer_is_requeued_not_resolved(monkeypatch, db):
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'agent_failure', 0.6, 'pending', 0)", (TOPIC,))
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue, UNSUPPORTED,
                             {"guard": True, "checked": 5, "unsupported": 5, "inert": False})

    out = await lp._research_one_curiosity(_Svc(queue))

    assert out.startswith("CURIOSITY UNRESOLVED"), out
    assert "unsupported_by_evidence" in out
    row = db.fetchone("SELECT status, attempts, resolution FROM curiosity_queue")
    assert row["status"] == "pending" and row["attempts"] == 1
    assert not row["resolution"]
    assert judged == [], "the judge cannot see evidence; it must not get the final say here"
    assert sent == [], "an unsupported answer must never reach the owner as a follow-up"


@pytest.mark.asyncio
async def test_a_clean_verdict_still_resolves_through_the_judge(monkeypatch, db):
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'agent_failure', 0.6, 'pending', 0)", (TOPIC,))
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue, SUPPORTED,
                             {"guard": False, "checked": 3, "unsupported": 0, "inert": False})

    out = await lp._research_one_curiosity(_Svc(queue))

    assert out.startswith("CURIOSITY RESOLVED"), out
    assert judged == [TOPIC]
    assert sent == [TOPIC]
    assert db.fetchone("SELECT status FROM curiosity_queue")["status"] == "resolved"


@pytest.mark.asyncio
async def test_no_verdict_at_all_keeps_the_old_path(monkeypatch, db):
    """Gate off, no tools, or sidecar inert: nothing is known, so nothing changes."""
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'agent_failure', 0.6, 'pending', 0)", (TOPIC,))
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue, SUPPORTED, {})

    out = await lp._research_one_curiosity(_Svc(queue))

    assert out.startswith("CURIOSITY RESOLVED"), out
    assert judged == [TOPIC]


@pytest.mark.asyncio
async def test_an_inert_gate_defers_rather_than_banks(monkeypatch, db):
    """2026-09-08 22:51 UTC, first curiosity run after an outage: the entailment
    sidecar was still loading, the gate logged "every chunk failed — gate inert",
    and the same Apple question that had scored 3 of 3 claims unsupported at
    17:08 was banked as a provisional resolution. A verifier that could not
    check has not cleared the answer: defer without burning the attempt, the way
    a judge that could not answer does."""
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'agent_failure', 0.6, 'pending', 0)", (TOPIC,))
    queue = CuriosityQueue(db)
    lp, judged, sent = _loop(monkeypatch, queue, SUPPORTED,
                             {"guard": False, "checked": 3, "unsupported": 0, "inert": True})

    out = await lp._research_one_curiosity(_Svc(queue))

    assert out.startswith("CURIOSITY DEFERRED"), out
    assert "grounding_unavailable" in out
    row = db.fetchone("SELECT status, attempts, resolution FROM curiosity_queue")
    assert row["status"] == "pending" and row["attempts"] == 0, dict(row)
    assert not row["resolution"]
    assert judged == [] and sent == []
