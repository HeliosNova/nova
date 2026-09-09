"""A run asks three DIFFERENT questions, not the same one three times (2026-09-07).

Live, 09:01-09:03 UTC on the first tick after a restart: item #308 ("Can the
personalized mRNA cancer immunotherapy mechanism be scaled...") was picked,
failed its closure check, picked AGAIN, failed AGAIN, and picked a THIRD time
in the same run - attempts 0 -> 1 -> 2 in three minutes - and the third answer
was accepted by a judge that had rejected two near-identical ones minutes
earlier. `get_next` orders by effective urgency with no term for "just tried",
so after `fail()` the same row is still the top of the queue, and with
`_CURIOSITY_BATCH == MAX_ATTEMPTS == 3` one hard question consumes the whole
hourly run and is either burned out or pushed through a noisy judge.

The hourly tally in the persisted log reads "3 closure check failed" almost
every run, which is what this looks like from the outside: one question per
hour, not three.
"""
from __future__ import annotations

import pytest

import app.monitors.heartbeat_loop as hb
from app.core.curiosity import CuriosityQueue


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


def _add(db, topic, urgency=0.6, days_old=1):
    db.execute(
        "INSERT INTO curiosity_queue (topic, source, urgency, status, attempts, created_at) "
        "VALUES (?, 'agent_failure', ?, 'pending', 0, datetime('now', ?))",
        (topic, urgency, f"-{days_old} days"))


# --- the queue -------------------------------------------------------------

def test_a_failed_attempt_alone_does_not_move_a_question_down(db):
    """Documenting the existing behaviour the loop has to work around: the
    pick order is urgency and age, and a failure changes neither."""
    _add(db, "hard question", days_old=3)
    _add(db, "easy question", days_old=2)
    q = CuriosityQueue(db)
    first = q.get_next()
    assert first.topic == "hard question"
    q.fail(first.id)
    assert q.get_next().topic == "hard question"


def test_get_next_can_skip_the_ids_already_tried_this_run(db):
    _add(db, "hard question", days_old=3)
    _add(db, "easy question", days_old=2)
    q = CuriosityQueue(db)
    first = q.get_next()
    assert q.get_next(exclude_ids={first.id}).topic == "easy question"


def test_get_next_with_everything_excluded_is_empty(db):
    _add(db, "only question")
    q = CuriosityQueue(db)
    only = q.get_next()
    assert q.get_next(exclude_ids={only.id}) is None


# --- the loop, with the real queue behind it ----------------------------------

class _Svc:
    def __init__(self, queue):
        self.curiosity = queue
        self.kg = None
        self.learning = None


def _loop(monkeypatch, queue, answer="A plain answer with nothing to verify.", meta=None,
          closure=False):
    lp = object.__new__(hb.HeartbeatLoop)
    asked: list[str] = []

    async def _meta(_self, query):
        asked.append(query)
        return answer, dict(meta or {})

    async def _text(_self, query):
        asked.append(query)
        return answer

    monkeypatch.setattr(hb.HeartbeatLoop, "_think_query_meta", _meta, raising=False)
    monkeypatch.setattr(hb.HeartbeatLoop, "_think_query", _text)

    async def _closure(_self, topic, result, **kw):
        return closure

    monkeypatch.setattr(hb.HeartbeatLoop, "_curiosity_closure_check", _closure)

    async def _follow(_self, topic, findings, **kw):
        return None

    monkeypatch.setattr(hb.HeartbeatLoop, "_send_curiosity_followup", _follow)
    monkeypatch.setattr(hb, "get_services", lambda: _Svc(queue), raising=False)
    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: _Svc(queue))
    import app.tools.native_search as ns
    monkeypatch.setattr(ns, "search_health", lambda: 1.0)
    return lp, asked


def _topics(asked):
    out = []
    for q in asked:
        line = q.split("Research question:", 1)[1].strip().splitlines()[0]
        out.append(line.strip())
    return out


@pytest.mark.asyncio
async def test_a_run_asks_three_different_questions_when_the_first_fails(monkeypatch, db):
    """The measured failure: three picks, one question, attempts 0 -> 1 -> 2."""
    for i, t in enumerate(("What is the current ECB deposit rate",
                           "What is the current price of Brent crude",
                           "Who is the current governor of the Bank of Japan")):
        _add(db, t, days_old=5 - i)
    queue = CuriosityQueue(db)
    lp, asked = _loop(monkeypatch, queue, closure=False)

    out = await lp._execute_curiosity_research({})

    assert len(asked) == 3
    assert len(set(_topics(asked))) == 3, f"the same question was re-asked: {_topics(asked)}"
    rows = db.fetchall("SELECT topic, attempts, status FROM curiosity_queue ORDER BY id")
    assert [r["attempts"] for r in rows] == [1, 1, 1], [dict(r) for r in rows]
    assert all(r["status"] == "pending" for r in rows), "one run must not burn a question out"
    assert "0/3 resolved" in out


@pytest.mark.asyncio
async def test_a_run_stops_when_only_tried_questions_remain(monkeypatch, db):
    """One pending question and a failing answer: ask once, then stop - do not
    spend the other two slots re-asking it."""
    _add(db, "What is the current federal funds rate")
    queue = CuriosityQueue(db)
    lp, asked = _loop(monkeypatch, queue, closure=False)

    await lp._execute_curiosity_research({})

    assert len(asked) == 1, _topics(asked)
    row = db.fetchone("SELECT attempts, status FROM curiosity_queue")
    assert row["attempts"] == 1 and row["status"] == "pending"
