"""Curiosity researches a dossier question the way a digest is written (2026-09-09).

After every integrity fix of 2026-09-07/08 the curiosity yield was still zero:
the 02:05 UTC run came back 6/6, 7/8 and 4/4 claims unsupported by the
answer's own search evidence. The gate was right. The research path was
chat-style think() - the 9B reading search SNIPPETS and writing from memory
past them - while the Domain Studies that the questions come from are written
by the evidence-first chain: gather and READ articles, extract findings on the
synthesis model, synthesize with citations, entail-gate against the articles.

`research_question` is that chain for one question. Dossier questions go
through it; the other sources (agent/reflexion/quiz failures, which may be
about the owner's own work) keep the memory-first think() path. The closure
judge, the provisional rescue and the ledger hooks are unchanged: they read
the text this returns.
"""
from __future__ import annotations

import pytest

import app.core.brain as brain
import app.monitors.heartbeat_loop as hb
from app.config import config as _cfg
from app.core.curiosity import CuriosityQueue
from app.monitors import deep_research as dr

QUESTION = "Science: What is the current response rate of personalized mRNA vaccines in NSCLC?"
ARTS = [
    ("Trial readout", "https://reuters.com/a", "On September 4, 2026 the phase 2 readout showed a 31 percent objective response rate. " * 8),
    ("Analysis", "https://statnews.com/b", "Analysts said the 31 percent rate compares with 20 percent for checkpoint inhibitors alone. " * 8),
    ("Company note", "https://fiercebiotech.com/c", "The company plans a phase 3 start in early 2027. " * 8),
]
FINDINGS = [
    ("Trial readout", "https://reuters.com/a", "The phase 2 readout on September 4, 2026 showed a 31 percent objective response rate."),
    ("Analysis", "https://statnews.com/b", "The 31 percent rate compares with 20 percent for checkpoint inhibitors alone."),
]
ANSWER = ("The phase 2 readout on September 4, 2026 showed a 31 percent objective response rate "
          "(reuters.com). That compares with 20 percent for checkpoint inhibitors alone (statnews.com).")


@pytest.fixture
def syn_model():
    old = getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "")
    _cfg.update(MONITOR_SYNTHESIS_MODEL="qwen3.8:27b")
    yield "qwen3.8:27b"
    _cfg.update(MONITOR_SYNTHESIS_MODEL=old)


def _stub_chain(monkeypatch, arts=ARTS, findings=FINDINGS, answer=ANSWER):
    calls = {"gather": [], "findings": [], "invoke": [], "gate": []}

    async def _gather(subject, **kw):
        calls["gather"].append((subject, kw))
        return list(arts)

    async def _findings(articles, subject, **kw):
        calls["findings"].append((subject, kw))
        return list(findings)

    async def _invoke(messages, **kw):
        calls["invoke"].append((messages, kw))
        return answer

    async def _gate(text, articles, **kw):
        calls["gate"].append((text, articles, kw))
        return text, 1

    monkeypatch.setattr(dr, "_gather_sources", _gather)
    monkeypatch.setattr(dr, "_findings", _findings)
    monkeypatch.setattr(dr, "_invoke_bg", _invoke)
    monkeypatch.setattr(dr, "_entailment_gate", _gate)
    return calls


# --- the chain for one question -------------------------------------------

@pytest.mark.asyncio
async def test_a_question_is_read_extracted_synthesized_and_gated(monkeypatch, syn_model):
    calls = _stub_chain(monkeypatch)
    text, stats = await dr.research_question(QUESTION)
    assert text == ANSWER
    assert calls["gather"] and calls["gather"][0][0] == QUESTION
    assert calls["findings"] and calls["findings"][0][1].get("model") == syn_model
    assert len(calls["invoke"]) == 1, "one synthesis call"
    assert calls["invoke"][0][1].get("model") == syn_model
    prompt = calls["invoke"][0][0][0]["content"]
    assert QUESTION in prompt and "reuters.com" in prompt, "the synthesis sees the question and the evidence"
    assert calls["gate"] and calls["gate"][0][1] == ARTS, "gated against the articles it read"
    assert stats["sources"] == 3 and stats["findings"] == 2 and stats["entail_dropped"] == 1
    assert stats["evidence_first"] is True


@pytest.mark.asyncio
async def test_too_few_sources_means_no_answer_and_no_model_call(monkeypatch, syn_model):
    calls = _stub_chain(monkeypatch, arts=ARTS[:1])
    text, stats = await dr.research_question(QUESTION)
    assert text == "" and stats["reason"] == "no_sources"
    assert not calls["invoke"] and not calls["findings"]


@pytest.mark.asyncio
async def test_no_findings_means_no_answer(monkeypatch, syn_model):
    calls = _stub_chain(monkeypatch, findings=[])
    text, stats = await dr.research_question(QUESTION)
    assert text == "" and stats["reason"] == "no_findings"
    assert not calls["invoke"]


# --- the curiosity loop routes by source ------------------------------------

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


def _loop(monkeypatch, queue):
    lp = object.__new__(hb.HeartbeatLoop)
    seen = {"think": 0, "evidence": 0}

    async def _meta(_self, query):
        seen["think"] += 1
        return ANSWER, {}

    async def _rq(question, **kw):
        seen["evidence"] += 1
        return ANSWER, {"sources": 3, "findings": 2, "entail_dropped": 0, "evidence_first": True}

    monkeypatch.setattr(hb.HeartbeatLoop, "_think_query_meta", _meta, raising=False)
    monkeypatch.setattr(dr, "research_question", _rq)

    async def _closure(_self, topic, result):
        return True

    monkeypatch.setattr(hb.HeartbeatLoop, "_curiosity_closure_check", _closure)

    async def _follow(_self, topic, findings):
        return None

    monkeypatch.setattr(hb.HeartbeatLoop, "_send_curiosity_followup", _follow)
    monkeypatch.setattr(hb, "get_services", lambda: _Svc(queue), raising=False)
    monkeypatch.setattr(brain, "get_services", lambda: _Svc(queue))
    import app.tools.native_search as ns
    monkeypatch.setattr(ns, "search_health", lambda: 1.0)
    return lp, seen


@pytest.mark.asyncio
async def test_a_dossier_question_takes_the_evidence_path(monkeypatch, db):
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'dossier_open_question', 0.6, 'pending', 0)", (QUESTION,))
    queue = CuriosityQueue(db)
    lp, seen = _loop(monkeypatch, queue)
    out = await lp._research_one_curiosity(_Svc(queue))
    assert out.startswith("CURIOSITY RESOLVED"), out
    assert seen == {"think": 0, "evidence": 1}
    assert db.fetchone("SELECT status FROM curiosity_queue")["status"] == "resolved"


@pytest.mark.asyncio
async def test_an_agent_failure_keeps_the_memory_first_path(monkeypatch, db):
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'agent_failure', 0.6, 'pending', 0)",
               ("What did the owner name the scheduler component?",))
    queue = CuriosityQueue(db)
    lp, seen = _loop(monkeypatch, queue)
    out = await lp._research_one_curiosity(_Svc(queue))
    assert out.startswith("CURIOSITY RESOLVED"), out
    assert seen == {"think": 1, "evidence": 0}


@pytest.mark.asyncio
async def test_no_sources_under_degraded_search_is_deferred_not_burned(monkeypatch, db):
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES (?, 'dossier_open_question', 0.6, 'pending', 0)", (QUESTION,))
    queue = CuriosityQueue(db)
    lp, seen = _loop(monkeypatch, queue)

    async def _rq_empty(question, **kw):
        return "", {"sources": 0, "reason": "no_sources"}

    monkeypatch.setattr(dr, "research_question", _rq_empty)
    import app.tools.native_search as ns
    monkeypatch.setattr(ns, "search_health", lambda: 0.1)
    out = await lp._research_one_curiosity(_Svc(queue))
    assert out.startswith("CURIOSITY DEFERRED"), out
    row = db.fetchone("SELECT status, attempts FROM curiosity_queue")
    assert row["status"] == "pending" and row["attempts"] == 0
