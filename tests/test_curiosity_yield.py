"""Curiosity yield (audit 2026-09-01).

Measured before the change: 97 of 107 research runs unresolved in 14 days,
8 of 13 dossier 'resolutions' were the literal monitor-status string
'[provisional] no change | last: …', the agent loop raw-INSERTed failures into
the queue past dedup/cooldown/probe gates (birthing five duplicate rate-limiter
lessons), owner sanity probes ('in one word, are you operational') were
researched six times, and 32 dossier tensions had 0 resolutions because they
starved behind 0.6-0.7 open questions.
"""
from __future__ import annotations

from app.core.curiosity import CuriosityQueue, _looks_like_operator_probe
from app.monitors.heartbeat_loop import _provisional_acceptable


# --- provisional resolutions must be knowledge, not status strings ----------

def test_status_string_is_not_a_provisional_answer():
    assert _provisional_acceptable(
        "Can India's markets absorb $50B of inflows?",
        "[provisional] no change | last: 2026-08-18T15:59:00Z") is False
    assert _provisional_acceptable("Can X scale?", "no change | last: 2026-08-18") is False


def test_provisional_answer_needs_a_date_and_a_source():
    topic = "Can CV-QKD maintain its key-rate advantage over DV-QKD in long-distance links?"
    undated = ("CV-QKD keeps a key-rate advantage on metropolitan links while DV-QKD "
               "wins beyond 100 km, according to researchers.")
    assert _provisional_acceptable(topic, undated) is False
    dated_sourced = ("On August 28, 2026 a Nature Photonics paper (nature.com) reported that "
                     "CV-QKD keeps its key-rate advantage on metropolitan links while DV-QKD "
                     "wins beyond 100 km.")
    assert _provisional_acceptable(topic, dated_sourced) is True


# --- operator probes and self-references never become research ---------------

def test_probe_detector():
    assert _looks_like_operator_probe("in one word, are you operational")
    assert _looks_like_operator_probe("what did your monitors learn about ai developments in the last day or two? give me the two or three")
    assert _looks_like_operator_probe("reply with exactly: operational")
    assert _looks_like_operator_probe("hi")
    assert not _looks_like_operator_probe("How does the EU AI Act classify general-purpose models?")
    assert not _looks_like_operator_probe("Why did Brent crude fall below $90 after the Hormuz deal?")


def test_queue_rejects_probes_from_organic_sources_only(db):
    q = CuriosityQueue(db)
    assert q.add("in one word, are you operational", source="reflexion_failure", urgency=0.7) == -1
    assert q.add("what did your monitors learn about ai developments in the last day or two", source="tool_failure", urgency=0.7) == -1
    assert db.fetchone("SELECT COUNT(*) AS c FROM curiosity_queue")["c"] == 0
    ok = q.add("How does the EU AI Act classify general-purpose models?", source="gap_detection", urgency=0.6)
    assert ok and ok > 0
    # dossier-derived questions are never screened as probes
    ok2 = q.add("AI and ML: what are your monitors' current benchmarks?", source="dossier_open_question", urgency=0.6)
    assert ok2 and ok2 > 0


# --- tensions get a quota --------------------------------------------------

def test_every_third_pick_prefers_a_pending_tension(db):
    q = CuriosityQueue(db)
    for i in range(4):
        db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
                   "VALUES (?, 'dossier_open_question', 0.8, 'pending', 0)",
                   (f"Open question number {i} about a domain",))
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
               "VALUES ('Macroeconomics says 5.3% but Finance says 3.8% for GDP growth', "
               "'dossier_tension', 0.5, 'pending', 0)")
    picks = [q.get_next().source for _ in range(3)]
    assert picks[:2] == ["dossier_open_question", "dossier_open_question"]
    assert picks[2] == "dossier_tension"


def test_quota_falls_through_when_no_tension_pending(db):
    q = CuriosityQueue(db)
    for i in range(3):
        db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts) "
                   "VALUES (?, 'dossier_open_question', 0.8, 'pending', 0)",
                   (f"Open question number {i} about a domain",))
    assert [q.get_next().source for _ in range(3)] == ["dossier_open_question"] * 3


# --- agent loop routes through the queue gate ------------------------------

def test_agent_loop_failures_go_through_queue_add(db, monkeypatch):
    from types import SimpleNamespace
    from app.core import agent_loop as al

    calls = []

    def _fake_add(self, topic, source="gap_detection", urgency=0.5):
        calls.append((topic, source, urgency))
        return 1

    monkeypatch.setattr(al.CuriosityQueue if hasattr(al, "CuriosityQueue") else CuriosityQueue, "add", _fake_add)
    monkeypatch.setattr("app.database.get_db", lambda: db)
    failed_step = SimpleNamespace(status=al.STEP_FAILED, action={"tool": "web_search"},
                                  description="Find the current ECB deposit rate after the September meeting",
                                  attempts=2, critique="")
    plan = SimpleNamespace(steps=[failed_step])
    result = SimpleNamespace(success=False, plan=plan, query="q", answer="", iterations=1)
    loop = al.AgentLoop.__new__(al.AgentLoop)
    loop._learn_from_run_sync(result)
    assert calls and calls[0][1] == "agent_failure" and calls[0][2] < 0.7
    assert db.fetchone("SELECT COUNT(*) AS c FROM curiosity_queue")["c"] == 0  # no raw INSERT


def test_dossier_prompts_ask_for_researchable_questions_and_watch_for_line():
    from app.core import dossiers
    for tpl in (dossiers._UPDATE_PROMPT, dossiers._WORLD_PROMPT):
        assert "answerable TODAY" in tpl
        assert "Watch for:" in tpl
