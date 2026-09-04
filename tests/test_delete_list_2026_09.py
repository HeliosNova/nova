"""Delete list, part 1 (audit 2026-09-01, owner-approved).

Loops that consumed GPU/DB writes while producing no verified behaviour change:
  - Skill Validation asked the skill's own name as the question and passed at
    the default score; Capability Review / Goal Derivation / Auto-Tool
    Synthesis had no inputs (7 of 8 goals failed, garbage titles);
  - a Lesson Quiz PASS credited times_helpful/retrieval_score — self-graded
    counters that then ordered retrieval, skill induction and principles;
  - trust scores were updated per tool call and gated nothing;
  - DPO pairs were written for a trainer archived in June;
  - RLVR signal collection had no consumer (flag off, 0 rows);
  - salience 'learned weights' were 12 junk tokens from one June test rating.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from app.monitors.monitor_store import MonitorStore


RETIRED_MONITORS = ("Skill Validation", "Capability Review", "Goal Derivation", "Auto-Tool Synthesis")


def test_retired_loops_are_seeded_disabled(db):
    store = MonitorStore(db)
    store.seed_defaults()
    for name in RETIRED_MONITORS:
        row = db.fetchone("SELECT enabled FROM monitors WHERE name = ?", (name,))
        assert row is not None, f"{name} should stay in the catalog (disabled), not vanish"
        assert int(row["enabled"]) == 0, f"{name} must be seeded disabled"
    # the loops that DO close feed the product and stay on
    for name in ("Knowledge Consolidation", "Forecast Resolution", "Storyline Tracker", "Curiosity Research"):
        row = db.fetchone("SELECT enabled FROM monitors WHERE name = ?", (name,))
        assert row is not None and int(row["enabled"]) == 1, name


def test_quiz_pass_no_longer_credits_the_lesson():
    from app.monitors.heartbeat_loop import HeartbeatLoop
    src = inspect.getsource(HeartbeatLoop._execute_quiz)
    assert "mark_lesson_helpful" not in src
    assert "quiz_failures = 0" in src, "the closure signal (clearing quiz_failures) stays"


def test_trust_records_in_memory_only():
    from app.core import trust as trust_mod
    db = MagicMock()
    db.fetchone.return_value = None
    cls = next(v for k, v in vars(trust_mod).items() if isinstance(v, type) and "Trust" in k)
    tm = cls.__new__(cls)
    tm._db = db
    tm._success_delta = getattr(trust_mod, "DEFAULT_SUCCESS_DELTA", 0.5)
    tm._failure_delta = getattr(trust_mod, "DEFAULT_FAILURE_DELTA", -2.0)
    tm._score = 62.0
    db.execute.reset_mock()
    tm.record_outcome("web_search", True)
    tm.record_outcome("web_search", False, action="x")
    tm.decay()
    assert db.execute.call_count == 0, "trust must not write to the database"
    assert tm.can_use("shell_exec") is True


@pytest.mark.asyncio
async def test_training_pair_writer_is_retired(db, tmp_path, monkeypatch):
    from app.config import config as _cfg
    from app.core.learning import LearningEngine
    path = tmp_path / "training_data.jsonl"
    monkeypatch.setattr(_cfg, "TRAINING_DATA_PATH", str(path), raising=False)
    eng = LearningEngine(db)
    await eng.save_training_pair(query="q", bad_answer="bad", good_answer="good", channel="api")
    assert not path.exists() or path.stat().st_size == 0


def test_no_pathway_watches_the_retired_trust_writer():
    """A liveness pathway must not outlive the writer it watches.

    trust stopped writing on 2026-09-01, so this entry reported DEAD every
    cycle from then on and the "all pathways alive" marker could never come
    back — which is how a canary stops being read.
    """
    from app.monitors.pathways import PATHWAYS
    names = {p.name for p in PATHWAYS}
    assert "trust_ledger" not in names,         "trust is in-memory since 2026-09-01; revive the writer before the pathway"
    tables = {p.table for p in PATHWAYS}
    assert "trust_scores" not in tables


def test_rlvr_and_grpo_modules_are_archived():
    import importlib.util
    for mod in ("app.core.rlvr", "app.core.grpo_dataset", "app.core.grpo_verifier",
                "app.monitors.domain_study_prompt"):
        assert importlib.util.find_spec(mod) is None, f"{mod} should be archived"
    from app.config import Config
    assert not hasattr(Config, "ENABLE_RLVR_SIGNALS")


def test_salience_ignores_learned_weights(db):
    from app.core import salience
    text = "boeing europe joint face base while"
    before = salience.score_text(db, text)
    db.execute("INSERT INTO salience_weights (topic, weight, updated_at) VALUES ('boeing', 4.0, datetime('now'))")
    after = salience.score_text(db, text)
    assert before == after


def test_daemon_no_longer_pursues_goals():
    from app.monitors import daemon as daemon_mod
    src = inspect.getsource(daemon_mod.DaemonOrchestrator)
    assert 'return {"action": "pursue_goal"}' not in src
