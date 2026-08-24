"""Goal verify-gate: a zero baseline must not fail the goal (audit 2026-08-23).

Every recurring_curiosity goal ever pursued was marked FAILED by the metric
verify gate in daemon._pursue_goal despite executing successfully: the hourly
Curiosity Research monitor resolves the pending items long before the daemon's
pursue tick (which only fires at BUDGET_FULL after >=30min idle), so by pursue
time before_metric == 0. execute_goal runs ephemerally and writes nothing, so
after_metric == 0, and `after >= before` (0 >= 0) failed the goal by
construction. A zero baseline cannot regress — the work the goal was minted for
is already done, so a successful execution completes. A positive baseline that
does not improve must still fail (the gate's whole purpose).
"""

import pytest

from app.monitors.daemon import DaemonOrchestrator


def _seed_goal(db, topic: str) -> int:
    from app.core.curiosity import CuriosityQueue
    from app.core.goals import GoalStore

    # Ensure curiosity_queue exists (lazily created by CuriosityQueue) so the
    # metric snapshot exercises the real 0-vs-N path, not the unknown-table
    # fallback that would make the zero-baseline test pass vacuously.
    CuriosityQueue(db)
    # init_schema seeds a phase_0_bootstrap goal at priority 1.0 which would
    # outrank the test goal in get_next_pending — clear it.
    db.execute("DELETE FROM goals")
    return GoalStore(db).add(
        goal=f"Re-research and verify: {topic}",
        priority=0.8,
        source="derived",
        context={"source": "recurring_curiosity", "topic": topic},
    )


def _goal_status(db, goal_id: int) -> str:
    return db.fetchone("SELECT status FROM goals WHERE id=?", (goal_id,))["status"]


async def _fake_execute_ok(goal):
    return True, "Investigated thoroughly; findings grounded in three sources."


@pytest.mark.asyncio
async def test_zero_baseline_completes_not_fails(db, monkeypatch):
    # Queue is EMPTY for the topic (the hourly monitor already resolved it):
    # before=0, after=0 — the goal must complete, not fail on 0 >= 0.
    goal_id = _seed_goal(db, "quantum error correction milestones")
    import app.core.goals as goals_mod

    monkeypatch.setattr(goals_mod, "execute_goal", _fake_execute_ok)
    await DaemonOrchestrator(db)._pursue_goal()
    assert _goal_status(db, goal_id) == "completed", (
        "zero-baseline goal was failed by the verify gate (the 0 >= 0 bug)"
    )


@pytest.mark.asyncio
async def test_positive_baseline_no_improvement_still_fails(db, monkeypatch):
    # Gate integrity: pending items EXIST (before=2) and the execution does not
    # reduce them (after=2) — the goal must still fail verification.
    topic = "semiconductor export controls"
    goal_id = _seed_goal(db, topic)
    for _ in range(2):
        db.execute(
            "INSERT INTO curiosity_queue (topic, source, status) VALUES (?, 'test', 'pending')",
            (topic,),
        )
    import app.core.goals as goals_mod

    monkeypatch.setattr(goals_mod, "execute_goal", _fake_execute_ok)
    await DaemonOrchestrator(db)._pursue_goal()
    assert _goal_status(db, goal_id) == "failed", (
        "verify gate no longer fails a goal whose positive metric did not improve"
    )
