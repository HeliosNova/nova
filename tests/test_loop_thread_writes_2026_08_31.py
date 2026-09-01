"""Event-loop DB-write regression tests (2026-08-31).

The `_warn_if_event_loop` tripwire logged 22 distinct WRITE statements running
on the event-loop thread on 2026-08-31 alone (78 sites total that day; the
2026-06-11 lock-convoy incident and the 54h freeze are this class). The fix
moved each async→sync boundary onto a worker thread with asyncio.to_thread —
one wrap per call chain, not per statement.

These tests run the real code paths inside a live event loop against an
instrumented DB and assert (a) no write executes on the loop thread and
(b) the writes still happen (no over-threading that drops behavior).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.tools.base import EPHEMERAL_REQUEST


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


class LoopGuardDB:
    """Wraps a SafeDB; records which thread ran each write statement."""

    def __init__(self, real):
        self._real = real
        self.loop_thread_writes: list[str] = []
        self.loop_thread_id: int | None = None

    def _check(self, sql: str):
        if self.loop_thread_id is not None and threading.get_ident() == self.loop_thread_id:
            head = sql.lstrip()[:10].upper()
            if head.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE")):
                self.loop_thread_writes.append(sql[:80])

    def execute(self, sql, params=()):
        self._check(sql)
        return self._real.execute(sql, params)

    def executemany(self, sql, params_list):
        self._check(sql)
        return self._real.executemany(sql, params_list)

    def fetchone(self, sql, params=()):
        return self._real.fetchone(sql, params)

    def fetchall(self, sql, params=()):
        return self._real.fetchall(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def guard_db(db):
    # auto_tool_candidates / curiosity_queue are created by their stores'
    # DDL, not init_schema — seed them before arming the loop guard.
    from app.core.curiosity import CuriosityQueue
    from app.core.tool_triggers import ToolCandidateStore
    ToolCandidateStore(db=db)
    for stmt in CuriosityQueue._SCHEMA.strip().split(";"):
        if stmt.strip():
            db.execute(stmt)
    return LoopGuardDB(db)


class TestLearnFromRunOffLoop:
    def _result(self):
        from app.core.agent_loop import (STEP_DONE, STEP_FAILED, AgentResult,
                                         Plan, Step)
        steps = [
            Step(id=1, description="look up GPU prices", status=STEP_DONE,
                 action={"tool": "web_search"}),
            Step(id=2, description="compare with historical data", status=STEP_DONE,
                 action={"tool": "calculator"}),
            Step(id=3, description="fetch vendor spec sheet", status=STEP_FAILED,
                 action={"tool": "web_search"}, attempts=2, critique="404s"),
        ]
        return AgentResult(query="compare GPU prices across vendors",
                           answer="done", plan=Plan(goal="g", steps=steps),
                           scratchpad_text="", iterations=3,
                           duration_seconds=1.0, success=True)

    def test_writes_happen_off_loop(self, guard_db, monkeypatch):
        import app.database as database
        from app.core.agent_loop import AgentLoop
        monkeypatch.setattr(database, "get_db", lambda: guard_db)

        loop_obj = AgentLoop.__new__(AgentLoop)  # no ctor deps needed
        result = self._result()

        async def run():
            guard_db.loop_thread_id = threading.get_ident()
            await loop_obj._learn_from_run(result, None)

        asyncio.run(run())
        assert guard_db.loop_thread_writes == [], \
            f"writes ran on the event loop: {guard_db.loop_thread_writes}"
        # Behavior preserved: candidate + gap + curiosity rows written
        assert guard_db.fetchone(
            "SELECT COUNT(*) c FROM auto_tool_candidates")["c"] == 1
        assert guard_db.fetchone("SELECT COUNT(*) c FROM capability_gaps")["c"] == 1
        assert guard_db.fetchone(
            "SELECT COUNT(*) c FROM curiosity_queue WHERE source='agent_failure'")["c"] == 1

    def test_ephemeral_and_scaffold_gates_still_hold(self, guard_db, monkeypatch):
        import app.database as database
        from app.core.agent_loop import AgentLoop
        monkeypatch.setattr(database, "get_db", lambda: guard_db)
        loop_obj = AgentLoop.__new__(AgentLoop)

        async def run():
            EPHEMERAL_REQUEST.set(True)
            await loop_obj._learn_from_run(self._result(), None)
            EPHEMERAL_REQUEST.set(False)
            r2 = self._result()
            r2.query = "=== WILL-MODULE TASK ===\ninternal"
            await loop_obj._learn_from_run(r2, None)

        asyncio.run(run())
        assert guard_db.fetchone(
            "SELECT COUNT(*) c FROM auto_tool_candidates")["c"] == 0


class TestToolTriggerOffLoop:
    def test_record_path_off_loop(self, guard_db, monkeypatch):
        from types import SimpleNamespace

        import app.database as database
        from app.config import config
        from app.core import tool_triggers
        monkeypatch.setattr(database, "get_db", lambda: guard_db)
        monkeypatch.setattr(config, "ENABLE_AUTONOMOUS_TOOL_CREATION", True,
                            raising=False)
        svc = SimpleNamespace(custom_tools=object(), skills=None)
        tool_results = [{"tool": "web_search"}, {"tool": "calculator"},
                        {"tool": "web_search"}]

        async def run():
            guard_db.loop_thread_id = threading.get_ident()
            await tool_triggers.maybe_trigger_tool_creation(
                "compare 3090 vs 4090 street prices", tool_results, svc)

        asyncio.run(run())
        assert guard_db.loop_thread_writes == [], \
            f"writes ran on the event loop: {guard_db.loop_thread_writes}"
        assert guard_db.fetchone(
            "SELECT COUNT(*) c FROM auto_tool_candidates")["c"] == 1


class TestStorylineForecastOffLoop:
    @pytest.mark.asyncio
    async def test_update_story_records_forecast_off_loop(self, guard_db, monkeypatch):
        from unittest.mock import AsyncMock

        import app.core.storylines as sl
        guard_db.execute(
            "INSERT INTO storylines (story_key, title, summary, monitors_csv, "
            "update_count, last_updated) VALUES ('opec','OPEC supply','prior',"
            "'Energy',1,datetime('now'))")
        out = ("OPEC extended cuts through Q1.\nCHANGED: cuts extended\n"
               "FORECAST: Brent stays above $80 through November (confidence 0.6)")
        monkeypatch.setattr(sl, "_bg_invoke", AsyncMock(return_value=out))
        guard_db.loop_thread_id = threading.get_ident()
        res = await sl._update_story(
            guard_db, {"key": "opec", "title": "OPEC supply",
                       "developments": ["cuts extended at Vienna meeting"],
                       "monitors": ["Energy Watch"]})
        assert res is not None
        assert guard_db.loop_thread_writes == [], \
            f"writes ran on the event loop: {guard_db.loop_thread_writes}"
