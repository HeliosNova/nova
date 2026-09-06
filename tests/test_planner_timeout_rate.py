"""A failure count is not a failure rate (2026-09-06).

Every planning failure since 2026-08-29 has been a `TimeoutError` against the
60s `INTERNAL_LLM_TIMEOUT`, and the obvious move was to raise it. The measured
week says otherwise:

    day        planned  timeout  rate
    2026-08-31      9       6    40%
    2026-09-02      7       4    36%
    2026-09-04     14       1     6%
    2026-09-05     35       3     7%
    2026-09-06     27       0     0%

The rate collapsed to zero while volume nearly quadrupled, so the timeout was a
symptom of the 08-30..09-03 scheduler thrash that other work has since fixed —
not a ceiling that needs raising. Raising a setting shared by eleven modules to
fix something currently producing zero failures would also have been unfalsifi-
able: `INTERNAL_LLM_TIMEOUT` is one value for correction extraction, skills,
routing, reflexion and the prompt optimizer alike.

So what shipped is the measurement that decides the question WHEN IT RECURS.
The report used to say "planner failed: TimeoutError x3", which reads exactly
the same at 3-of-3 and at 3-of-38 — its own version of the mistake
schedule_pressure made by reporting demand with no before. It now carries the
denominator, and the slowest plan that actually finished: well under 60s means
a timeout is queueing behind a model load and the ceiling is irrelevant; near
60s means the ceiling is genuinely tight. Opposite fixes.
"""
from __future__ import annotations

from app.monitors import engineering_report as er

PRE = "2026-09-06 09:00:00,000 [INFO] app.core.planning []: "
WARN = "2026-09-06 09:00:00,000 [WARNING] app.core.planning []: "


def _log(tmp_path, lines):
    p = tmp_path / "nova-app.log"
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)


def _ok(secs, steps=3):
    return f"{PRE}[planning] plan ready in {secs}s ({steps} steps, multi_step)" + chr(10)


def _timeout():
    return f"{WARN}Planning failed: TimeoutError() after 60.0s" + chr(10)


def test_a_successful_plan_records_how_long_it_took(tmp_path):
    got = er.planner_health(1, _log(tmp_path, [_ok(4.2), _ok(8.1)]), today="2026-09-06")
    assert got["planned"] == 2 and got["failed"] == 0
    assert got["fail_rate"] == 0.0
    assert got["slowest_ok"] == 8.1


def test_the_rate_distinguishes_a_bad_afternoon_from_an_outage(tmp_path):
    """3 timeouts out of 38 and 3 out of 3 used to render identically."""
    afternoon = er.planner_health(
        1, _log(tmp_path, [_ok(5)] * 35 + [_timeout()] * 3), today="2026-09-06")
    assert round(afternoon["fail_rate"], 2) == 0.08

    outage = er.planner_health(
        1, _log(tmp_path, [_timeout()] * 3), today="2026-09-06")
    assert outage["fail_rate"] == 1.0
    assert "slowest_ok" not in outage, "nothing finished, so there is no margin to report"


def test_yesterdays_real_numbers_raise_nothing(monkeypatch):
    """27 planned, 0 timeouts. The field speaks; the alarm stays quiet."""
    _stub(monkeypatch, {"truncations": {}, "plan_failures": {},
                        "planned": 27, "failed": 0, "fail_rate": 0.0, "slowest_ok": 6.4})
    status, _summary, fields = er.build_report(_DB())
    assert "planning 27 ok / 0 timeout (0%)" in fields["truncation"]
    assert "slowest plan 6s" in fields["truncation"]
    assert not [k for k in fields if k.startswith("look_at")]
    assert status == "info"


def test_the_08_31_rate_would_have_been_named(monkeypatch):
    _stub(monkeypatch, {"truncations": {}, "plan_failures": {"TimeoutError": 6},
                        "planned": 9, "failed": 6, "fail_rate": 0.4, "slowest_ok": 7.2})
    _status, _summary, fields = er.build_report(_DB())
    look = " ".join(str(v) for k, v in fields.items() if k.startswith("look_at"))
    assert "40% of plans timed out (6/15)" in look


def test_a_fast_slowest_plan_points_AWAY_from_the_ceiling(monkeypatch):
    """7s of a 60s budget means the failures are not slow generation."""
    _stub(monkeypatch, {"truncations": {}, "plan_failures": {"TimeoutError": 6},
                        "planned": 9, "failed": 6, "fail_rate": 0.4, "slowest_ok": 7.2})
    _status, _summary, fields = er.build_report(_DB())
    look = " ".join(str(v) for k, v in fields.items() if k.startswith("look_at"))
    assert "queueing behind a model load" in look


def test_a_slow_slowest_plan_points_AT_the_ceiling(monkeypatch):
    """A plan that needed 47s of 60 says the budget really is tight."""
    _stub(monkeypatch, {"truncations": {}, "plan_failures": {"TimeoutError": 6},
                        "planned": 9, "failed": 6, "fail_rate": 0.4, "slowest_ok": 47.0})
    _status, _summary, fields = er.build_report(_DB())
    look = " ".join(str(v) for k, v in fields.items() if k.startswith("look_at"))
    assert "the ceiling is tight" in look


def test_a_high_rate_on_a_tiny_sample_is_not_an_alarm(monkeypatch):
    """One timeout out of two is noise, not a regression."""
    _stub(monkeypatch, {"truncations": {}, "plan_failures": {"TimeoutError": 1},
                        "planned": 1, "failed": 1, "fail_rate": 0.5, "slowest_ok": 5.0})
    _status, _summary, fields = er.build_report(_DB())
    assert not [k for k in fields if k.startswith("look_at")]


def test_planning_logs_the_duration_on_both_paths():
    """Neither number exists without the other: a duration only on success
    cannot tell you how close a timeout came."""
    import inspect
    from app.core import planning
    src = inspect.getsource(planning.create_plan)
    assert "[planning] plan ready in %.1fs" in src
    assert "after %.1fs" in src, "a failure must carry its elapsed time too"
    assert src.index("t0 = time.monotonic()") < src.index("asyncio.wait_for")


class _DB:
    def fetchone(self, _q, _a=()):
        return {"c": 0, "oldest": None, "d": None}

    def fetchall(self, _q, _a=()):
        return []


def _stub(monkeypatch, ph):
    monkeypatch.setattr(er, "planner_health", lambda *a, **k: ph)
    monkeypatch.setattr(er, "curiosity_stage1", lambda *a, **k: None)
    monkeypatch.setattr(er, "cascade_support", lambda *a, **k: None)
    monkeypatch.setattr(er, "product_quality", lambda *a, **k: None)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: None)
    monkeypatch.setattr(pw, "schedule_pressure", lambda db, **k: {
        "ratio": 0.95, "delivered": 100, "demanded": 105, "starved": []})
    monkeypatch.setattr(pw, "constant_monitors", lambda db, **k: [])
    monkeypatch.setattr(pw, "snapshot", lambda db, **k: [])
    import app.monitors.health_checks as hc
    monkeypatch.setattr(hc, "entail_gate_totals", lambda d, *a, **k: (0, 0))
    import app.core.forecasts as fc
    monkeypatch.setattr(fc, "calibration", lambda db, **k: None)


# ---------------------------------------------------------------------------
# The ceiling itself. Measured, then split by who is waiting.
# ---------------------------------------------------------------------------
import pytest


@pytest.mark.asyncio
async def test_background_planning_gets_room_for_a_model_swap(monkeypatch):
    """The tail is a model load, not slow generation, so the ceiling must
    absorb one. Background work pays nothing for waiting."""
    seen = {}

    async def _slow(*_a, **_k):
        return '{"steps": [{"description": "d", "tool": "none"}], "complexity": "simple"}'

    real = __import__("asyncio").wait_for

    async def _spy(aw, timeout):
        seen["timeout"] = timeout
        return await real(aw, timeout)

    import asyncio as _a
    from app.core import llm, planning
    monkeypatch.setattr(llm, "invoke_nothink", _slow)
    monkeypatch.setattr(_a, "wait_for", _spy)

    await planning.create_plan("q", ["none"], timeout=120.0)
    assert seen["timeout"] == 120.0


@pytest.mark.asyncio
async def test_interactive_planning_keeps_the_shorter_budget(monkeypatch):
    """A user waiting two minutes for a PLAN is worse than an answer written
    without one — planning failure is graceful."""
    seen = {}

    async def _ok(*_a, **_k):
        return '{"steps": [{"description": "d", "tool": "none"}], "complexity": "simple"}'

    import asyncio as _a
    real = _a.wait_for

    async def _spy(aw, timeout):
        seen["timeout"] = timeout
        return await real(aw, timeout)

    from app.core import llm, planning
    from app.config import config
    monkeypatch.setattr(llm, "invoke_nothink", _ok)
    monkeypatch.setattr(_a, "wait_for", _spy)

    await planning.create_plan("q", ["none"])
    assert seen["timeout"] == float(config.INTERNAL_LLM_TIMEOUT)


def test_only_the_monitor_channel_gets_the_larger_ceiling():
    """The split is by CHANNEL, not by intent: the 56.5s plan observed on
    2026-09-06 was Curiosity Research on channel=monitor, behind a batch of
    seven digests."""
    import inspect
    from app.core import brain
    src = inspect.getsource(brain._build_messages)
    assert 'channel == "monitor"' in src
    assert "120.0" in src and "timeout=plan_timeout" in src


def test_the_shared_timeout_is_not_raised_for_everyone():
    """INTERNAL_LLM_TIMEOUT is one value for correction extraction, skills,
    routing, reflexion and the prompt optimizer. Raising it globally to fix the
    planner would change ten unrelated call sites, and none of them were
    measured. This is a per-call-site adjustment, the gsw.py idiom."""
    from app.config import config
    assert config.INTERNAL_LLM_TIMEOUT == 60
