"""Schedule pressure: is the declared cadence actually being met? (2026-09-03)

Every monitor merely looked "a bit late", so nobody could see that the card
was 2.5x oversubscribed. Measured on the live install: 1,646 runs demanded per
week against 646 delivered (39%), with Curiosity Research at 20% of its
declared hourly cadence — which also degrades every priority rule, since a
starvation floor cannot discriminate when everything is overdue at once.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.monitors.pathways import (
    SCHEDULE_PRESSURE_FLOOR,
    liveness_report,
    schedule_pressure,
)
from app.monitors.monitor_store import MonitorStore


@pytest.fixture(autouse=True)
def _aged_install(db):
    """schedule_pressure refuses to judge an install younger than a day."""
    db.execute("UPDATE schema_version SET applied_at = datetime('now', '-30 days')")


def _mon(db, name, sched, runs, *, enabled=1, config="{}", age_days=30):
    """`age_days` matters: a monitor younger than the pressure window cannot
    have delivered a window's worth of runs, so it is excluded from the STARVED
    list (it still counts toward demand). Added 2026-09-05 after the Engineering
    Report named ITSELF the worst offender hours after being created."""
    db.execute("INSERT INTO monitors (name, check_type, check_config, schedule_seconds, "
               "enabled, cooldown_minutes, notify_condition, category, created_at) "
               "VALUES (?, 'system_health', ?, ?, ?, 60, 'on_change', 'system', "
               "datetime('now', ?))",
               (name, config, sched, enabled, f"-{age_days} days"))
    mid = db.fetchone("SELECT id FROM monitors WHERE name = ?", (name,))["id"]
    for i in range(runs):
        db.execute("INSERT INTO monitor_results (monitor_id, status, value, created_at) "
                   "VALUES (?, 'ok', 'x', datetime('now', ?))", (mid, f"-{i} hours"))
    return mid


def test_ratio_is_delivered_over_demanded(db):
    # hourly monitor: 168 demanded in 7 days, 42 delivered
    _mon(db, "Hourly", 3600, 42)
    p = schedule_pressure(db)
    assert p["demanded"] == 168
    assert p["delivered"] == 42
    assert p["ratio"] == pytest.approx(0.25, abs=0.01)


def test_a_monitor_cannot_bank_credit_for_extra_runs(db):
    """Running more often than declared must not mask another monitor's starvation."""
    _mon(db, "Overrun", 86400 * 7, 50)      # 1 demanded, 50 delivered
    _mon(db, "Starved", 3600, 0)            # 168 demanded, 0 delivered
    p = schedule_pressure(db)
    assert p["demanded"] == 169
    assert p["delivered"] == 1, "credit is capped at what was demanded"
    assert p["ratio"] < 0.02


def test_anchored_dailies_count_once_per_day(db):
    """Morning Check-in has a 24h interval AND an anchor hour; it can only run once."""
    _mon(db, "Morning Check-in", 86400, 7, config='{"anchor_hour": 7}')
    p = schedule_pressure(db)
    assert p["demanded"] == 7 and p["ratio"] == 1.0


def test_starved_list_names_the_worst_offenders(db):
    _mon(db, "Fine", 86400, 7)
    _mon(db, "Struggling", 3600, 34)
    _mon(db, "Dead last", 3600, 4)
    p = schedule_pressure(db)
    names = [s["name"] for s in p["starved"]]
    assert names[0] == "Dead last"
    assert "Struggling" in names
    assert p["starved"][0]["delivered"] == 4 and p["starved"][0]["demanded"] == 168


def test_disabled_monitors_are_not_demanded(db):
    _mon(db, "Off", 3600, 0, enabled=0)
    p = schedule_pressure(db)
    assert p["demanded"] == 0 and p["ratio"] is None


def test_severe_pressure_breaks_the_healthy_marker_so_it_delivers(db):
    """All writers alive but the schedule badly unmet is its own failure."""
    MonitorStore(db).seed_defaults()
    db.execute("UPDATE schema_version SET applied_at = datetime('now', '-30 days')")
    _mon(db, "Hourly starved", 3600, 1)     # drags the ratio under the floor
    # enough delivered runs that the ratio means something
    _mon(db, "Busy", 86400, 40)
    status, summary, fields = liveness_report(db)
    assert "schedule" in fields
    if status == "warning":
        assert "schedule is not being met" in summary
    assert SCHEDULE_PRESSURE_FLOOR == 0.25


def test_pressure_is_reported_even_when_everything_is_alive(db):
    MonitorStore(db).seed_defaults()
    _mon(db, "Hourly", 3600, 168)
    _status, _summary, fields = liveness_report(db)
    assert "schedule" in fields and "demanded runs delivered" in str(fields["schedule"])


def test_status_schema_carries_the_block():
    from app.schema import StatusResponse
    assert StatusResponse().schedule == {}
    r = StatusResponse(schedule={"demanded": 1646, "delivered": 646, "ratio": 0.39, "starved": []})
    assert r.schedule["ratio"] == 0.39
