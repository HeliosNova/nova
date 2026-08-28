"""Wall-clock anchored daily monitors (2026-08-27).

Interval scheduling stamps last_check_at when a check RAN, so queue latency
compounds day over day — Morning Check-in drifted 18:19 → 20:11 → 22:34 UTC
across three consecutive days (mid-afternoon local and sliding). Monitors
with check_config.anchor_hour are due once per LOCAL date, any time after
that local hour.
"""
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.monitors.monitor_store import MonitorStore


def _m(last_check_at=None):
    return SimpleNamespace(last_check_at=last_check_at)


def _at(hour, minute=0, day=27):
    return datetime(2026, 8, day, hour, minute, tzinfo=timezone.utc)


def _patched(now_utc):
    """Patch _local_now to a fixed UTC-as-local clock for determinism."""
    return patch.object(MonitorStore, "_local_now", staticmethod(lambda: now_utc))


class TestAnchoredDue:
    def test_before_anchor_hour_not_due(self):
        with _patched(_at(6, 30)):
            assert MonitorStore._anchored_due(_m(None), 8) is False

    def test_after_anchor_never_run_due(self):
        with _patched(_at(8, 5)):
            assert MonitorStore._anchored_due(_m(None), 8) is True

    def test_already_ran_today_not_due(self):
        with _patched(_at(14, 0)):
            assert MonitorStore._anchored_due(_m("2026-08-27 08:10:00"), 8) is False

    def test_ran_yesterday_due_after_anchor(self):
        with _patched(_at(8, 1)):
            assert MonitorStore._anchored_due(_m("2026-08-26 12:00:00"), 8) is True

    def test_ran_yesterday_before_anchor_not_due(self):
        # Ran yesterday, but it's only 05:00 local — waits for 08:00.
        with _patched(_at(5, 0)):
            assert MonitorStore._anchored_due(_m("2026-08-26 08:05:00"), 8) is False

    def test_no_daily_drift(self):
        # The failure mode: each run later than the last. Anchored, a run at
        # 11:40 (late, queue depth) still leaves tomorrow due at 08:00.
        with _patched(_at(8, 0, day=28)):
            assert MonitorStore._anchored_due(_m("2026-08-27 11:40:00"), 8) is True
