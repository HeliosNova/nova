"""Tests for event-driven trigger system."""

import json

import pytest

from app.monitors.heartbeat import Monitor, MonitorStore


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------


def test_event_matches_exact():
    assert MonitorStore._event_matches("internal:lesson_saved", "internal:lesson_saved")


def test_event_matches_wildcard():
    assert MonitorStore._event_matches("webhook:github_push", "webhook:*")


def test_event_no_match():
    assert not MonitorStore._event_matches("internal:lesson_saved", "webhook:*")


def test_event_no_match_partial():
    assert not MonitorStore._event_matches("internal:lesson_saved", "internal:kg")


# ---------------------------------------------------------------------------
# Monitor dataclass with trigger fields
# ---------------------------------------------------------------------------


def test_monitor_trigger_fields():
    m = Monitor(
        id=1, name="test", check_type="query", check_config={},
        schedule_seconds=300, enabled=True, cooldown_minutes=60,
        notify_condition="on_change", last_check_at=None,
        last_alert_at=None, last_result=None, created_at="2026-01-01",
        trigger_events=["internal:lesson_saved", "internal:correction_detected"],
        trigger_mode="both",
    )
    assert m.trigger_mode == "both"
    assert len(m.trigger_events) == 2


def test_monitor_default_trigger_fields():
    m = Monitor(
        id=1, name="test", check_type="query", check_config={},
        schedule_seconds=300, enabled=True, cooldown_minutes=60,
        notify_condition="on_change", last_check_at=None,
        last_alert_at=None, last_result=None, created_at="2026-01-01",
    )
    assert m.trigger_mode == "schedule"
    assert m.trigger_events == []


# ---------------------------------------------------------------------------
# EventTrigger emit_event
# ---------------------------------------------------------------------------


def test_emit_event_noop_when_disabled():
    """emit_event should silently no-op when no trigger is configured."""
    from app.monitors.event_trigger import emit_event, set_event_trigger
    set_event_trigger(None)
    # Should not raise
    emit_event("internal:test", {"key": "value"})


# ---------------------------------------------------------------------------
# Event type validation
# ---------------------------------------------------------------------------


def test_event_type_format():
    """Event types must be namespace:name format."""
    import re
    pattern = re.compile(r"^[a-zA-Z0-9_]+:[a-zA-Z0-9_.]+$")
    assert pattern.match("internal:lesson_saved")
    assert pattern.match("webhook:github_push")
    assert pattern.match("file:config.changed")
    assert not pattern.match("no_namespace")
    assert not pattern.match("")
    assert not pattern.match("too:many:colons")
