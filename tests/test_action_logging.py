"""Tests for action logging tool — structured action audit trail."""

from __future__ import annotations

import json

import pytest

from app.tools.action_logging import log_action


def test_log_action_success(db):
    """log_action logs successful actions to the database."""
    log_action("email", {"to": "test@example.com"}, "sent", True)
    rows = db.fetchall("SELECT * FROM action_log WHERE action_type = 'email'")
    assert len(rows) >= 1
    assert rows[0]["success"] == 1


def test_log_action_failure(db):
    """log_action logs failed actions to the database."""
    log_action("webhook", {"url": "https://example.com"}, "rate limited", False)
    rows = db.fetchall("SELECT * FROM action_log WHERE action_type = 'webhook'")
    assert len(rows) >= 1
    assert rows[0]["success"] == 0


def test_log_action_with_dict_params(db):
    """log_action handles dict params and serializes to JSON."""
    log_action("calendar", {"action": "create", "title": "Meeting"}, "created", True)
    rows = db.fetchall("SELECT * FROM action_log WHERE action_type = 'calendar'")
    assert len(rows) >= 1
    params = json.loads(rows[0]["params"])
    assert params["action"] == "create"


def test_log_action_truncates_result(db):
    """log_action truncates result to 2000 chars."""
    long_result = "x" * 3000
    log_action("test", {}, long_result, True)
    rows = db.fetchall("SELECT * FROM action_log WHERE action_type = 'test'")
    assert len(rows) >= 1
    assert len(rows[0]["result"]) <= 2000
