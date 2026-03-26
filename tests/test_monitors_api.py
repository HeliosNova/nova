"""Item 65: Test monitor API HTTP endpoints.

Tests monitor CRUD and validation. Uses the same fixture pattern as
test_security.py: create a bare client, then inject services per-test.
"""

from __future__ import annotations

import pytest

from app.core.brain import Services, set_services, get_services
from app.core.memory import ConversationStore, UserFactStore
from app.monitors.heartbeat import MonitorStore


@pytest.fixture
def monitor_client(db):
    """FastAPI test client with MonitorStore properly injected.

    Follows the pattern from test_security.py's TestMonitorValidation:
    create a client, then set services including monitor_store right
    before making the request.
    """
    import importlib
    import app.config
    import app.auth

    importlib.reload(app.config)
    importlib.reload(app.auth)

    from fastapi.testclient import TestClient
    from app.main import app, _rate_limit_requests

    _rate_limit_requests.clear()

    ms = MonitorStore(db)
    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
        monitor_store=ms,
    )
    set_services(svc)
    tc = TestClient(app)
    # Re-inject after TestClient init (lifespan may have overwritten)
    set_services(svc)
    return tc


class TestMonitorListEndpoint:
    """GET /api/monitors"""

    def test_list_monitors_empty(self, monitor_client):
        resp = monitor_client.get("/api/monitors")
        assert resp.status_code == 200
        data = resp.json()
        assert "monitors" in data
        assert data["count"] == 0

    def test_list_monitors_after_create(self, monitor_client):
        monitor_client.post("/api/monitors", json={
            "name": "Test Monitor",
            "check_type": "search",
            "check_config": {"query": "test"},
        })
        resp = monitor_client.get("/api/monitors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1


class TestMonitorCreateEndpoint:
    """POST /api/monitors"""

    def test_create_monitor_success(self, monitor_client):
        resp = monitor_client.post("/api/monitors", json={
            "name": "Web Health Check",
            "check_type": "url",
            "check_config": {"url": "https://example.com"},
            "schedule_seconds": 300,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Web Health Check"
        assert data["check_type"] == "url"
        assert data["id"] is not None

    def test_create_monitor_invalid_check_type(self, monitor_client):
        resp = monitor_client.post("/api/monitors", json={
            "name": "Bad Monitor",
            "check_type": "dangerous_type",
        })
        assert resp.status_code == 422

    def test_create_monitor_invalid_name(self, monitor_client):
        resp = monitor_client.post("/api/monitors", json={
            "name": "<script>alert(1)</script>",
            "check_type": "search",
        })
        assert resp.status_code == 422

    def test_create_monitor_schedule_too_low(self, monitor_client):
        resp = monitor_client.post("/api/monitors", json={
            "name": "Fast Monitor",
            "check_type": "search",
            "schedule_seconds": 1,
        })
        assert resp.status_code == 422

    def test_create_duplicate_name_rejected(self, monitor_client):
        monitor_client.post("/api/monitors", json={
            "name": "Unique Monitor",
            "check_type": "search",
        })
        resp = monitor_client.post("/api/monitors", json={
            "name": "Unique Monitor",
            "check_type": "search",
        })
        assert resp.status_code == 409


class TestMonitorGetEndpoint:
    """GET /api/monitors/{id}"""

    def test_get_monitor_by_id(self, monitor_client):
        create_resp = monitor_client.post("/api/monitors", json={
            "name": "Get Test",
            "check_type": "search",
        })
        monitor_id = create_resp.json()["id"]

        resp = monitor_client.get(f"/api/monitors/{monitor_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Get Test"
        assert "results" in data

    def test_get_nonexistent_monitor(self, monitor_client):
        resp = monitor_client.get("/api/monitors/99999")
        assert resp.status_code == 404


class TestMonitorDeleteEndpoint:
    """DELETE /api/monitors/{id}"""

    def test_delete_monitor(self, monitor_client):
        create_resp = monitor_client.post("/api/monitors", json={
            "name": "Delete Me",
            "check_type": "search",
        })
        monitor_id = create_resp.json()["id"]

        resp = monitor_client.delete(f"/api/monitors/{monitor_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify it's gone
        resp = monitor_client.get(f"/api/monitors/{monitor_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_monitor(self, monitor_client):
        resp = monitor_client.delete("/api/monitors/99999")
        assert resp.status_code == 404


class TestMonitorTriggerEndpoint:
    """POST /api/monitors/{id}/trigger"""

    def test_trigger_nonexistent_monitor(self, monitor_client):
        resp = monitor_client.post("/api/monitors/99999/trigger")
        assert resp.status_code == 404

    def test_trigger_without_heartbeat(self, monitor_client):
        """Trigger should fail gracefully when heartbeat is not initialized."""
        create_resp = monitor_client.post("/api/monitors", json={
            "name": "Trigger Test",
            "check_type": "search",
        })
        monitor_id = create_resp.json()["id"]

        resp = monitor_client.post(f"/api/monitors/{monitor_id}/trigger")
        # Should return 503 because heartbeat loop is not initialized
        assert resp.status_code == 503
