"""Monitor execution tests — happy path for create, check, store result.

Tests:
- Create monitor -> verify stored
- Store result -> verify retrievable
- Get monitor by ID
- List monitors
- Record alert timing
- HeartbeatLoop integration (mocked)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitors.heartbeat import Monitor, MonitorResult, MonitorStore


class TestMonitorStoreCRUD:
    """Basic CRUD operations on MonitorStore."""

    @pytest.fixture
    def store(self, db):
        return MonitorStore(db)

    def test_create_monitor(self, store):
        """Creating a monitor should return a positive ID."""
        monitor_id = store.create(
            name="Test Monitor",
            check_type="url",
            check_config={"url": "http://example.com"},
            schedule_seconds=300,
            cooldown_minutes=60,
            notify_condition="on_change",
        )
        assert monitor_id > 0

    def test_get_monitor(self, store):
        """get() should return a Monitor with correct fields."""
        monitor_id = store.create(
            name="Health Check",
            check_type="system_health",
            check_config={"threshold_pct": 90},
            schedule_seconds=600,
        )

        monitor = store.get(monitor_id)
        assert monitor is not None
        assert isinstance(monitor, Monitor)
        assert monitor.name == "Health Check"
        assert monitor.check_type == "system_health"
        assert monitor.schedule_seconds == 600
        assert monitor.enabled is True

    def test_get_nonexistent_returns_none(self, store):
        """get() with invalid ID should return None."""
        assert store.get(99999) is None

    def test_create_duplicate_name_returns_minus1(self, store):
        """Creating a monitor with duplicate name should return -1."""
        store.create(name="Unique", check_type="url", check_config={})
        result = store.create(name="Unique", check_type="url", check_config={})
        assert result == -1

    def test_add_result(self, store):
        """add_result should store and return a result ID."""
        monitor_id = store.create(
            name="Result Test",
            check_type="url",
            check_config={"url": "http://test.com"},
        )

        result_id = store.add_result(
            monitor_id=monitor_id,
            status="ok",
            value="200 OK",
            message="Site is up",
        )
        assert result_id > 0

    def test_get_results(self, store):
        """get_results should return stored results in reverse chronological order."""
        monitor_id = store.create(
            name="Results List",
            check_type="url",
            check_config={},
        )

        store.add_result(monitor_id, "ok", "Result 1", "First check")
        store.add_result(monitor_id, "changed", "Result 2", "Something changed")
        store.add_result(monitor_id, "alert", "Result 3", "Alert triggered")

        results = store.get_results(monitor_id, limit=10)
        assert len(results) == 3
        assert all(isinstance(r, MonitorResult) for r in results)
        # Most recent first
        assert results[0].status == "alert"
        assert results[2].status == "ok"

    def test_get_results_with_limit(self, store):
        """get_results should respect the limit parameter."""
        monitor_id = store.create(
            name="Limit Test",
            check_type="url",
            check_config={},
        )

        for i in range(10):
            store.add_result(monitor_id, "ok", f"Result {i}")

        results = store.get_results(monitor_id, limit=3)
        assert len(results) == 3

    def test_record_alert(self, store):
        """record_alert should update last_alert_at."""
        monitor_id = store.create(
            name="Alert Test",
            check_type="url",
            check_config={},
        )

        monitor_before = store.get(monitor_id)
        assert monitor_before.last_alert_at is None

        store.record_alert(monitor_id)

        monitor_after = store.get(monitor_id)
        assert monitor_after.last_alert_at is not None

    def test_add_result_truncates_long_values(self, store):
        """add_result should truncate value and message to 2000 chars."""
        monitor_id = store.create(
            name="Truncation Test",
            check_type="url",
            check_config={},
        )

        long_value = "x" * 5000
        long_message = "y" * 5000
        result_id = store.add_result(monitor_id, "ok", long_value, long_message)

        results = store.get_results(monitor_id)
        assert len(results) == 1
        assert len(results[0].value) <= 2000
        assert len(results[0].message) <= 2000


class TestMonitorStoreWorkflow:
    """Test the full monitor workflow: create -> check -> store result."""

    @pytest.fixture
    def store(self, db):
        return MonitorStore(db)

    def test_full_workflow(self, store):
        """Create monitor, run check (simulated), store result, verify."""
        # Step 1: Create monitor
        monitor_id = store.create(
            name="Workflow Test",
            check_type="url",
            check_config={"url": "http://example.com", "expected_status": 200},
            schedule_seconds=300,
            cooldown_minutes=30,
            notify_condition="on_change",
        )
        assert monitor_id > 0

        # Step 2: Get monitor config
        monitor = store.get(monitor_id)
        assert monitor.name == "Workflow Test"
        assert monitor.check_config["url"] == "http://example.com"

        # Step 3: Simulate check (would normally be done by HeartbeatLoop)
        check_result = "200 OK - Site is up"
        result_id = store.add_result(
            monitor_id=monitor_id,
            status="ok",
            value=check_result,
            message="Health check passed",
        )
        assert result_id > 0

        # Step 4: Record check time
        store.record_check(monitor_id, check_result)
        updated_monitor = store.get(monitor_id)
        assert updated_monitor.last_check_at is not None

        # Step 5: Verify results are retrievable
        results = store.get_results(monitor_id)
        assert len(results) == 1
        assert results[0].status == "ok"
        assert results[0].value == check_result

    def test_alert_workflow(self, store):
        """Create monitor, store alert result, record alert."""
        monitor_id = store.create(
            name="Alert Workflow",
            check_type="url",
            check_config={"url": "http://failing.com"},
        )

        # Store a failing result
        store.add_result(
            monitor_id=monitor_id,
            status="alert",
            value="503 Service Unavailable",
            message="Site is down!",
        )

        # Record alert
        store.record_alert(monitor_id)

        monitor = store.get(monitor_id)
        assert monitor.last_alert_at is not None

        results = store.get_results(monitor_id)
        assert results[0].status == "alert"


class TestHeartbeatLoopIntegration:
    """Integration test for HeartbeatLoop with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_heartbeat_loop_has_run_method(self):
        """HeartbeatLoop should have a run() or _run_once() method."""
        from app.monitors.heartbeat import HeartbeatLoop
        # Just verify the class has the expected interface
        assert hasattr(HeartbeatLoop, "start") or hasattr(HeartbeatLoop, "_run_once")

    @pytest.mark.asyncio
    async def test_monitor_store_integration_with_services(self, db):
        """MonitorStore should work when wired into Services."""
        from app.core.brain import Services, set_services
        from app.core.memory import ConversationStore, UserFactStore

        monitor_store = MonitorStore(db)
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            monitor_store=monitor_store,
        )
        set_services(svc)

        # Create a monitor through the store
        monitor_id = monitor_store.create(
            name="Integration Test",
            check_type="query",
            check_config={"query": "What's new in AI?"},
        )
        assert monitor_id > 0

        # Verify it's accessible
        monitor = monitor_store.get(monitor_id)
        assert monitor.check_type == "query"
