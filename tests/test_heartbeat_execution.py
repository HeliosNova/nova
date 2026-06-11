"""Tests for HeartbeatLoop._check_monitor() / _execute_check().

Covers URL monitor execution, error status derivation, reminder
auto-disable, and cooldown enforcement.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitors.heartbeat import (
    HeartbeatLoop,
    Monitor,
    MonitorStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monitor(
    *,
    id: int = 1,
    name: str = "Test Monitor",
    check_type: str = "url",
    check_config: dict | None = None,
    schedule_seconds: int = 300,
    enabled: bool = True,
    cooldown_minutes: int = 60,
    notify_condition: str = "always",
    last_check_at: str | None = None,
    last_alert_at: str | None = None,
    last_result: str | None = None,
    created_at: str = "2026-01-01T00:00:00",
) -> Monitor:
    """Create a Monitor dataclass with sensible defaults."""
    return Monitor(
        id=id,
        name=name,
        check_type=check_type,
        check_config=check_config or {},
        schedule_seconds=schedule_seconds,
        enabled=enabled,
        cooldown_minutes=cooldown_minutes,
        notify_condition=notify_condition,
        last_check_at=last_check_at,
        last_alert_at=last_alert_at,
        last_result=last_result,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestHeartbeatExecution:
    """Tests for HeartbeatLoop._check_monitor and _execute_check."""

    @pytest.fixture
    def store(self, tmp_path):
        """Fresh MonitorStore backed by a temp database."""
        from app.database import SafeDB
        db = SafeDB(str(tmp_path / "test.db"))
        db.init_schema()
        return MonitorStore(db)

    @pytest.fixture
    def loop(self, store):
        """HeartbeatLoop with no channel bots."""
        return HeartbeatLoop(store)

    # ------------------------------------------------------------------
    # 1. URL monitor — verify _execute_check returns fetched content
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_url_monitor(self, store, loop):
        """URL monitor should call http_fetch via the tool registry and return content."""
        # Create a URL monitor in the store
        mid = store.create(
            "URL Watcher",
            "url",
            {"url": "https://example.com"},
            schedule_seconds=300,
        )
        monitor = store.get(mid)

        # Mock the services + tool registry
        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(
            return_value="<html><body>Example Domain</body></html>"
        )

        mock_svc = MagicMock()
        mock_svc.tool_registry = mock_registry
        mock_svc.kg = None

        # get_services is imported locally inside _execute_check from app.core.brain
        with patch("app.core.brain.get_services", return_value=mock_svc):
            result = await loop._execute_check(monitor)

        # Verify http_fetch was called with the correct URL
        mock_registry.execute.assert_called_once_with(
            "http_fetch", {"url": "https://example.com"}
        )
        assert "Example Domain" in result

    # ------------------------------------------------------------------
    # 2. Error status derived from result content
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_status_derived(self, store, loop):
        """Monitor with notify_condition='on_error' should alert when status is derived as 'error'."""
        mid = store.create(
            "Error Watcher",
            "url",
            {"url": "https://broken.example.com"},
            schedule_seconds=300,
            notify_condition="on_error",
        )
        monitor = store.get(mid)

        # Mock _execute_check to return an error-prefixed string
        # (_check_monitor derives status from the returned content)
        with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
             patch.object(loop, "_send_alert", new_callable=AsyncMock) as mock_send:
            mock_exec.return_value = "error: connection refused"
            await loop._check_monitor(monitor)

        # on_error monitors should fire an alert when the content starts with "error"
        mock_send.assert_called_once()
        alert_msg = mock_send.call_args[0][1]
        assert "connection refused" in alert_msg

        # A result should have been recorded with the error content
        results = store.get_results(mid)
        assert len(results) >= 1
        latest = results[0]
        assert "error" in (latest.value or "").lower()

    # ------------------------------------------------------------------
    # 3. Reminder auto-disable — [Reminder] monitors disable after alert
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reminder_auto_disable(self, store, loop):
        """A monitor named '[Reminder] Test' should be disabled after its first alert fires."""
        mid = store.create(
            "[Reminder] Test",
            "url",
            {"url": "https://example.com"},
            schedule_seconds=60,
            cooldown_minutes=0,
            notify_condition="always",
        )
        monitor = store.get(mid)

        # Verify monitor is enabled before check
        assert monitor.enabled is True

        # Mock _execute_check to return valid content and _send_alert to no-op
        with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
             patch.object(loop, "_send_alert", new_callable=AsyncMock):
            mock_exec.return_value = "Reminder content here"
            await loop._check_monitor(monitor)

        # After the alert, the monitor should be disabled
        updated = store.get(mid)
        assert updated.enabled is False

    # ------------------------------------------------------------------
    # 4. Cooldown respected — recent alert skips next alert
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cooldown_respected(self, store, loop):
        """When last_alert_at is recent (within cooldown), the alert should be skipped."""
        # Set last_alert_at to 5 minutes ago with a 60-minute cooldown
        recent_time = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).replace(tzinfo=None).isoformat()

        mid = store.create(
            "Cooldown Test",
            "url",
            {"url": "https://example.com"},
            schedule_seconds=60,
            cooldown_minutes=60,
            notify_condition="always",
        )

        # Manually set last_alert_at to recent time
        store._db.execute(
            "UPDATE monitors SET last_alert_at = ? WHERE id = ?",
            (recent_time, mid),
        )
        monitor = store.get(mid)
        assert monitor.last_alert_at is not None

        # Mock _execute_check and track whether _send_alert is called
        with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
             patch.object(loop, "_send_alert", new_callable=AsyncMock) as mock_send:
            mock_exec.return_value = "Fresh content"
            await loop._check_monitor(monitor)

        # Alert should NOT have been sent (in cooldown)
        mock_send.assert_not_called()

        # A result should still be recorded, with "in cooldown" message
        results = store.get_results(mid)
        assert len(results) >= 1
        latest = results[0]
        assert "cooldown" in (latest.message or "").lower()


# ---------------------------------------------------------------------------
# Auto-finetune capability detection + safe-guard tests
# ---------------------------------------------------------------------------


class TestAutoFinetuneGuards:
    """Covers _can_auto_finetune() + _execute_finetune_check() fallback path.

    Production history (2026-05-07..12): six consecutive auto-fires failed
    silently because the container lacked finetune_auto.py / unsloth / CUDA.
    These tests pin the new behavior so the failure can't recur.
    """

    @pytest.fixture
    def loop(self, tmp_path):
        from app.database import SafeDB
        db = SafeDB(str(tmp_path / "test.db"))
        db.init_schema()
        return HeartbeatLoop(MonitorStore(db))

    def test_can_auto_finetune_blocks_when_script_missing(self, loop):
        # When neither absolute nor relative path exists, blocker should
        # identify the missing script — this is the production failure mode.
        with patch("pathlib.Path.exists", return_value=False):
            can, reason = loop._can_auto_finetune()
        assert can is False
        assert "finetune_auto" in reason

    def test_can_auto_finetune_blocks_when_unsloth_missing(self, loop):
        # Script present but unsloth not installed — read-only nova-app
        # container case. find_spec returns None for a missing module.
        with patch("pathlib.Path.exists", return_value=True), \
             patch("importlib.util.find_spec", return_value=None):
            can, reason = loop._can_auto_finetune()
        assert can is False
        assert "unsloth" in reason

    def test_can_auto_finetune_swallows_unexpected_exceptions(self, loop):
        # If Path.exists or importlib raises something unusual (e.g. odd
        # filesystem state), we must still return a (False, reason) tuple
        # so the heartbeat tick doesn't crash. This is a defense-in-depth
        # contract on the probe.
        with patch("pathlib.Path.exists", side_effect=OSError("disk gone")):
            can, reason = loop._can_auto_finetune()
        assert can is False
        assert "probe raised" in reason or "disk gone" in reason

    def test_can_auto_finetune_blocks_when_no_cuda(self, loop):
        # Script + unsloth present but CUDA not visible — GPU lives in
        # the nova-ollama sibling container, not nova-app.
        fake_spec = MagicMock()  # truthy → unsloth importable
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        import sys as _sys
        with patch("pathlib.Path.exists", return_value=True), \
             patch("importlib.util.find_spec", return_value=fake_spec), \
             patch.dict(_sys.modules, {"torch": fake_torch}):
            can, reason = loop._can_auto_finetune()
        assert can is False
        assert "CUDA" in reason

    @pytest.mark.asyncio
    async def test_finetune_check_notifies_when_auto_blocked(self, loop, tmp_path, monkeypatch):
        # Even when ENABLE_AUTO_FINETUNE=true and we have enough new pairs,
        # if the environment can't support training we degrade to NOTIFY
        # with a clear blocker reason — never silently fail to subprocess.
        data_path = tmp_path / "training_data.jsonl"
        # Write >= FINETUNE_MIN_NEW_PAIRS valid pairs.
        with open(data_path, "w", encoding="utf-8") as fh:
            for i in range(50):
                fh.write(json.dumps({"query": f"q{i}", "chosen": f"a{i}"}) + "\n")
        output_dir = tmp_path / "finetune"
        output_dir.mkdir()

        from app.config import config
        monkeypatch.setattr(config, "TRAINING_DATA_PATH", str(data_path))
        monkeypatch.setattr(config, "FINETUNE_OUTPUT_DIR", str(output_dir))
        monkeypatch.setattr(config, "FINETUNE_MIN_NEW_PAIRS", 15)
        monkeypatch.setattr(config, "ENABLE_AUTO_FINETUNE", True)

        # Force the capability probe to report a known blocker so we can
        # assert the message surfaces it instead of attempting Popen.
        with patch.object(loop, "_can_auto_finetune", return_value=(False, "unsloth not installed in this Python")), \
             patch("subprocess.Popen") as mock_popen:
            result = await loop._execute_finetune_check({})

        # Must not have tried to spawn a process — that's the whole point.
        mock_popen.assert_not_called()
        # Must surface FINETUNE READY + the specific blocker reason so the
        # operator sees it on the dashboard instead of "STARTED ... silently failing".
        assert result.startswith("FINETUNE READY")
        assert "blocked" in result.lower()
        assert "unsloth" in result

    @pytest.mark.asyncio
    async def test_finetune_check_notifies_when_disabled(self, loop, tmp_path, monkeypatch):
        # Plain "not enabled" path still works — opt-in operator gets the
        # standard READY message without a blocker prefix.
        data_path = tmp_path / "training_data.jsonl"
        with open(data_path, "w", encoding="utf-8") as fh:
            for i in range(50):
                fh.write(json.dumps({"query": f"q{i}", "chosen": f"a{i}"}) + "\n")
        output_dir = tmp_path / "finetune"
        output_dir.mkdir()

        from app.config import config
        monkeypatch.setattr(config, "TRAINING_DATA_PATH", str(data_path))
        monkeypatch.setattr(config, "FINETUNE_OUTPUT_DIR", str(output_dir))
        monkeypatch.setattr(config, "FINETUNE_MIN_NEW_PAIRS", 15)
        monkeypatch.setattr(config, "ENABLE_AUTO_FINETUNE", False)

        with patch.object(loop, "_can_auto_finetune", return_value=(True, "")), \
             patch("subprocess.Popen") as mock_popen:
            result = await loop._execute_finetune_check({})

        mock_popen.assert_not_called()
        assert result.startswith("FINETUNE READY")
        assert "ENABLE_AUTO_FINETUNE=true" in result

    @pytest.mark.asyncio
    async def test_finetune_check_surfaces_fast_failure(self, loop, tmp_path, monkeypatch):
        # When the subprocess DOES launch but crashes within the 3-second
        # probe window (script not found, import error), the monitor must
        # report LAUNCH FAILED rather than STARTED. This is the regression
        # guard against the silent-failure pattern that ran for 6 days.
        data_path = tmp_path / "training_data.jsonl"
        with open(data_path, "w", encoding="utf-8") as fh:
            for i in range(50):
                fh.write(json.dumps({"query": f"q{i}", "chosen": f"a{i}"}) + "\n")
        output_dir = tmp_path / "finetune"
        output_dir.mkdir()

        from app.config import config
        monkeypatch.setattr(config, "TRAINING_DATA_PATH", str(data_path))
        monkeypatch.setattr(config, "FINETUNE_OUTPUT_DIR", str(output_dir))
        monkeypatch.setattr(config, "FINETUNE_MIN_NEW_PAIRS", 15)
        monkeypatch.setattr(config, "ENABLE_AUTO_FINETUNE", True)

        # Simulate a process that already exited with non-zero code by the
        # time the probe runs. poll() must return an int (not None) so the
        # code path treats it as a crash.
        fake_proc = MagicMock()
        fake_proc.poll.return_value = 2

        # Pre-populate the log file so the tail read shows the error.
        # The Popen mock won't actually write the log, so we do it here.
        # Filename includes loop.time() so we match whatever the code wrote.
        import asyncio as _asyncio
        _t = int(_asyncio.get_event_loop().time())
        log_path = output_dir / f"auto_finetune_{_t}.log"
        log_path.write_text("python: can't open file 'scripts/finetune_auto.py'\n")

        with patch.object(loop, "_can_auto_finetune", return_value=(True, "")), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await loop._execute_finetune_check({})

        assert result.startswith("FINETUNE LAUNCH FAILED")
        assert "exit_code=2" in result
