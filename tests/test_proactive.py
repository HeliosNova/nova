"""Tests for Phase 8.2: DailyDigest proactive engine."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitors.proactive import DailyDigest


# ===========================================================================
# Helpers
# ===========================================================================

def _make_monitor(id: int, name: str, check_type: str = "url") -> MagicMock:
    """Create a mock Monitor dataclass."""
    m = MagicMock()
    m.id = id
    m.name = name
    m.check_type = check_type
    return m


def _make_result(
    id: int,
    monitor_id: int,
    status: str = "ok",
    value: str | None = None,
    message: str | None = None,
) -> MagicMock:
    """Create a mock MonitorResult dataclass."""
    r = MagicMock()
    r.id = id
    r.monitor_id = monitor_id
    r.status = status
    r.value = value
    r.message = message
    return r


def _make_monitor_store(
    monitors: list | None = None,
    results: list | None = None,
) -> MagicMock:
    """Create a mock MonitorStore."""
    store = MagicMock()
    store.list_all.return_value = monitors or []
    store.get_recent_results.return_value = results or []
    return store


# ===========================================================================
# Instantiation
# ===========================================================================

class TestDailyDigestInit:
    def test_creates_with_store_only(self):
        """DailyDigest can be created with just a monitor store."""
        store = _make_monitor_store()
        digest = DailyDigest(store)
        assert digest._store is store
        assert digest._discord is None
        assert digest._telegram is None
        assert digest._whatsapp is None
        assert digest._signal is None
        assert digest._learning is None
        assert digest._last_digest is None
        assert digest._running is False

    def test_creates_with_all_channels(self):
        """DailyDigest accepts all optional channel bots."""
        store = _make_monitor_store()
        discord = MagicMock()
        telegram = MagicMock()
        whatsapp = MagicMock()
        signal = MagicMock()
        learning = MagicMock()

        digest = DailyDigest(
            store,
            discord_bot=discord,
            telegram_bot=telegram,
            whatsapp_bot=whatsapp,
            signal_bot=signal,
            learning_engine=learning,
        )
        assert digest._discord is discord
        assert digest._telegram is telegram
        assert digest._whatsapp is whatsapp
        assert digest._signal is signal
        assert digest._learning is learning

    def test_stop_cancels_task(self):
        """stop() should cancel the background task."""
        store = _make_monitor_store()
        digest = DailyDigest(store)

        mock_task = MagicMock()
        digest._task = mock_task
        digest._running = True

        digest.stop()
        assert digest._running is False
        mock_task.cancel.assert_called_once()

    def test_stop_without_task(self):
        """stop() should not crash when no task is running."""
        store = _make_monitor_store()
        digest = DailyDigest(store)

        # Should not raise
        digest.stop()
        assert digest._running is False


# ===========================================================================
# Generation with mock data
# ===========================================================================

class TestDailyDigestGeneration:
    @pytest.mark.asyncio
    async def test_generates_digest_with_results(self):
        """send_digest() should generate a digest when results exist."""
        monitors = [_make_monitor(1, "API Health"), _make_monitor(2, "DB Check")]
        results = [
            _make_result(1, 1, status="ok", message="API responding normally"),
            _make_result(2, 2, status="alert", message="DB latency high"),
        ]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(
                return_value="**Monitors:** 1 passed, 1 alert\n**Issues:** DB latency high"
            )

            result = await digest.send_digest()

        assert result is not None
        assert "Evening Digest" in result
        assert "Monitors" in result
        mock_llm.invoke_nothink.assert_called_once()

    @pytest.mark.asyncio
    async def test_digest_includes_monitor_names(self):
        """The LLM prompt should include monitor names and results."""
        monitors = [_make_monitor(1, "Web Server")]
        results = [_make_result(1, 1, status="ok", message="200 OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        captured_messages = []
        with patch("app.monitors.proactive.llm") as mock_llm:
            async def capture_invoke(messages, **kwargs):
                captured_messages.extend(messages)
                return "**Monitors:** 1 passed"

            mock_llm.invoke_nothink = AsyncMock(side_effect=capture_invoke)

            await digest.send_digest()

        # The user content should mention the monitor name
        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "Web Server" in user_msg["content"]
        assert "200 OK" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_digest_caps_at_30_results(self):
        """Digest should cap at 30 results even if more exist."""
        monitors = [_make_monitor(i, f"Monitor {i}") for i in range(50)]
        results = [_make_result(i, i, status="ok", message=f"Result {i}") for i in range(50)]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        captured_messages = []
        with patch("app.monitors.proactive.llm") as mock_llm:
            async def capture_invoke(messages, **kwargs):
                captured_messages.extend(messages)
                return "**Monitors:** 30 checked"

            mock_llm.invoke_nothink = AsyncMock(side_effect=capture_invoke)

            await digest.send_digest()

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        # Should only have 30 monitor lines (each starts with "- [")
        monitor_lines = [l for l in user_msg["content"].split("\n") if l.startswith("- [")]
        assert len(monitor_lines) == 30

    @pytest.mark.asyncio
    async def test_fallback_when_llm_returns_empty(self):
        """If LLM returns empty, use raw results as fallback."""
        monitors = [_make_monitor(1, "Health Check")]
        results = [_make_result(1, 1, status="ok", message="All good")]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="")

            result = await digest.send_digest()

        assert result is not None
        assert "Evening Digest" in result
        assert "1 monitor result(s)" in result

    @pytest.mark.asyncio
    async def test_fallback_when_llm_raises(self):
        """If LLM raises, use raw results as fallback."""
        monitors = [_make_monitor(1, "Health Check")]
        results = [_make_result(1, 1, status="ok", message="All good")]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=RuntimeError("LLM down"))

            result = await digest.send_digest()

        assert result is not None
        assert "Evening Digest" in result
        assert "1 monitor result(s)" in result

    @pytest.mark.asyncio
    async def test_digest_includes_learning_section(self):
        """When learning_engine is provided, digest includes learning summary."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        learning = MagicMock()
        learning.get_learning_summary.return_value = {
            "new_lessons": [{"topic": "Python", "lesson": "Use list comprehensions"}],
            "new_skills": [],
            "new_tools": [],
            "new_reflexions": [],
            "new_training_pairs": 5,
            "degraded_skills": [],
            "totals": {"total_lessons": 10, "total_skills": 3, "training_examples": 50},
        }

        digest = DailyDigest(store, learning_engine=learning)

        captured_messages = []
        with patch("app.monitors.proactive.llm") as mock_llm:
            async def capture_invoke(messages, **kwargs):
                captured_messages.extend(messages)
                return "**Monitors:** 1 passed\n**Learning:** 1 new lesson"

            mock_llm.invoke_nothink = AsyncMock(side_effect=capture_invoke)

            await digest.send_digest()

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "Learning Activity" in user_msg["content"]
        assert "Python" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_digest_includes_teaching_section(self):
        """When quiz/skill_test results exist, digest includes teaching section."""
        monitors = [
            _make_monitor(1, "Lesson Quiz", check_type="quiz"),
            _make_monitor(2, "Skill Test", check_type="skill_test"),
        ]
        results = [
            _make_result(1, 1, status="ok", value="QUIZ PASSED: score 90%"),
            _make_result(2, 1, status="ok", value="QUIZ FAILED: score 40%"),
            _make_result(3, 2, status="ok", value="SKILL TEST PASSED"),
        ]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        captured_messages = []
        with patch("app.monitors.proactive.llm") as mock_llm:
            async def capture_invoke(messages, **kwargs):
                captured_messages.extend(messages)
                return "**Teaching:** 1/2 quizzes passed"

            mock_llm.invoke_nothink = AsyncMock(side_effect=capture_invoke)

            await digest.send_digest()

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "Teaching Activity" in user_msg["content"]
        assert "Lesson quizzes" in user_msg["content"]


# ===========================================================================
# Empty digest handling
# ===========================================================================

class TestEmptyDigest:
    @pytest.mark.asyncio
    async def test_no_results_returns_none(self):
        """send_digest() returns None when no monitor results exist."""
        store = _make_monitor_store(monitors=[], results=[])
        digest = DailyDigest(store)

        with patch("app.monitors.proactive.llm") as mock_llm:
            result = await digest.send_digest()

        assert result is None
        mock_llm.invoke_nothink.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_results_list_returns_none(self):
        """send_digest() returns None when results list is explicitly empty."""
        store = MagicMock()
        store.get_recent_results.return_value = []
        digest = DailyDigest(store)

        result = await digest.send_digest()
        assert result is None


# ===========================================================================
# Channel delivery
# ===========================================================================

class TestChannelDelivery:
    @pytest.mark.asyncio
    async def test_sends_to_discord(self):
        """Digest should be sent to Discord when discord_bot is provided."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        discord = MagicMock()
        discord.send_alert = AsyncMock()

        digest = DailyDigest(store, discord_bot=discord)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            await digest.send_digest()

        discord.send_alert.assert_called_once()
        sent_msg = discord.send_alert.call_args[0][0]
        assert "Evening Digest" in sent_msg

    @pytest.mark.asyncio
    async def test_sends_to_telegram(self):
        """Digest should be sent to Telegram when telegram_bot is provided."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        telegram = MagicMock()
        telegram.send_alert = AsyncMock()

        digest = DailyDigest(store, telegram_bot=telegram)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            await digest.send_digest()

        telegram.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_to_whatsapp(self):
        """Digest should be sent to WhatsApp when whatsapp_bot is provided."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        whatsapp = MagicMock()
        whatsapp.send_alert = AsyncMock()

        digest = DailyDigest(store, whatsapp_bot=whatsapp)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            await digest.send_digest()

        whatsapp.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_to_signal(self):
        """Digest should be sent to Signal when signal_bot is provided."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        signal = MagicMock()
        signal.send_alert = AsyncMock()

        digest = DailyDigest(store, signal_bot=signal)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            await digest.send_digest()

        signal.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_to_all_channels(self):
        """Digest should be sent to ALL configured channels."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        discord = MagicMock()
        discord.send_alert = AsyncMock()
        telegram = MagicMock()
        telegram.send_alert = AsyncMock()
        whatsapp = MagicMock()
        whatsapp.send_alert = AsyncMock()
        signal = MagicMock()
        signal.send_alert = AsyncMock()

        digest = DailyDigest(
            store,
            discord_bot=discord,
            telegram_bot=telegram,
            whatsapp_bot=whatsapp,
            signal_bot=signal,
        )

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            await digest.send_digest()

        discord.send_alert.assert_called_once()
        telegram.send_alert.assert_called_once()
        whatsapp.send_alert.assert_called_once()
        signal.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_channel_failure_does_not_crash(self):
        """If one channel fails, others should still receive the digest."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        discord = MagicMock()
        discord.send_alert = AsyncMock(side_effect=RuntimeError("Discord down"))
        telegram = MagicMock()
        telegram.send_alert = AsyncMock()

        digest = DailyDigest(store, discord_bot=discord, telegram_bot=telegram)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            # Should not raise despite Discord failure
            result = await digest.send_digest()

        assert result is not None
        discord.send_alert.assert_called_once()
        telegram.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_channels_still_returns_digest(self):
        """send_digest() returns the digest text even with no channels configured."""
        monitors = [_make_monitor(1, "Health")]
        results = [_make_result(1, 1, status="ok", message="OK")]
        store = _make_monitor_store(monitors=monitors, results=results)

        digest = DailyDigest(store)

        with patch("app.monitors.proactive.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="**Monitors:** 1 passed")

            result = await digest.send_digest()

        assert result is not None
        assert "Evening Digest" in result
