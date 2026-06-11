"""Tests for monitor output formatting (app/monitors/format.py).

Verifies Discord-safe formatting, truncation, and status classification.
"""

from __future__ import annotations

import pytest

from app.monitors.format import (
    DISCORD_LIMIT,
    classify_status,
    format_monitor_output,
    format_system_health,
)


class TestClassifyStatus:
    """Test automatic status classification from result text."""

    def test_error_keywords(self):
        assert classify_status("Connection error: timeout") == "error"
        assert classify_status("CRITICAL: disk full") == "error"
        assert classify_status("Service is down") == "error"

    def test_warning_keywords(self):
        assert classify_status("Memory usage elevated at 85%") == "warning"
        assert classify_status("Response time degraded") == "warning"
        assert classify_status("Latency is slow: 3000ms") == "warning"

    def test_ok_default(self):
        assert classify_status("All systems operational") == "ok"
        assert classify_status("Latency: 150ms") == "ok"
        assert classify_status("") == "ok"


class TestFormatMonitorOutput:
    """Test format_monitor_output() for Discord messages."""

    def test_basic_output(self):
        result = format_monitor_output("System Health", "All systems operational", status="ok")
        assert "\u2705" in result  # ✅
        assert "**System Health**" in result
        assert "All systems operational" in result

    def test_error_output(self):
        result = format_monitor_output("DB Monitor", "Connection failed", status="error")
        assert "\u274c" in result  # ❌
        assert "**DB Monitor**" in result

    def test_warning_output(self):
        result = format_monitor_output("Latency", "Response slow: 3s", status="warning")
        assert "\u26a0\ufe0f" in result  # ⚠️

    def test_auto_status_classification(self):
        result = format_monitor_output("Test", "Something failed badly")
        assert "\u274c" in result  # auto-classified as error

    def test_with_metrics(self):
        result = format_monitor_output(
            "DB Size",
            "Normal growth",
            status="ok",
            metrics={"DB size": "42.5 MB", "Rows": "1500"},
        )
        assert "**DB size:**" in result
        assert "42.5 MB" in result
        assert "\U0001f4ca" in result  # 📊

    def test_long_messages_pass_through_for_channel_split(self):
        # Truncation moved out of format_monitor_output 2026-04-26 — channel
        # adapters (Discord/Telegram) split for their per-platform limits.
        # The renderer should pass long messages through whole, not cut them
        # mid-sentence with a "[truncated]" note.
        long_text = "x" * 3000
        result = format_monitor_output("Test Monitor", long_text)
        # Body is preserved (no truncation marker injected by the renderer)
        assert "[truncated for Discord limit]" not in result
        assert long_text in result  # full body passes through

    def test_no_truncation_marker_in_long_output(self):
        long_text = "This is a long sentence. " * 200
        result = format_monitor_output("Test Monitor", long_text)
        # Renderer no longer injects a truncation marker — channel adapter
        # handles the split. The full body is present (modulo trailing
        # whitespace which the formatter strips).
        assert "[truncated" not in result.lower()
        assert long_text.rstrip() in result

    def test_timestamp_included(self):
        result = format_monitor_output("Test", "ok")
        assert "UTC" in result

    def test_empty_value(self):
        result = format_monitor_output("Test", "")
        assert "**Test**" in result
        assert len(result) <= DISCORD_LIMIT

    def test_short_message_no_truncation(self):
        result = format_monitor_output("Test", "Everything is fine")
        assert "truncated" not in result.lower()


class TestFormatSystemHealth:
    """Test format_system_health() helper."""

    def test_all_metrics(self):
        result = format_system_health(
            "System Health",
            db_size_mb=42.5,
            memory_pct=65.0,
            disk_pct=70.0,
            ollama_latency_ms=250.0,
            chromadb_docs=1500,
            skill_health="5/5 passing",
        )
        assert "42.5 MB" in result
        assert "65%" in result
        assert "70%" in result
        assert "250ms" in result
        assert "1500" in result
        assert "\u2705" in result  # ok status

    def test_high_memory_triggers_warning(self):
        result = format_system_health("Test", memory_pct=85.0)
        assert "\u26a0\ufe0f" in result  # warning

    def test_critical_memory_triggers_error(self):
        result = format_system_health("Test", memory_pct=95.0)
        assert "\u274c" in result  # error

    def test_high_latency_triggers_warning(self):
        result = format_system_health("Test", ollama_latency_ms=3000.0)
        assert "\u26a0\ufe0f" in result  # warning

    def test_stays_under_discord_limit(self):
        result = format_system_health(
            "System Health",
            db_size_mb=42.5,
            memory_pct=65.0,
            disk_pct=70.0,
            ollama_latency_ms=250.0,
            chromadb_docs=1500,
            extra_lines=["Line " + str(i) for i in range(100)],
        )
        assert len(result) <= DISCORD_LIMIT


class TestNewMonitorSeeds:
    """Test that new health monitors are seeded correctly."""

    @pytest.fixture
    def store(self, tmp_path):
        from app.database import SafeDB
        from app.monitors.heartbeat import MonitorStore
        db_path = str(tmp_path / "test.db")
        _db = SafeDB(db_path)
        conn = _db._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL, predicate TEXT NOT NULL,
                object TEXT NOT NULL, confidence REAL DEFAULT 0.8,
                source TEXT DEFAULT 'extracted', valid_from TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, predicate, object)
            )
        """)
        conn.commit()
        _db.init_schema()
        return MonitorStore(_db)

    def test_new_monitors_seeded(self, store):
        count = store.seed_defaults()
        monitors = store.list_all()
        names = {m.name for m in monitors}
        assert "DB Size Monitor" in names
        assert "Ollama Latency Monitor" in names
        assert "Skill Quality Monitor" in names
        assert "ChromaDB Integrity" in names
        assert "KG Health Monitor" in names

    def test_new_monitors_are_fast_type(self, store):
        store.seed_defaults()
        monitors = store.list_all()
        fast_types = {"db_size", "ollama_latency", "skill_quality", "chromadb_integrity", "kg_health"}
        health_monitors = [m for m in monitors if m.check_type in fast_types]
        assert len(health_monitors) == 5


# ===========================================================================
# Unified format_monitor_result() — one-liner contract
# ===========================================================================


class TestFormatMonitorResult:
    """Validate the unified one-line monitor result format."""

    def test_basic_shape(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result("X", "ok", "all healthy", {"docs": 110})
        assert out.startswith("\u2705 all healthy")  # ✅
        assert "\u2502" in out                       # │
        assert "docs: 110" in out

    def test_status_emojis(self):
        from app.monitors.format import format_monitor_result

        assert format_monitor_result("X", "ok", "good").startswith("\u2705")
        assert format_monitor_result("X", "warning", "wobbly").startswith("\u26a0")
        assert format_monitor_result("X", "error", "boom").startswith("\u274c")
        assert format_monitor_result("X", "skip", "nope").startswith("\U0001f4a4")
        assert format_monitor_result("X", "info", "data").startswith("\U0001f4ca")

    def test_unknown_status_falls_back(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result("X", "nonsense", "hi")
        assert out.startswith("\u2753")  # ❓

    def test_summary_truncation_80_chars(self):
        from app.monitors.format import format_monitor_result

        long = "a" * 200
        out = format_monitor_result("X", "ok", long)
        # Strip emoji + space, then check summary length
        summary = out.split(" ", 1)[1].split(" \u2502 ")[0]
        assert len(summary) <= 80

    def test_empty_summary_placeholder(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result("X", "ok", "")
        assert "no summary" in out

    def test_field_ordering_preserved(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result(
            "X", "ok", "ok", {"first": 1, "second": 2, "third": 3},
        )
        i1 = out.index("first: 1")
        i2 = out.index("second: 2")
        i3 = out.index("third: 3")
        assert i1 < i2 < i3

    def test_none_empty_fields_dropped(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result(
            "X", "ok", "ok", {"a": 1, "b": None, "c": ""},
        )
        assert "a: 1" in out
        assert "b:" not in out
        assert "c:" not in out

    def test_collapses_newlines_in_summary(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result("X", "ok", "line one\n\nline two\ttab")
        assert "\n" not in out
        assert "\t" not in out


class TestStripToolCallArtifacts:
    """Defensive scrubbing of leaked LLM tool-call syntax."""

    def test_strips_args_variant(self):
        from app.monitors.format import strip_tool_call_artifacts

        leaked = 'hello {"tool": "web_search", "args": {"query": "x"}}</tool_call> world'
        out = strip_tool_call_artifacts(leaked)
        assert "tool" not in out
        assert "</tool_call>" not in out
        assert "hello" in out and "world" in out

    def test_strips_arguments_variant(self):
        from app.monitors.format import strip_tool_call_artifacts

        leaked = '{"tool": "web_search", "arguments": {"query": "y"}}</tool_call>'
        out = strip_tool_call_artifacts(leaked)
        assert out == ""

    def test_strips_xml_wrapped(self):
        from app.monitors.format import strip_tool_call_artifacts

        leaked = 'prose <tool_call>{"tool":"x","args":{}}</tool_call> more prose'
        out = strip_tool_call_artifacts(leaked)
        assert "<tool_call>" not in out
        assert "</tool_call>" not in out
        assert "prose" in out

    def test_strips_orphan_close_tag(self):
        from app.monitors.format import strip_tool_call_artifacts

        out = strip_tool_call_artifacts("hi there </tool_call>")
        assert "</tool_call>" not in out
        assert "hi there" in out

    def test_noop_on_clean_text(self):
        from app.monitors.format import strip_tool_call_artifacts

        clean = "BTC at $67,500 — up 3.2% over 24h."
        assert strip_tool_call_artifacts(clean) == clean

    def test_empty_string(self):
        from app.monitors.format import strip_tool_call_artifacts

        assert strip_tool_call_artifacts("") == ""

    def test_collapses_excess_blank_lines(self):
        from app.monitors.format import strip_tool_call_artifacts

        out = strip_tool_call_artifacts("line1\n\n\n\n\nline2")
        assert out == "line1\n\nline2"


class TestDreamConsolidationSkipFormat:
    """Dream Consolidation skip message should be the concise unified line."""

    def test_skip_format_is_short(self):
        from app.monitors.format import format_monitor_result

        out = format_monitor_result(
            "Dream Consolidation", "skip", "cooldown",
            {"cooldown": "0.5h/6.0h"},
        )
        # Compact: under 80 chars, emoji + summary + fields separator
        assert len(out) < 100
        assert "\U0001f4a4" in out       # 💤
        assert "0.5h/6.0h" in out
        # Not the old 60-word paragraph
        assert "minimum" not in out.lower()
        assert "skipped because" not in out.lower()


class TestMonitorRouting:
    """System-category monitors must never reach Discord."""

    @pytest.fixture
    def store(self, tmp_path):
        from app.database import SafeDB
        from app.monitors.heartbeat import MonitorStore
        db = SafeDB(str(tmp_path / "test.db"))
        db.init_schema()
        return MonitorStore(db)

    @pytest.mark.asyncio
    async def test_system_monitor_skips_discord(self, store):
        """A system-category monitor result should NOT reach the Discord bot."""
        from unittest.mock import AsyncMock
        from app.monitors.heartbeat import HeartbeatLoop
        from app.monitors.monitor_store import Monitor

        mid = store.create("DB Size Monitor", "db_size", {})
        m = store.get(mid)
        assert m.category == "system"

        discord_bot = AsyncMock()
        telegram_bot = AsyncMock()
        whatsapp_bot = AsyncMock()
        signal_bot = AsyncMock()

        loop = HeartbeatLoop(
            store,
            discord_bot=discord_bot,
            telegram_bot=telegram_bot,
            whatsapp_bot=whatsapp_bot,
            signal_bot=signal_bot,
        )
        await loop._send_alert(m, "db healthy")

        discord_bot.send_alert.assert_not_called()
        whatsapp_bot.send_alert.assert_not_called()
        signal_bot.send_alert.assert_not_called()
        telegram_bot.send_alert.assert_called_once()
        # Must carry the name prefix so Telegram users still see which monitor
        sent_msg = telegram_bot.send_alert.call_args[0][0]
        assert sent_msg.startswith("[DB Size Monitor]")

    @pytest.mark.asyncio
    async def test_content_monitor_hits_all_channels(self, store):
        """A content-category monitor broadcasts to all configured channels."""
        from unittest.mock import AsyncMock
        from app.monitors.heartbeat import HeartbeatLoop

        mid = store.create("World Awareness", "query", {"query": "news"})
        m = store.get(mid)
        assert m.category == "content"

        discord_bot = AsyncMock()
        telegram_bot = AsyncMock()
        whatsapp_bot = AsyncMock()
        signal_bot = AsyncMock()

        loop = HeartbeatLoop(
            store,
            discord_bot=discord_bot,
            telegram_bot=telegram_bot,
            whatsapp_bot=whatsapp_bot,
            signal_bot=signal_bot,
        )
        await loop._send_alert(m, "major news")

        discord_bot.send_alert.assert_called_once()
        telegram_bot.send_alert.assert_called_once()
        whatsapp_bot.send_alert.assert_called_once()
        signal_bot.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_monitor_without_telegram_is_suppressed(self, store):
        """System monitor with no Telegram channel should not fall through to Discord."""
        from unittest.mock import AsyncMock
        from app.monitors.heartbeat import HeartbeatLoop

        mid = store.create("DB Size Monitor", "db_size", {})
        m = store.get(mid)

        discord_bot = AsyncMock()
        loop = HeartbeatLoop(store, discord_bot=discord_bot, telegram_bot=None)
        await loop._send_alert(m, "stats")

        discord_bot.send_alert.assert_not_called()
