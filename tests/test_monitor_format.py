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

    def test_stays_under_discord_limit(self):
        long_text = "x" * 3000
        result = format_monitor_output("Test Monitor", long_text)
        assert len(result) <= DISCORD_LIMIT

    def test_truncation_adds_note(self):
        long_text = "This is a long sentence. " * 200
        result = format_monitor_output("Test Monitor", long_text)
        assert "truncated" in result.lower()

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
