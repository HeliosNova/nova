"""Tests for the heartbeat/monitor system."""

from __future__ import annotations

import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from app.monitors.heartbeat import (
    MonitorStore,
    HeartbeatLoop,
    detect_change,
    extract_numbers,
)
from app.tools.monitor_tool import MonitorTool


# ===========================================================================
# Change Detection
# ===========================================================================


class TestExtractNumbers:
    def test_simple_integer(self):
        assert extract_numbers("The price is 42") == [42.0]

    def test_decimal(self):
        assert extract_numbers("BTC at 67543.21") == [67543.21]

    def test_with_dollar_sign(self):
        nums = extract_numbers("$1,234.56 per unit")
        assert nums == [1234.56]

    def test_with_suffix_k(self):
        nums = extract_numbers("Revenue: 45K")
        assert nums == [45000.0]

    def test_with_suffix_m(self):
        nums = extract_numbers("Market cap: 2.5M")
        assert nums == [2500000.0]

    def test_with_suffix_b(self):
        nums = extract_numbers("$1.2B valuation")
        assert nums == [1200000000.0]

    def test_percentage(self):
        nums = extract_numbers("Up 15.3%")
        assert nums == [15.3]

    def test_multiple_numbers(self):
        nums = extract_numbers("From $100 to $150")
        assert len(nums) == 2
        assert nums[0] == 100.0
        assert nums[1] == 150.0

    def test_no_numbers(self):
        assert extract_numbers("Hello world") == []

    def test_commas_in_number(self):
        nums = extract_numbers("Population: 1,234,567")
        assert nums == [1234567.0]


class TestDetectChange:
    def test_numeric_increase(self):
        result = detect_change("$100", "$120", threshold_pct=5.0)
        assert result is not None
        assert result["type"] == "numeric"
        assert result["direction"] == "up"
        assert result["pct_change"] == 20.0

    def test_numeric_decrease(self):
        result = detect_change("$100", "$80", threshold_pct=5.0)
        assert result is not None
        assert result["direction"] == "down"
        assert result["pct_change"] == 20.0

    def test_numeric_below_threshold(self):
        result = detect_change("$100", "$102", threshold_pct=5.0)
        assert result is None

    def test_text_change(self):
        result = detect_change("Hello world", "Hello universe")
        assert result is not None
        assert result["type"] == "text"
        assert result["changed"] is True

    def test_no_change(self):
        result = detect_change("Same content", "Same content")
        assert result is None

    def test_empty_values(self):
        assert detect_change("", "something") is None
        assert detect_change("something", "") is None

    def test_numeric_exact_threshold(self):
        result = detect_change("$100", "$105", threshold_pct=5.0)
        assert result is not None  # 5% is >= threshold


# ===========================================================================
# MonitorStore (requires SQLite)
# ===========================================================================


class TestMonitorStore:
    @pytest.fixture
    def store(self, tmp_path):
        from app.database import SafeDB
        db = SafeDB(str(tmp_path / "test.db"))
        db.init_schema()
        return MonitorStore(db)

    def test_create_and_get(self, store):
        mid = store.create("Test Monitor", "search", {"query": "test"})
        assert mid > 0
        m = store.get(mid)
        assert m is not None
        assert m.name == "Test Monitor"
        assert m.check_type == "search"
        assert m.check_config == {"query": "test"}
        assert m.enabled is True

    def test_create_duplicate_name(self, store):
        store.create("Unique", "search", {"query": "a"})
        mid2 = store.create("Unique", "search", {"query": "b"})
        assert mid2 == -1

    def test_list_all(self, store):
        store.create("A", "url", {"url": "https://a.com"})
        store.create("B", "search", {"query": "b"})
        monitors = store.list_all()
        assert len(monitors) == 2

    def test_get_by_name(self, store):
        store.create("Named", "command", {"command": "ls"})
        m = store.get_by_name("Named")
        assert m is not None
        assert m.check_type == "command"

    def test_update(self, store):
        mid = store.create("Updatable", "search", {"query": "old"})
        store.update(mid, enabled=False)
        m = store.get(mid)
        assert m.enabled is False

    def test_delete(self, store):
        mid = store.create("Deletable", "search", {"query": "bye"})
        store.delete(mid)
        assert store.get(mid) is None

    def test_get_due_fresh_monitor(self, store):
        """A freshly created monitor with no last_check_at should be due."""
        store.create("Due", "search", {"query": "test"}, schedule_seconds=60)
        due = store.get_due()
        assert len(due) == 1
        assert due[0].name == "Due"

    def test_get_due_recently_checked(self, store):
        """A recently checked monitor should not be due."""
        from datetime import datetime
        mid = store.create("Recent", "search", {"query": "test"}, schedule_seconds=3600)
        store.record_check(mid, "ok")
        due = store.get_due()
        assert len(due) == 0

    def test_record_check_and_result(self, store):
        mid = store.create("Checked", "url", {"url": "https://example.com"})
        store.record_check(mid, "200 OK")
        store.add_result(mid, "ok", value="200 OK", message="Success")
        results = store.get_results(mid)
        assert len(results) == 1
        assert results[0].status == "ok"
        assert results[0].value == "200 OK"

    def test_seed_defaults(self, store):
        count = store.seed_defaults()
        assert count == 14
        monitors = store.list_all()
        names = {m.name for m in monitors}
        assert "Morning Check-in" in names
        assert "System Health" in names
        assert "World Awareness" in names
        assert "Self-Reflection" in names
        assert "Domain Study: Science" in names
        assert "Domain Study: Technology" in names
        assert "Domain Study: Current Events" in names
        assert "Domain Study: Finance" in names
        assert "Fine-Tune Check" in names
        assert "Lesson Quiz" in names
        assert "Skill Validation" in names

    def test_quiz_and_skill_seeded_on_change(self, store):
        """Quiz and Skill Validation should seed with notify_condition='on_change'."""
        store.seed_defaults()
        quiz = store.get_by_name("Lesson Quiz")
        skill = store.get_by_name("Skill Validation")
        assert quiz is not None
        assert skill is not None
        assert quiz.notify_condition == "on_change"
        assert skill.notify_condition == "on_change"

    def test_seed_defaults_idempotent(self, store):
        store.seed_defaults()
        count2 = store.seed_defaults()
        assert count2 == 0  # Already seeded

    def test_get_recent_results(self, store):
        mid = store.create("Recents", "search", {"query": "test"})
        store.add_result(mid, "ok", value="a")
        store.add_result(mid, "changed", value="b", message="changed!")
        results = store.get_recent_results(hours=1)
        assert len(results) == 2


# ===========================================================================
# MonitorTool
# ===========================================================================


class TestMonitorTool:
    @pytest.fixture
    def tool_with_store(self, tmp_path):
        from app.database import SafeDB
        db = SafeDB(str(tmp_path / "test.db"))
        db.init_schema()
        store = MonitorStore(db)
        return MonitorTool(monitor_store=store), store

    @pytest.mark.asyncio
    async def test_no_action(self):
        tool = MonitorTool()
        result = await tool.execute()
        assert not result.success
        assert "action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_store(self):
        tool = MonitorTool()
        result = await tool.execute(action="list")
        assert not result.success
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_monitor(self, tool_with_store):
        tool, store = tool_with_store
        result = await tool.execute(
            action="create",
            name="BTC Price",
            check_type="search",
            config={"query": "bitcoin price"},
            schedule_minutes=15,
        )
        assert result.success
        assert "BTC Price" in result.output
        assert store.get_by_name("BTC Price") is not None

    @pytest.mark.asyncio
    async def test_create_no_name(self, tool_with_store):
        tool, _ = tool_with_store
        result = await tool.execute(action="create", name="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_empty(self, tool_with_store):
        tool, _ = tool_with_store
        result = await tool.execute(action="list")
        assert result.success
        assert "No monitors" in result.output

    @pytest.mark.asyncio
    async def test_list_with_monitors(self, tool_with_store):
        tool, store = tool_with_store
        store.create("Test1", "search", {"query": "a"})
        store.create("Test2", "url", {"url": "https://b.com"})
        result = await tool.execute(action="list")
        assert result.success
        assert "Test1" in result.output
        assert "Test2" in result.output

    @pytest.mark.asyncio
    async def test_delete_by_name(self, tool_with_store):
        tool, store = tool_with_store
        store.create("ToDelete", "search", {"query": "bye"})
        result = await tool.execute(action="delete", name="ToDelete")
        assert result.success
        assert store.get_by_name("ToDelete") is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, tool_with_store):
        tool, _ = tool_with_store
        result = await tool.execute(action="delete", name="Nope")
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool_with_store):
        tool, _ = tool_with_store
        result = await tool.execute(action="fly")
        assert not result.success

    @pytest.mark.asyncio
    async def test_invalid_check_type(self, tool_with_store):
        tool, _ = tool_with_store
        result = await tool.execute(action="create", name="Bad", check_type="invalid")
        assert not result.success

    @pytest.mark.asyncio
    async def test_config_as_string(self, tool_with_store):
        tool, store = tool_with_store
        result = await tool.execute(
            action="create",
            name="StringConfig",
            check_type="search",
            check_config='{"query": "test"}',
            schedule_minutes=10,
        )
        assert result.success
        m = store.get_by_name("StringConfig")
        assert m.check_config == {"query": "test"}
