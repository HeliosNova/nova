"""Tests for Feature 1: Autonomous tool-creation trigger (tool_triggers.py).

Covers:
- ToolCandidateStore: record, count, mark_triggered, is_already_triggered
- Pattern detection fires at threshold, not below
- Blocked-tool guard (tool_create in sequence → skip)
- Skill-coverage guard (existing skill → skip)
- Already-triggered guard (no double-fire)
- Generated tool code persists and is loadable
- Disabled flag prevents any action
- LLM skip response is handled gracefully
- Generated code containing blocked references is rejected
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.tool_triggers import (
    ToolCandidateStore,
    _BLOCKED_TOOLS,
    _generate_tool_spec,
    maybe_trigger_tool_creation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(db):
    return ToolCandidateStore(db=db)


def _make_tool_results(tools: list[str]) -> list[dict]:
    return [{"tool": t, "args": {}, "output": f"output of {t}"} for t in tools]


def _make_svc(db, *, has_custom_tools: bool = True, skill_match=None):
    """Build a minimal mock Services object."""
    svc = MagicMock()
    if has_custom_tools:
        from app.core.custom_tools import CustomToolStore
        svc.custom_tools = CustomToolStore(db)
    else:
        svc.custom_tools = None

    if skill_match is not None:
        svc.skills.get_matching_skill.return_value = skill_match
    else:
        svc.skills = None

    svc.tool_registry = None
    return svc


# ---------------------------------------------------------------------------
# ToolCandidateStore unit tests
# ---------------------------------------------------------------------------

class TestToolCandidateStore:
    def test_sequence_key_canonical(self):
        """Sorted, deduplicated tool names → canonical key."""
        key1 = ToolCandidateStore._sequence_key(["web_search", "calculator"])
        key2 = ToolCandidateStore._sequence_key(["calculator", "web_search"])
        assert key1 == key2
        assert "|" in key1

    def test_record_returns_id(self, db):
        store = _make_store(db)
        rid = store.record("test query", ["web_search", "calculator"])
        assert isinstance(rid, int) and rid > 0

    def test_count_untriggered_starts_at_zero(self, db):
        store = _make_store(db)
        count = store.count_untriggered(["web_search", "calculator"])
        assert count == 0

    def test_count_increments_on_record(self, db):
        store = _make_store(db)
        store.record("query A", ["web_search", "calculator"])
        assert store.count_untriggered(["web_search", "calculator"]) == 1
        store.record("query B", ["web_search", "calculator"])
        assert store.count_untriggered(["web_search", "calculator"]) == 2

    def test_count_is_sequence_specific(self, db):
        store = _make_store(db)
        store.record("query A", ["web_search", "calculator"])
        # Different sequence → different count
        assert store.count_untriggered(["web_search", "code_exec"]) == 0

    def test_mark_triggered_resets_count(self, db):
        store = _make_store(db)
        store.record("query A", ["web_search", "calculator"])
        store.record("query B", ["web_search", "calculator"])
        store.mark_triggered(["web_search", "calculator"])
        # Triggered entries don't count
        assert store.count_untriggered(["web_search", "calculator"]) == 0

    def test_is_already_triggered_false_initially(self, db):
        store = _make_store(db)
        store.record("query A", ["web_search", "calculator"])
        assert store.is_already_triggered(["web_search", "calculator"]) is False

    def test_is_already_triggered_true_after_mark(self, db):
        store = _make_store(db)
        store.record("query A", ["web_search", "calculator"])
        store.mark_triggered(["web_search", "calculator"])
        assert store.is_already_triggered(["web_search", "calculator"]) is True

    def test_get_example_queries(self, db):
        store = _make_store(db)
        store.record("find price of gold", ["web_search", "calculator"])
        store.record("what is AAPL stock price", ["web_search", "calculator"])
        examples = store.get_example_queries(["web_search", "calculator"])
        assert len(examples) == 2
        assert any("gold" in q for q in examples)


# ---------------------------------------------------------------------------
# maybe_trigger_tool_creation — guard tests
# ---------------------------------------------------------------------------

class TestTriggerGuards:
    @pytest.mark.asyncio
    async def test_disabled_flag_exits_immediately(self, db, monkeypatch):
        """ENABLE_AUTONOMOUS_TOOL_CREATION=false → nothing happens."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "false")
        from app.config import reset_config
        reset_config()

        svc = _make_svc(db)
        # Should not raise and should not record anything
        with patch("app.core.tool_triggers.ToolCandidateStore") as mock_store_cls:
            await maybe_trigger_tool_creation("query", _make_tool_results(["web_search", "calculator", "code_exec"]), svc)
        mock_store_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_custom_tools_exits(self, db, monkeypatch):
        """No custom_tools on svc → exits without recording."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        svc = _make_svc(db, has_custom_tools=False)
        store = _make_store(db)
        await maybe_trigger_tool_creation(
            "query", _make_tool_results(["web_search", "calculator", "code_exec"]),
            svc, _candidate_store=store,
        )
        assert store.count_untriggered(["web_search", "calculator", "code_exec"]) == 0

    @pytest.mark.asyncio
    async def test_blocked_tool_in_sequence_exits(self, db, monkeypatch):
        """tool_create in the sequence → skip (infinite loop guard)."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        svc = _make_svc(db)
        tools = ["web_search", "tool_create", "calculator"]
        store = _make_store(db)
        await maybe_trigger_tool_creation("query", _make_tool_results(tools), svc, _candidate_store=store)
        assert store.count_untriggered(tools) == 0

    @pytest.mark.asyncio
    async def test_existing_skill_coverage_exits(self, db, monkeypatch):
        """If a skill already covers the query, don't record a candidate."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        from app.config import reset_config
        reset_config()

        mock_skill = MagicMock()
        mock_skill.name = "existing_skill"
        svc = _make_svc(db, skill_match=mock_skill)
        tools = ["web_search", "calculator", "code_exec"]
        store = _make_store(db)

        with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_skill)):
            await maybe_trigger_tool_creation(
                "query", _make_tool_results(tools), svc, _candidate_store=store,
            )

        assert store.count_untriggered(tools) == 0

    @pytest.mark.asyncio
    async def test_already_triggered_exits(self, db, monkeypatch):
        """If this sequence was already triggered, skip recording again."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        monkeypatch.setenv("AUTO_TOOL_CREATION_THRESHOLD", "2")
        from app.config import reset_config
        reset_config()

        tools = ["web_search", "calculator", "code_exec"]
        svc = _make_svc(db)
        svc.skills = None

        # Pre-mark as triggered
        store = _make_store(db)
        store.record("prior query", tools)
        store.mark_triggered(tools)

        # Should exit without doing anything
        with patch("app.core.tool_triggers._generate_tool_spec") as mock_gen:
            await maybe_trigger_tool_creation(
                "new query", _make_tool_results(tools), svc, _candidate_store=store,
            )
        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Threshold detection
# ---------------------------------------------------------------------------

class TestThresholdDetection:
    @pytest.mark.asyncio
    async def test_below_threshold_no_generation(self, db, monkeypatch):
        """Below threshold → candidate recorded, no LLM call."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        monkeypatch.setenv("AUTO_TOOL_CREATION_THRESHOLD", "3")
        from app.config import reset_config
        reset_config()

        tools = ["web_search", "calculator", "code_exec"]
        svc = _make_svc(db)
        svc.skills = None
        store = _make_store(db)

        with patch("app.core.tool_triggers._generate_tool_spec") as mock_gen:
            await maybe_trigger_tool_creation("q1", _make_tool_results(tools), svc, _candidate_store=store)
            await maybe_trigger_tool_creation("q2", _make_tool_results(tools), svc, _candidate_store=store)

        mock_gen.assert_not_called()
        assert store.count_untriggered(tools) == 2

    @pytest.mark.asyncio
    async def test_at_threshold_triggers_generation(self, db, monkeypatch):
        """At threshold → asyncio.create_task is called for LLM generation."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        monkeypatch.setenv("AUTO_TOOL_CREATION_THRESHOLD", "3")
        from app.config import reset_config
        reset_config()

        tools = ["web_search", "calculator", "code_exec"]
        svc = _make_svc(db)
        svc.skills = None
        store = _make_store(db)

        with patch("asyncio.create_task") as mock_task:
            await maybe_trigger_tool_creation("q1", _make_tool_results(tools), svc, _candidate_store=store)
            await maybe_trigger_tool_creation("q2", _make_tool_results(tools), svc, _candidate_store=store)
            # Third call hits threshold
            await maybe_trigger_tool_creation("q3", _make_tool_results(tools), svc, _candidate_store=store)

        assert mock_task.called

    @pytest.mark.asyncio
    async def test_no_double_fire_after_threshold(self, db, monkeypatch):
        """Threshold fires once — subsequent calls don't trigger again."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        monkeypatch.setenv("AUTO_TOOL_CREATION_THRESHOLD", "2")
        from app.config import reset_config
        reset_config()

        tools = ["web_search", "calculator", "code_exec"]
        svc = _make_svc(db)
        svc.skills = None
        store = _make_store(db)

        task_calls = []

        with patch("asyncio.create_task", side_effect=lambda c: task_calls.append(c)):
            await maybe_trigger_tool_creation("q1", _make_tool_results(tools), svc, _candidate_store=store)
            await maybe_trigger_tool_creation("q2", _make_tool_results(tools), svc, _candidate_store=store)  # fires
            await maybe_trigger_tool_creation("q3", _make_tool_results(tools), svc, _candidate_store=store)  # skipped
            await maybe_trigger_tool_creation("q4", _make_tool_results(tools), svc, _candidate_store=store)  # skipped

        # create_task called exactly once
        assert len(task_calls) == 1


# ---------------------------------------------------------------------------
# Generated tool code validation
# ---------------------------------------------------------------------------

class TestGeneratedToolPersistence:
    @pytest.mark.asyncio
    async def test_generated_tool_persists_in_custom_tool_store(self, db, monkeypatch):
        """When LLM returns valid spec, tool is created in CustomToolStore."""
        monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "true")
        monkeypatch.setenv("AUTO_TOOL_CREATION_THRESHOLD", "1")
        from app.config import reset_config
        reset_config()

        tools = ["web_search", "calculator", "code_exec"]
        svc = _make_svc(db)
        svc.skills = None

        valid_spec = {
            "name": "auto_price_checker",
            "description": "Checks the price of a given asset",
            "parameters": [{"name": "asset", "type": "str", "description": "Asset name"}],
            "code": "def run(asset: str) -> str:\n    return 'Price: 100'\n",
        }

        gen_coro = None

        def _capture_task(coro):
            nonlocal gen_coro
            gen_coro = coro
            import asyncio as _asyncio
            fut = _asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

        store = _make_store(db)
        # Keep _generate_tool_spec patched while running the captured coroutine
        with patch("app.core.tool_triggers._generate_tool_spec", new=AsyncMock(return_value=valid_spec)):
            with patch("asyncio.create_task", side_effect=_capture_task):
                await maybe_trigger_tool_creation("q1", _make_tool_results(tools), svc, _candidate_store=store)
            # Still inside the _generate_tool_spec patch when we run it
            if gen_coro is not None:
                await gen_coro

        record = svc.custom_tools.get_tool("auto_price_checker")
        assert record is not None
        assert "asset" in record.description.lower() or "price" in record.description.lower()


# ---------------------------------------------------------------------------
# _generate_tool_spec LLM guards
# ---------------------------------------------------------------------------

class TestGenerateToolSpec:
    @pytest.mark.asyncio
    async def test_skip_response_returns_none(self):
        """LLM skip response → None returned."""
        with patch("app.core.tool_triggers.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"skip": true, "reason": "not useful"}')
            mock_llm.extract_json_object.return_value = {"skip": True, "reason": "not useful"}
            result = await _generate_tool_spec(["query1"], ["web_search", "calculator"])
        assert result is None

    @pytest.mark.asyncio
    async def test_code_with_tool_create_rejected(self):
        """Generated code that calls tool_create is rejected."""
        bad_code = "def run(x: str) -> str:\n    tool_create('bomb', '', '[]', 'pass')\n"
        with patch("app.core.tool_triggers.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="{}")
            mock_llm.extract_json_object.return_value = {
                "name": "bad_tool",
                "description": "Does bad things",
                "parameters": [],
                "code": bad_code,
            }
            result = await _generate_tool_spec(["q"], ["web_search"])
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_name_returns_none(self):
        """LLM response with no name → None."""
        with patch("app.core.tool_triggers.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="{}")
            mock_llm.extract_json_object.return_value = {
                "description": "something",
                "code": "def run() -> str: return 'x'",
            }
            result = await _generate_tool_spec(["q"], ["web_search"])
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_response_returned(self):
        """Valid LLM response is returned as-is."""
        spec = {
            "name": "my_tool",
            "description": "Does something useful",
            "parameters": [{"name": "x", "type": "str", "description": "input"}],
            "code": "def run(x: str) -> str:\n    return x\n",
        }
        with patch("app.core.tool_triggers.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="{}")
            mock_llm.extract_json_object.return_value = spec
            result = await _generate_tool_spec(["q"], ["web_search"])
        assert result is not None
        assert result["name"] == "my_tool"
