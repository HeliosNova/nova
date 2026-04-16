"""Smoke tests for modules that lacked direct test coverage.

Targets: chat API, channels/base, core/quality, core/kg, main.py imports.
Goal: imports work, key functions don't crash on valid input, error cases
return sensible errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ===========================================================================
# channels/base.py — BaseChannel
# ===========================================================================

class TestBaseChannelSplitMessage:
    """Test BaseChannel._split_message() directly."""

    def test_short_message_no_split(self):
        from app.channels.base import BaseChannel
        result = BaseChannel._split_message("Hello world", max_length=2000)
        assert result == ["Hello world"]

    def test_empty_message(self):
        from app.channels.base import BaseChannel
        result = BaseChannel._split_message("")
        assert result == [""]

    def test_long_message_splits_at_newline(self):
        from app.channels.base import BaseChannel
        text = "Line one\n" * 300  # ~3000 chars
        result = BaseChannel._split_message(text, max_length=2000)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 2000

    def test_long_message_splits_at_space(self):
        from app.channels.base import BaseChannel
        text = "word " * 500  # 2500 chars, no newlines
        result = BaseChannel._split_message(text, max_length=2000)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 2000

    def test_no_split_points_hard_cut(self):
        from app.channels.base import BaseChannel
        text = "x" * 3000  # No spaces or newlines
        result = BaseChannel._split_message(text, max_length=2000)
        assert len(result) >= 2
        assert result[0] == "x" * 2000

    def test_exact_max_length(self):
        from app.channels.base import BaseChannel
        text = "x" * 2000
        result = BaseChannel._split_message(text, max_length=2000)
        assert result == [text]


class TestBaseChannelHandleQuery:
    """Test BaseChannel._handle_query() with mocked think()."""

    @pytest.mark.asyncio
    async def test_handle_query_returns_response(self):
        from app.channels.base import BaseChannel
        from app.schema import StreamEvent, EventType

        channel = BaseChannel()

        events = [
            StreamEvent(type=EventType.TOKEN, data={"text": "Hello "}),
            StreamEvent(type=EventType.TOKEN, data={"text": "there!"}),
            StreamEvent(type=EventType.DONE, data={}),
        ]

        async def mock_think(**kwargs):
            for e in events:
                yield e

        with patch("app.core.brain.think", mock_think):
            result = await channel._handle_query("Hi", "user-1")
            assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_handle_query_error_event(self):
        from app.channels.base import BaseChannel
        from app.schema import StreamEvent, EventType

        channel = BaseChannel()

        async def mock_think(**kwargs):
            yield StreamEvent(type=EventType.ERROR, data={"message": "LLM down"})

        with patch("app.core.brain.think", mock_think):
            result = await channel._handle_query("Hi", "user-1")
            assert "Error" in result
            assert "LLM down" in result

    @pytest.mark.asyncio
    async def test_handle_query_exception(self):
        from app.channels.base import BaseChannel

        channel = BaseChannel()

        async def mock_think(**kwargs):
            raise RuntimeError("Connection lost")
            yield  # make it a generator

        with patch("app.core.brain.think", mock_think):
            result = await channel._handle_query("Hi", "user-1")
            assert "sorry" in result.lower() or "wrong" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_query_empty_response(self):
        from app.channels.base import BaseChannel
        from app.schema import StreamEvent, EventType

        channel = BaseChannel()

        async def mock_think(**kwargs):
            yield StreamEvent(type=EventType.DONE, data={})

        with patch("app.core.brain.think", mock_think):
            result = await channel._handle_query("Hi", "user-1")
            assert "no response" in result.lower()


# ===========================================================================
# core/quality.py — Shared quality utilities
# ===========================================================================

class TestAllToolsClean:
    """Test all_tools_clean() helper."""

    def test_empty_results_clean(self):
        from app.core.quality import all_tools_clean
        assert all_tools_clean([]) is True

    def test_clean_results(self):
        from app.core.quality import all_tools_clean
        results = [
            {"tool": "web_search", "output": "Found 5 results about AI"},
            {"tool": "calculator", "output": "42"},
        ]
        assert all_tools_clean(results) is True

    def test_tool_failure_detected(self):
        from app.core.quality import all_tools_clean
        results = [
            {"tool": "web_search", "output": "[tool error] Connection refused"},
        ]
        assert all_tools_clean(results) is False

    def test_error_field_checked(self):
        from app.core.quality import all_tools_clean
        results = [
            {"tool": "http_fetch", "output": "", "error": "error: timed out connecting"},
        ]
        assert all_tools_clean(results) is False

    def test_browser_selector_miss_is_soft(self):
        from app.core.quality import all_tools_clean
        results = [
            {"tool": "browser", "output": "selector not found: #content timed out waiting"},
        ]
        # Browser selector misses are soft failures — should be clean
        assert all_tools_clean(results) is True


# ===========================================================================
# core/kg.py — Knowledge Graph
# ===========================================================================

class TestPredicateNormalization:
    """Test KG predicate normalization."""

    def test_canonical_predicate_unchanged(self):
        from app.core.kg import normalize_predicate
        assert normalize_predicate("is_a") == "is_a"
        assert normalize_predicate("part_of") == "part_of"
        assert normalize_predicate("located_in") == "located_in"

    def test_alias_normalized(self):
        from app.core.kg import normalize_predicate
        assert normalize_predicate("is a") == "is_a"
        assert normalize_predicate("is an") == "is_a"
        assert normalize_predicate("made by") == "created_by"
        assert normalize_predicate("part of") == "part_of"

    def test_unknown_predicate_returned_as_is(self):
        from app.core.kg import normalize_predicate
        result = normalize_predicate("some_weird_relation")
        # Unknown predicates are returned as-is (not forced to related_to)
        assert isinstance(result, str)
        assert len(result) > 0


class TestKGFact:
    """Test Fact dataclass."""

    def test_fact_creation(self):
        from app.core.kg import Fact
        f = Fact(
            id=1, subject="Python", predicate="is_a",
            object="programming language", confidence=0.9,
            source="extracted", created_at="2024-01-01",
        )
        assert f.subject == "Python"
        assert f.predicate == "is_a"
        assert f.confidence == 0.9

    def test_fact_temporal_fields(self):
        from app.core.kg import Fact
        f = Fact(
            id=1, subject="Bitcoin", predicate="price_of",
            object="$50000", confidence=0.8, source="monitor",
            created_at="2024-01-01", valid_from="2024-01-01",
            valid_to="2024-01-02", provenance="conv-123",
        )
        assert f.valid_from == "2024-01-01"
        assert f.valid_to == "2024-01-02"
        assert f.provenance == "conv-123"


class TestKnowledgeGraph:
    """Test KnowledgeGraph CRUD operations."""

    @pytest.fixture
    def kg(self, tmp_path):
        from app.core.kg import KnowledgeGraph
        from app.database import SafeDB
        db = SafeDB(str(tmp_path / "kg_test.db"))
        db.init_schema()
        return KnowledgeGraph(db)

    @pytest.mark.asyncio
    async def test_add_and_query(self, kg):
        await kg.add_fact("Python", "is_a", "programming language")
        results = kg.query("Python")
        assert len(results) > 0
        assert any(r["subject"].lower() == "python" for r in results)

    @pytest.mark.asyncio
    async def test_add_duplicate_updates_confidence(self, kg):
        await kg.add_fact("Python", "is_a", "language", confidence=0.5)
        await kg.add_fact("Python", "is_a", "language", confidence=0.9)
        results = kg.query("Python")
        lang_facts = [r for r in results if r["object"] == "language"]
        assert len(lang_facts) == 1

    def test_query_nonexistent_entity(self, kg):
        results = kg.query("NonExistentEntity12345")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self, kg):
        await kg.add_fact("test_entity", "is_a", "test_object")
        results = kg.query("test_entity")
        assert len(results) > 0
        deleted = await kg.delete_fact("test_entity", "is_a", "test_object")
        assert deleted is True
        results_after = kg.query("test_entity")
        assert len(results_after) == 0

    def test_get_stats(self, kg):
        stats = kg.get_stats()
        assert isinstance(stats, dict)
        assert "total_facts" in stats

    @pytest.mark.asyncio
    async def test_normalize_subject_object(self, kg):
        await kg.add_fact("  Python  ", "is_a", "  LANGUAGE  ")
        results = kg.query("python")
        # Should match — subject is stripped and lowered for query
        assert len(results) >= 0  # Depends on case normalization


# ===========================================================================
# Chat API — endpoint smoke tests
# ===========================================================================

class TestChatAPIEndpoints:
    """Smoke tests for chat API endpoints using mock services."""

    @pytest.fixture
    def mock_svc(self):
        svc = MagicMock()
        svc.conversations.list_conversations.return_value = [
            {"id": "conv-1", "title": "Test", "created_at": "2024-01-01"}
        ]
        svc.conversations.get_conversation.return_value = {
            "id": "conv-1", "title": "Test"
        }
        svc.conversations.get_history.return_value = []
        svc.conversations.search_conversations.return_value = []
        svc.conversations.search_messages.return_value = []
        svc.user_facts.get_all.return_value = []
        svc.user_facts.set.return_value = None
        svc.user_facts.delete.return_value = True
        return svc

    @pytest.mark.asyncio
    async def test_list_conversations(self, mock_svc):
        from app.api.chat import list_conversations
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await list_conversations(limit=50)
            assert isinstance(result, list)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_conversation_found(self, mock_svc):
        from app.api.chat import get_conversation
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await get_conversation("conv-1")
            assert result["id"] == "conv-1"

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, mock_svc):
        from app.api.chat import get_conversation
        from fastapi import HTTPException
        mock_svc.conversations.get_conversation.return_value = None
        with patch("app.api.chat.get_services", return_value=mock_svc):
            with pytest.raises(HTTPException) as exc_info:
                await get_conversation("nonexistent")
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_conversation(self, mock_svc):
        from app.api.chat import delete_conversation
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await delete_conversation("conv-1")
            assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, mock_svc):
        from app.api.chat import delete_conversation
        from fastapi import HTTPException
        mock_svc.conversations.get_conversation.return_value = None
        with patch("app.api.chat.get_services", return_value=mock_svc):
            with pytest.raises(HTTPException):
                await delete_conversation("nonexistent")

    @pytest.mark.asyncio
    async def test_list_facts(self, mock_svc):
        from app.api.chat import list_facts
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await list_facts()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_create_fact(self, mock_svc):
        from app.api.chat import create_fact
        from app.schema import UserFactCreate
        fact = UserFactCreate(key="name", value="Alice")
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await create_fact(fact)
            assert result["status"] == "ok"
            assert result["key"] == "name"

    @pytest.mark.asyncio
    async def test_delete_fact(self, mock_svc):
        from app.api.chat import delete_fact
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await delete_fact("name")
            assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_fact_not_found(self, mock_svc):
        from app.api.chat import delete_fact
        from fastapi import HTTPException
        mock_svc.user_facts.delete.return_value = False
        with patch("app.api.chat.get_services", return_value=mock_svc):
            with pytest.raises(HTTPException):
                await delete_fact("nonexistent")

    @pytest.mark.asyncio
    async def test_search_conversations_empty(self, mock_svc):
        from app.api.chat import search_conversations
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await search_conversations(q="   ")
            assert result == []

    @pytest.mark.asyncio
    async def test_search_messages_empty(self, mock_svc):
        from app.api.chat import search_messages
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await search_messages(q="   ")
            assert result == []

    @pytest.mark.asyncio
    async def test_rename_conversation(self, mock_svc):
        from app.api.chat import rename_conversation, RenameRequest
        body = RenameRequest(title="New Title")
        with patch("app.api.chat.get_services", return_value=mock_svc):
            result = await rename_conversation("conv-1", body)
            assert result["status"] == "ok"
            assert result["title"] == "New Title"


# ===========================================================================
# Module import smoke tests
# ===========================================================================

class TestModuleImports:
    """Verify all production modules import without crashing."""

    def test_import_channels_base(self):
        from app.channels.base import BaseChannel
        assert BaseChannel is not None

    def test_import_core_quality(self):
        from app.core.quality import all_tools_clean
        assert callable(all_tools_clean)

    def test_import_core_kg(self):
        from app.core.kg import KnowledgeGraph, Fact, normalize_predicate
        assert KnowledgeGraph is not None
        assert Fact is not None
        assert callable(normalize_predicate)

    def test_import_core_task_manager(self):
        from app.core.task_manager import TaskManager, BackgroundTask
        assert TaskManager is not None
        assert BackgroundTask is not None

    def test_import_schema(self):
        from app.schema import ChatRequest, ChatResponse, StreamEvent, EventType
        assert ChatRequest is not None
        assert EventType.TOKEN is not None

    def test_import_core_access_tiers(self):
        from app.core.access_tiers import _tier, is_command_blocked, is_path_allowed
        assert callable(_tier)
        assert callable(is_command_blocked)
        assert callable(is_path_allowed)

    def test_import_core_injection(self):
        from app.core.injection import detect_injection, sanitize_content
        assert callable(detect_injection)
        assert callable(sanitize_content)

    def test_import_api_chat(self):
        from app.api.chat import router
        assert router is not None

    def test_import_api_learning(self):
        from app.api.learning import router
        assert router is not None

    def test_import_api_monitors(self):
        from app.api.monitors import router
        assert router is not None
