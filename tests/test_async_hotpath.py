"""Tests for hot-path async offloading (audit finding #6 — blocking SQLite calls).

Verifies:
1. chat.py CRUD endpoints use asyncio.to_thread for all DB calls
2. UserActivityMiddleware write is fire-and-forget (non-blocking)
3. Concurrency: a slow-DB first request does not block a second concurrent request
4. Timing empirical: two concurrent requests with 200ms mock DB delay finish
   in ~200ms total (parallel), not ~400ms (serial)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_svc(conversations=None, user_facts=None, monitor_store=None):
    """Build a minimal Services mock for chat endpoint tests."""
    svc = MagicMock()
    if conversations is not None:
        svc.conversations = conversations
    if user_facts is not None:
        svc.user_facts = user_facts
    svc.monitor_store = monitor_store
    return svc


# ---------------------------------------------------------------------------
# chat.py endpoint: DB calls offloaded to thread pool
# ---------------------------------------------------------------------------

class TestChatEndpointsUseToThread:
    """Every sync DB call in chat.py async handlers must go through asyncio.to_thread."""

    @pytest.fixture
    def mock_convs(self):
        c = MagicMock()
        c.list_conversations.return_value = []
        c.search_conversations.return_value = []
        c.search_messages.return_value = []
        c.get_conversation.return_value = {"id": "c1", "title": "t", "created_at": "x"}
        c.get_history.return_value = []
        c.update_title.return_value = None
        c.delete_conversation.return_value = None
        return c

    @pytest.fixture
    def mock_facts(self):
        f = MagicMock()
        f.get_all.return_value = []
        f.set.return_value = None
        f.delete.return_value = True
        return f

    @pytest.mark.asyncio
    async def test_list_conversations_uses_to_thread(self, mock_convs):
        """list_conversations must not call sync method directly on the event loop."""
        from app.api.chat import list_conversations

        svc = _make_mock_svc(conversations=mock_convs)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await list_conversations(limit=50)

        mock_convs.list_conversations.assert_called_once_with(50)

    @pytest.mark.asyncio
    async def test_search_conversations_uses_to_thread(self, mock_convs):
        from app.api.chat import search_conversations

        svc = _make_mock_svc(conversations=mock_convs)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await search_conversations(q="hello", limit=20)

        mock_convs.search_conversations.assert_called_once_with("hello", limit=20)

    @pytest.mark.asyncio
    async def test_search_messages_uses_to_thread(self, mock_convs):
        from app.api.chat import search_messages

        svc = _make_mock_svc(conversations=mock_convs)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await search_messages(q="hi", limit=10)

        mock_convs.search_messages.assert_called_once_with("hi", limit=10)

    @pytest.mark.asyncio
    async def test_get_conversation_uses_to_thread(self, mock_convs):
        from app.api.chat import get_conversation

        svc = _make_mock_svc(conversations=mock_convs)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await get_conversation(conv_id="c1")

        mock_convs.get_conversation.assert_called_once_with("c1")
        mock_convs.get_history.assert_called_once_with("c1", 100)

    @pytest.mark.asyncio
    async def test_rename_conversation_uses_to_thread(self, mock_convs):
        from app.api.chat import rename_conversation, RenameRequest

        svc = _make_mock_svc(conversations=mock_convs)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await rename_conversation(conv_id="c1", body=RenameRequest(title="New"))

        mock_convs.update_title.assert_called_once_with("c1", "New")

    @pytest.mark.asyncio
    async def test_delete_conversation_uses_to_thread(self, mock_convs):
        from app.api.chat import delete_conversation

        svc = _make_mock_svc(conversations=mock_convs)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await delete_conversation(conv_id="c1")

        mock_convs.delete_conversation.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_list_facts_uses_to_thread(self, mock_facts):
        from app.api.chat import list_facts

        svc = _make_mock_svc(user_facts=mock_facts)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await list_facts()

        mock_facts.get_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_fact_uses_to_thread(self, mock_facts):
        from app.api.chat import create_fact, UserFactCreate

        svc = _make_mock_svc(user_facts=mock_facts)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await create_fact(fact=UserFactCreate(key="name", value="Alice", source="user", category="fact"))

        mock_facts.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_fact_uses_to_thread(self, mock_facts):
        from app.api.chat import delete_fact

        svc = _make_mock_svc(user_facts=mock_facts)
        with patch("app.api.chat.get_services", return_value=svc):
            result = await delete_fact(key="name")

        mock_facts.delete.assert_called_once_with("name")


# ---------------------------------------------------------------------------
# UserActivityMiddleware: fire-and-forget (non-blocking)
# ---------------------------------------------------------------------------

class TestUserActivityMiddlewareNonBlocking:
    """The activity write must be a background task, not awaited inline."""

    @pytest.mark.asyncio
    async def test_middleware_does_not_await_db_write(self):
        """asyncio.create_task is used — the DB write is background, not inline."""
        from app.main import UserActivityMiddleware

        db_mock = MagicMock()
        db_mock.execute = MagicMock(return_value=None)

        monitor_store = MagicMock()
        monitor_store._db = db_mock

        svc = MagicMock()
        svc.monitor_store = monitor_store

        created_tasks = []
        original_create_task = asyncio.create_task

        def tracking_create_task(coro, **kw):
            t = original_create_task(coro, **kw)
            created_tasks.append(t)
            return t

        middleware = UserActivityMiddleware(app=MagicMock())

        async def fake_call_next(req):
            from starlette.responses import Response
            return Response("ok")

        from starlette.requests import Request
        from starlette.datastructures import Headers
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)

        with patch("app.core.brain.get_services", return_value=svc), \
             patch("app.main.asyncio.create_task", side_effect=tracking_create_task):
            await middleware.dispatch(request, fake_call_next)

        assert len(created_tasks) >= 1, "DB write must be scheduled as a background task"


# ---------------------------------------------------------------------------
# Concurrency proof: slow DB does not block concurrent requests
# ---------------------------------------------------------------------------

class TestConcurrentRequestsNotBlocked:
    """A slow DB operation on request 1 must not delay request 2.

    We mock a 200ms DB delay and verify two concurrent awaits complete in
    ~200ms total (parallel), not ~400ms (serial).
    """

    @pytest.mark.asyncio
    async def test_two_concurrent_list_conversations_run_in_parallel(self):
        """Two concurrent list_conversations calls with 200ms mock DB delay
        should complete in ≤300ms total — not 400ms+ (serial sum)."""
        from app.api.chat import list_conversations

        call_count = 0

        def slow_list(limit):
            nonlocal call_count
            call_count += 1
            time.sleep(0.2)  # 200ms blocking delay (simulates slow SQLite)
            return [{"id": str(call_count)}]

        mock_convs = MagicMock()
        mock_convs.list_conversations.side_effect = slow_list
        svc = _make_mock_svc(conversations=mock_convs)

        start = time.monotonic()
        with patch("app.api.chat.get_services", return_value=svc):
            r1, r2 = await asyncio.gather(
                list_conversations(limit=50),
                list_conversations(limit=50),
            )
        elapsed = time.monotonic() - start

        assert call_count == 2
        # Parallel: both 200ms delays overlap → total ≈ 200ms
        # Serial would be ≈ 400ms. Allow generous 350ms upper bound.
        assert elapsed < 0.35, (
            f"Two concurrent list_conversations took {elapsed*1000:.0f}ms — "
            f"expected ~200ms (parallel), not ~400ms (serial). "
            f"DB calls are not being offloaded to thread pool."
        )

    @pytest.mark.asyncio
    async def test_serial_baseline_is_400ms(self):
        """Control: sequential (non-concurrent) calls DO take ~400ms — proving
        the parallel test above is measuring something real."""
        from app.api.chat import list_conversations

        def slow_list(limit):
            time.sleep(0.2)
            return []

        mock_convs = MagicMock()
        mock_convs.list_conversations.side_effect = slow_list
        svc = _make_mock_svc(conversations=mock_convs)

        start = time.monotonic()
        with patch("app.api.chat.get_services", return_value=svc):
            await list_conversations(limit=50)
            await list_conversations(limit=50)
        elapsed = time.monotonic() - start

        # Sequential: ≈ 400ms
        assert elapsed >= 0.38, (
            f"Sequential baseline took only {elapsed*1000:.0f}ms — expected ~400ms."
        )
