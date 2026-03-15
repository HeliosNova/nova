"""Tests for Telegram channel adapter."""

from __future__ import annotations

import collections
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot package not installed")


class TestTelegramMessageSplitting:
    """Test message splitting logic."""

    def test_short_message_no_split(self):
        from app.channels.telegram import TelegramBot
        chunks = TelegramBot._split_message("Hello world")
        assert chunks == ["Hello world"]

    def test_exact_limit(self):
        from app.channels.telegram import TelegramBot
        msg = "a" * 4096
        chunks = TelegramBot._split_message(msg)
        assert len(chunks) == 1

    def test_over_limit_splits(self):
        from app.channels.telegram import TelegramBot
        msg = "word " * 2000  # ~10000 chars
        chunks = TelegramBot._split_message(msg, limit=4096)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_splits_at_newline(self):
        from app.channels.telegram import TelegramBot
        msg = "line1\n" * 1000  # ~6000 chars
        chunks = TelegramBot._split_message(msg, limit=4096)
        assert len(chunks) >= 2

    def test_empty_message(self):
        from app.channels.telegram import TelegramBot
        chunks = TelegramBot._split_message("")
        assert chunks == [""]


class TestTelegramAllowedUsers:
    """Test user authorization logic."""

    def test_empty_allowlist_permits_all(self):
        from app.channels.telegram import TelegramBot
        with patch("app.channels.telegram.config") as mock_config:
            mock_config.TELEGRAM_TOKEN = ""
            mock_config.TELEGRAM_CHAT_ID = ""
            mock_config.TELEGRAM_ALLOWED_USERS = ""
            bot = MagicMock()
            bot._allowed_users = set()
            bot._is_allowed = TelegramBot._is_allowed.__get__(bot)
            assert bot._is_allowed(12345) is True
            assert bot._is_allowed(99999) is True

    def test_allowlist_restricts_users(self):
        bot = MagicMock()
        bot._allowed_users = {100, 200, 300}
        from app.channels.telegram import TelegramBot
        bot._is_allowed = TelegramBot._is_allowed.__get__(bot)
        assert bot._is_allowed(100) is True
        assert bot._is_allowed(999) is False

    def test_parse_allowed_users(self):
        from app.channels.telegram import TelegramBot
        with patch("app.channels.telegram.config") as mock_config:
            mock_config.TELEGRAM_ALLOWED_USERS = "100,200,300"
            result = TelegramBot._parse_allowed_users()
            assert result == {100, 200, 300}

    def test_parse_allowed_users_empty(self):
        from app.channels.telegram import TelegramBot
        with patch("app.channels.telegram.config") as mock_config:
            mock_config.TELEGRAM_ALLOWED_USERS = ""
            result = TelegramBot._parse_allowed_users()
            assert result == set()

    def test_parse_allowed_users_with_spaces(self):
        from app.channels.telegram import TelegramBot
        with patch("app.channels.telegram.config") as mock_config:
            mock_config.TELEGRAM_ALLOWED_USERS = " 100 , 200 , 300 "
            result = TelegramBot._parse_allowed_users()
            assert result == {100, 200, 300}


class TestTelegramHandleQuery:
    """Test query handling with mocked brain."""

    @pytest.mark.asyncio
    async def test_collects_tokens(self):
        from app.channels.telegram import TelegramBot

        bot = TelegramBot.__new__(TelegramBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})

        async def mock_think(**kwargs):
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.TOKEN, data={"text": "Hello "})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "world!"})
            yield StreamEvent(type=EventType.DONE, data={
                "conversation_id": "conv-1",
                "lessons_used": 0,
                "skill_used": None,
            })

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert answer == "Hello world!"

    @pytest.mark.asyncio
    async def test_empty_response_returns_fallback(self):
        from app.channels.telegram import TelegramBot

        bot = TelegramBot.__new__(TelegramBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})

        async def mock_think(**kwargs):
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.DONE, data={
                "conversation_id": "conv-1",
                "lessons_used": 0,
                "skill_used": None,
            })

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert "processed" in answer.lower() or "no response" in answer.lower()

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        from app.channels.telegram import TelegramBot

        bot = TelegramBot.__new__(TelegramBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})

        async def mock_think(**kwargs):
            raise RuntimeError("test error")

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert "went wrong" in answer.lower()
