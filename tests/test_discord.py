"""Tests for Discord channel adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord", reason="discord.py package not installed")


class TestDiscordMessageSplitting:
    """Test message splitting logic (no mocking needed)."""

    def test_short_message_no_split(self):
        from app.channels.discord import DiscordBot
        chunks = DiscordBot._split_message("Hello world")
        assert chunks == ["Hello world"]

    def test_exact_limit(self):
        from app.channels.discord import DiscordBot
        msg = "a" * 2000
        chunks = DiscordBot._split_message(msg)
        assert len(chunks) == 1

    def test_over_limit_splits(self):
        from app.channels.discord import DiscordBot
        msg = "word " * 1000  # ~5000 chars
        chunks = DiscordBot._split_message(msg, limit=2000)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 2000

    def test_splits_at_newline(self):
        from app.channels.discord import DiscordBot
        msg = "line1\n" * 400  # ~2400 chars
        chunks = DiscordBot._split_message(msg, limit=2000)
        assert len(chunks) >= 2
        # Should split at a newline, not mid-word
        assert chunks[0].endswith("line1")

    def test_splits_at_space_when_no_newline(self):
        from app.channels.discord import DiscordBot
        msg = "word " * 500  # ~2500 chars, no newlines
        chunks = DiscordBot._split_message(msg, limit=2000)
        assert len(chunks) >= 2

    def test_empty_message(self):
        from app.channels.discord import DiscordBot
        chunks = DiscordBot._split_message("")
        assert chunks == [""]

    def test_custom_limit(self):
        from app.channels.discord import DiscordBot
        msg = "a" * 100
        chunks = DiscordBot._split_message(msg, limit=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 50


class TestDiscordHandleQuery:
    """Test query handling with mocked brain."""

    @pytest.fixture
    def bot(self):
        with patch("app.channels.discord.config") as mock_config:
            mock_config.DISCORD_TOKEN = "test-token"
            mock_config.DISCORD_CHANNEL_ID = "12345"
            bot = MagicMock()
            bot._conversations = {}
            bot._handle_query = DiscordBot._handle_query.__get__(bot)
            bot._split_message = DiscordBot._split_message
            return bot

    @pytest.mark.asyncio
    async def test_creates_conversation_on_first_message(self):
        from app.channels.discord import DiscordBot
        from app.core.brain import Services, set_services
        from app.core.memory import ConversationStore, UserFactStore

        # We need real services for conversation creation
        with patch("app.channels.discord.config") as mock_config:
            mock_config.DISCORD_TOKEN = ""
            mock_config.DISCORD_CHANNEL_ID = ""

            bot = DiscordBot.__new__(DiscordBot)
            bot._conversations = {}

            # Mock think() to yield a simple response
            async def mock_think(**kwargs):
                from app.schema import EventType, StreamEvent
                yield StreamEvent(type=EventType.TOKEN, data={"text": "Hello!"})
                yield StreamEvent(type=EventType.DONE, data={
                    "conversation_id": "conv-1",
                    "lessons_used": 0,
                    "skill_used": None,
                })

            mock_svc = MagicMock()
            mock_svc.conversations.create_conversation.return_value = "conv-1"

            with patch("app.core.brain.think", side_effect=mock_think), \
                 patch("app.core.brain.get_services", return_value=mock_svc):
                    answer = await bot._handle_query("Hi", user_id=42)
                    assert "Hello!" in answer

    @pytest.mark.asyncio
    async def test_error_event_returns_error_message(self):
        from app.channels.discord import DiscordBot

        import collections
        bot = DiscordBot.__new__(DiscordBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})

        async def mock_think(**kwargs):
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.ERROR, data={"message": "model offline"})

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert "Error" in answer
            assert "model offline" in answer


class TestDiscordSendAlert:
    """Test alert sending with mocked client."""

    @pytest.mark.asyncio
    async def test_send_alert_no_channel(self):
        from app.channels.discord import DiscordBot
        with patch("app.channels.discord.config") as mock_config:
            mock_config.DISCORD_TOKEN = ""
            mock_config.DISCORD_CHANNEL_ID = ""
            bot = DiscordBot.__new__(DiscordBot)
            bot.default_channel_id = ""
            bot._client = MagicMock()
            # Should return without error
            await bot.send_alert("test message")
