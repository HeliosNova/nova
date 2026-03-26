"""Tests for channel adapters: Discord, Telegram, Signal, WhatsApp.

Each adapter follows the same pattern: message splitting, user allowlist,
query handling via brain.think(), and alert sending.
"""

from __future__ import annotations

import asyncio
import collections
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Discord
# ===========================================================================

discord_mod = pytest.importorskip("discord", reason="discord.py package not installed")


class TestDiscord:
    def test_short_message_no_split(self):
        from app.channels.discord import DiscordBot
        assert DiscordBot._split_message("Hello world") == ["Hello world"]

    def test_over_limit_splits(self):
        from app.channels.discord import DiscordBot
        msg = "word " * 1000  # ~5000 chars
        chunks = DiscordBot._split_message(msg, limit=2000)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 2000

    def test_splits_at_newline(self):
        from app.channels.discord import DiscordBot
        msg = "line1\n" * 400
        chunks = DiscordBot._split_message(msg, limit=2000)
        assert len(chunks) >= 2
        assert chunks[0].endswith("line1")

    @pytest.mark.asyncio
    async def test_error_event_returns_error_message(self):
        from app.channels.discord import DiscordBot

        bot = DiscordBot.__new__(DiscordBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})
        bot._conv_lock = asyncio.Lock()

        async def mock_think(**kwargs):
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.ERROR, data={"message": "model offline"})

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert "Error" in answer
            assert "model offline" in answer


# ===========================================================================
# Telegram
# ===========================================================================

telegram_mod = pytest.importorskip("telegram", reason="python-telegram-bot package not installed")


class TestTelegram:
    def test_short_message_no_split(self):
        from app.channels.telegram import TelegramBot
        assert TelegramBot._split_message("Hello world") == ["Hello world"]

    def test_over_limit_splits(self):
        from app.channels.telegram import TelegramBot
        msg = "word " * 2000
        chunks = TelegramBot._split_message(msg, limit=4096)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 4096

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

    def test_allowlist_restricts_users(self):
        from app.channels.telegram import TelegramBot
        bot = MagicMock()
        bot._allowed_users = {100, 200, 300}
        bot._is_allowed = TelegramBot._is_allowed.__get__(bot)
        assert bot._is_allowed(100) is True
        assert bot._is_allowed(999) is False

    def test_parse_allowed_users(self):
        from app.channels.telegram import TelegramBot
        with patch("app.channels.telegram.config") as mock_config:
            mock_config.TELEGRAM_ALLOWED_USERS = " 100 , 200 , 300 "
            result = TelegramBot._parse_allowed_users()
            assert result == {100, 200, 300}

    @pytest.mark.asyncio
    async def test_collects_tokens(self):
        from app.channels.telegram import TelegramBot

        bot = TelegramBot.__new__(TelegramBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})
        bot._conv_lock = asyncio.Lock()

        async def mock_think(**kwargs):
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.TOKEN, data={"text": "Hello "})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "world!"})
            yield StreamEvent(type=EventType.DONE, data={
                "conversation_id": "conv-1", "lessons_used": 0, "skill_used": None,
            })

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert answer == "Hello world!"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        from app.channels.telegram import TelegramBot

        bot = TelegramBot.__new__(TelegramBot)
        bot._conversations = collections.OrderedDict({42: "conv-1"})
        bot._conv_lock = asyncio.Lock()

        async def mock_think(**kwargs):
            raise RuntimeError("test error")

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", user_id=42)
            assert "went wrong" in answer.lower()


# ===========================================================================
# Signal
# ===========================================================================


class TestSignal:
    def test_init_reads_config(self):
        with patch("app.channels.signal.config") as mock_config:
            mock_config.SIGNAL_API_URL = "http://signal-cli:8080/"
            mock_config.SIGNAL_PHONE_NUMBER = "+15551234567"
            mock_config.SIGNAL_CHAT_ID = "+15559999999"
            mock_config.SIGNAL_ALLOWED_USERS = "+15551111111,+15552222222"
            mock_config.SIGNAL_POLL_INTERVAL = 2

            from app.channels.signal import SignalBot
            bot = SignalBot()
            assert bot.api_url == "http://signal-cli:8080"
            assert bot.phone_number == "+15551234567"

    def test_allowlist_filters(self):
        from app.channels.signal import SignalBot
        bot = MagicMock()
        bot._allowed_users = {"+15551111111", "+15552222222"}
        bot._is_allowed = SignalBot._is_allowed.__get__(bot)
        assert bot._is_allowed("+15551111111") is True
        assert bot._is_allowed("+15559999999") is False

    def test_message_splitting(self):
        from app.channels.signal import SignalBot
        assert SignalBot._split_message("Hello world") == ["Hello world"]
        msg = "word " * 2000
        chunks = SignalBot._split_message(msg, limit=4096)
        assert all(len(c) <= 4096 for c in chunks)

    @pytest.mark.asyncio
    async def test_duplicate_message_dedup(self):
        from app.channels.signal import SignalBot

        bot = SignalBot.__new__(SignalBot)
        bot._processed_ids = collections.OrderedDict()
        bot._allowed_users = set()
        bot._conversations = collections.OrderedDict()
        bot._conv_lock = asyncio.Lock()
        bot._client = AsyncMock()

        # Track dedup in memory instead of hitting real DB
        _seen = set()

        def mock_check_dedup(msg_id):
            return msg_id in _seen

        def mock_record_dedup(msg_id):
            _seen.add(msg_id)

        bot._check_dedup = mock_check_dedup
        bot._record_dedup = mock_record_dedup

        call_count = 0

        async def mock_handle_query(query, sender):
            nonlocal call_count
            call_count += 1
            return "response"

        bot._handle_query = mock_handle_query
        bot._send_message = AsyncMock()
        bot._split_message = SignalBot._split_message
        bot._is_allowed = SignalBot._is_allowed.__get__(bot)

        envelope = {
            "envelope": {
                "sourceNumber": "+15551234567",
                "dataMessage": {"timestamp": 1234567890, "message": "Hello"},
            }
        }

        await bot._process_envelope(envelope)
        await bot._process_envelope(envelope)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_send_alert_posts_to_api(self):
        from app.channels.signal import SignalBot

        bot = SignalBot.__new__(SignalBot)
        bot.api_url = "http://signal-cli:8080"
        bot.phone_number = "+15550000000"
        bot.default_recipient = "+15559999999"

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        bot._client = mock_client

        await bot.send_alert("System alert!")
        mock_client.post.assert_called_once()
        payload = mock_client.post.call_args[1]["json"]
        assert payload["message"] == "System alert!"


# ===========================================================================
# WhatsApp
# ===========================================================================


class TestWhatsApp:
    def test_init_reads_config(self):
        with patch("app.channels.whatsapp.config") as mock_config:
            mock_config.WHATSAPP_API_URL = "https://graph.facebook.com/v17.0/12345/messages"
            mock_config.WHATSAPP_API_TOKEN = "test-token"
            mock_config.WHATSAPP_VERIFY_TOKEN = "verify-secret"
            mock_config.WHATSAPP_PHONE_ID = "12345"
            mock_config.WHATSAPP_CHAT_ID = "+15551234567"
            mock_config.WHATSAPP_ALLOWED_USERS = "+15551111111,+15552222222"

            from app.channels.whatsapp import WhatsAppBot
            bot = WhatsAppBot()
            assert bot.api_url == "https://graph.facebook.com/v17.0/12345/messages"
            assert bot.verify_token == "verify-secret"

    def test_allowlist_filters(self):
        from app.channels.whatsapp import WhatsAppBot
        bot = MagicMock()
        bot._allowed_users = {"+15551111111", "+15552222222"}
        bot._is_allowed = WhatsAppBot._is_allowed.__get__(bot)
        assert bot._is_allowed("+15551111111") is True
        assert bot._is_allowed("+15559999999") is False

    def test_message_splitting(self):
        from app.channels.whatsapp import WhatsAppBot
        assert WhatsAppBot._split_message("Hello world") == ["Hello world"]
        msg = "word " * 2000
        chunks = WhatsAppBot._split_message(msg, limit=4096)
        assert all(len(c) <= 4096 for c in chunks)

    @pytest.mark.asyncio
    async def test_collects_tokens(self):
        from app.channels.whatsapp import WhatsAppBot

        bot = WhatsAppBot.__new__(WhatsAppBot)
        bot._conversations = collections.OrderedDict({"+15551234567": "conv-1"})
        bot._conv_lock = asyncio.Lock()

        async def mock_think(**kwargs):
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.TOKEN, data={"text": "Hello "})
            yield StreamEvent(type=EventType.TOKEN, data={"text": "from WhatsApp!"})
            yield StreamEvent(type=EventType.DONE, data={})

        with patch("app.core.brain.think", side_effect=mock_think):
            answer = await bot._handle_query("Hi", "+15551234567")
            assert answer == "Hello from WhatsApp!"

    @pytest.mark.asyncio
    async def test_send_alert_posts_to_api(self):
        from app.channels.whatsapp import WhatsAppBot

        bot = WhatsAppBot.__new__(WhatsAppBot)
        bot.api_url = "https://graph.facebook.com/v17.0/12345/messages"
        bot.api_token = "test-token"
        bot.default_chat_id = "+15551234567"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        bot._client = mock_client

        await bot.send_alert("Server is down!")
        mock_client.post.assert_called_once()
        payload = mock_client.post.call_args[1]["json"]
        assert payload["to"] == "+15551234567"
        assert payload["text"]["body"] == "Server is down!"
