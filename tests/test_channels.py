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

class TestSendSerialization:
    """Concurrent multi-chunk alerts must NOT interleave their parts — the
    'monitors crossing paths on Discord' bug (2026-06-14). Two monitors firing
    via the heartbeat gather used to interleave their split messages."""

    @pytest.mark.asyncio
    async def test_concurrent_discord_alerts_do_not_interleave(self):
        from app.channels.discord import DiscordBot

        bot = DiscordBot.__new__(DiscordBot)
        bot.default_channel_id = "123"
        bot._send_lock = asyncio.Lock()

        sent: list[str] = []

        class _Chan:
            async def send(self, chunk):
                # Yield mid-send so a competing sender could interleave if unlocked.
                await asyncio.sleep(0)
                sent.append(chunk)

        class _Client:
            def is_ready(self): return True
            def get_channel(self, _id): return _Chan()

        bot._client = _Client()

        # Two long alerts that each split into multiple Discord messages.
        msg_a = "A" * 5000   # ~3 chunks of 'A'
        msg_b = "B" * 5000   # ~3 chunks of 'B'
        await asyncio.gather(bot.send_alert(msg_a), bot.send_alert(msg_b))

        # Every A-chunk must be contiguous (no B between them) and vice-versa.
        # Continuation chunks carry a '_(cont. i/n)_' prefix — strip it before
        # reading the marker char that identifies which alert the chunk belongs to.
        import re as _re

        def _kind(c: str) -> str:
            return _re.sub(r"^_\(cont\. \d+/\d+\)_\n", "", c)[0]

        kinds = [_kind(c) for c in sent]  # first content char marks which alert
        # find the boundary: kinds should be all-one-letter then all-the-other
        first = kinds[0]
        switches = sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i-1])
        assert switches == 1, f"alerts interleaved (kinds={kinds})"
