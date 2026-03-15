"""Discord channel adapter — connects Nova_ to Discord via discord.py."""

from __future__ import annotations

import asyncio
import collections
import logging

import discord

from app.config import config
from app.schema import EventType

logger = logging.getLogger(__name__)


class DiscordBot:
    """Discord bot that calls think() directly for each user message."""

    def __init__(self):
        self.token = config.DISCORD_TOKEN
        self.default_channel_id = config.DISCORD_CHANNEL_ID

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._conversations: collections.OrderedDict[int, str] = collections.OrderedDict()  # discord user_id → conv_id
        self._allowed_users = self._parse_allowed_users()
        self._setup_events()

    @staticmethod
    def _parse_allowed_users() -> set[int]:
        """Parse comma-separated user IDs from config."""
        raw = config.DISCORD_ALLOWED_USERS
        if not raw:
            return set()
        try:
            return {int(uid.strip()) for uid in raw.split(",") if uid.strip()}
        except ValueError:
            logger.warning("[Discord] Invalid DISCORD_ALLOWED_USERS: %s", raw)
            return set()

    def _is_allowed(self, user_id: int) -> bool:
        """Check if user is in the allowlist. Empty list = allow all."""
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _setup_events(self):
        @self._client.event
        async def on_ready():
            guilds = [g.name for g in self._client.guilds]
            logger.info(
                "[Discord] Connected as %s to %d guild(s): %s",
                self._client.user, len(guilds), ", ".join(guilds),
            )

        @self._client.event
        async def on_message(message: discord.Message):
            if message.author.bot:
                return

            # Respond to DMs or when mentioned
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = self._client.user in message.mentions if self._client.user else False

            if not is_dm and not is_mentioned:
                return

            # Strip bot mention from content
            content = message.content
            if self._client.user:
                content = content.replace(f"<@{self._client.user.id}>", "").strip()
                content = content.replace(f"<@!{self._client.user.id}>", "").strip()

            if not content:
                return

            if not self._is_allowed(message.author.id):
                await message.reply("Sorry, you're not authorized to use this bot.")
                return

            async with message.channel.typing():
                answer = await self._handle_query(content, message.author.id)

            for chunk in self._split_message(answer):
                await message.reply(chunk)

    async def _handle_query(self, query: str, user_id: int) -> str:
        """Run query through think() and collect the response."""
        from app.core.brain import think
        from app.core.brain import get_services

        # Get or create conversation for this user
        conv_id = self._conversations.get(user_id)
        if conv_id:
            # Move to end so it's marked as recently used
            self._conversations.move_to_end(user_id)
        else:
            svc = get_services()
            conv_id = svc.conversations.create_conversation()
            self._conversations[user_id] = conv_id
            # LRU eviction: cap at 1000 entries
            while len(self._conversations) > 1000:
                self._conversations.popitem(last=False)

        try:
            tokens = []
            async for event in think(query=query, conversation_id=conv_id, channel="discord"):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
                elif event.type == EventType.ERROR:
                    return f"Error: {event.data.get('message', 'unknown error')}"

            answer = "".join(tokens).strip()
            return answer if answer else "I processed your message but had no response."

        except Exception as e:
            logger.error("[Discord] Query failed: %s", e, exc_info=True)
            return "Sorry, something went wrong while processing your message."

    @staticmethod
    def _split_message(text: str, limit: int = 2000) -> list[str]:
        """Split a message into chunks that fit Discord's character limit."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            # Try to split at a newline
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = text.rfind(" ", 0, limit)
            if split_at == -1:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        return chunks

    async def send_alert(self, message: str):
        """Send a message to the default channel."""
        if not self.default_channel_id or not self._client.is_ready():
            return
        try:
            channel = self._client.get_channel(int(self.default_channel_id))
            if channel:
                for chunk in self._split_message(message):
                    await channel.send(chunk)
        except Exception as e:
            logger.error("[Discord] Alert send failed: %s", e)

    async def start(self):
        """Start the Discord bot (blocks until disconnected)."""
        if not self.token:
            logger.warning("[Discord] No token configured, skipping")
            return
        try:
            await self._client.start(self.token)
        except discord.LoginFailure as e:
            logger.error("[Discord] Authentication failed (check DISCORD_TOKEN): %s", e)
        except discord.ConnectionClosed as e:
            logger.error("[Discord] Connection closed: code=%s", e.code)
        except Exception as e:
            logger.error("[Discord] Bot failed: %s", e, exc_info=True)

    async def close(self):
        """Gracefully close the Discord connection."""
        if self._client and not self._client.is_closed():
            await self._client.close()
