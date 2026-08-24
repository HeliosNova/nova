"""Discord channel adapter — connects Nova to Discord via discord.py."""

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
        self._shutdown = False  # set by close(); distinguishes app shutdown from a dead client
        self._conversations: collections.OrderedDict[int, str] = collections.OrderedDict()  # discord user_id → conv_id
        self._conv_store = None  # lazy-init DB store
        self._conv_lock = asyncio.Lock()
        # Serializes ALL multi-chunk sends. Slow monitors run concurrently
        # (heartbeat gather), so two alerts that each split into several 2000-char
        # Discord messages used to interleave their chunks ("crossing paths").
        # Every send path holds this lock so one message's chunks post fully
        # before the next begins; a tiny inter-chunk delay also respects Discord's
        # per-channel rate limit.
        self._send_lock = asyncio.Lock()
        self._allowed_users = self._parse_allowed_users()
        if not self._allowed_users:
            logger.warning(
                "[Discord] DISCORD_ALLOWED_USERS is empty — ALL users are denied "
                "(fail-closed). Set your Discord user ID to use the bot."
            )
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
        """Check if user is in the allowlist. Empty list = deny all (fail-closed)."""
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
                logger.debug("[Discord] Ignoring message — not a DM and bot not mentioned (requires either)")
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
                await self._stream_reply(message, content)

    async def _get_conversation_id(self, user_id: int) -> str:
        """Get or create the conversation for this user (memory cache + DB fallback)."""
        from app.core.brain import get_services

        async with self._conv_lock:
            conv_id = self._conversations.get(user_id)
            if conv_id:
                self._conversations.move_to_end(user_id)
            else:
                # Try DB recovery
                if self._conv_store is None:
                    from app.database import get_db, ChannelConversationStore
                    self._conv_store = ChannelConversationStore(get_db())
                conv_id = self._conv_store.get("discord", str(user_id))
                if not conv_id:
                    svc = get_services()
                    conv_id = svc.conversations.create_conversation()
                    self._conv_store.set("discord", str(user_id), conv_id)
                self._conversations[user_id] = conv_id
                while len(self._conversations) > 1000:  # LRU cap for personal bot
                    self._conversations.popitem(last=False)
        return conv_id

    async def _stream_reply(self, message: "discord.Message", query: str) -> None:
        """Stream-first delivery (#47): send the DRAFT the moment it's complete
        (the 'refining' stage signal), then EDIT it in place when the refine
        chain's REVISION lands — the reader gets an answer in generation time,
        not generation+refine time. Long multi-chunk drafts fall back to a
        single send at the end (editing a chunk train isn't worth the states)."""
        from app.core.brain import think

        conv_id = await self._get_conversation_id(message.author.id)
        _REFINING = "\n\n-# refining…"
        tokens: list[str] = []
        draft_msg = None
        final: str | None = None
        error: str | None = None
        try:
            async for event in think(query=query, conversation_id=conv_id, channel="discord"):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
                elif (event.type == EventType.THINKING
                      and event.data.get("stage") == "refining" and draft_msg is None):
                    draft = "".join(tokens).strip()
                    if draft and len(draft) + len(_REFINING) <= 2000:
                        try:
                            async with self._send_lock:
                                draft_msg = await message.reply(draft + _REFINING)
                        except Exception as e:
                            logger.warning("[Discord] draft send failed: %s — deliver at end", e)
                            draft_msg = None
                elif event.type == EventType.REVISION:
                    final = event.data.get("text", "")
                elif event.type == EventType.ERROR:
                    error = f"Error: {event.data.get('message', 'unknown error')}"
        except Exception as e:
            logger.error("[Discord] Query failed: %s", e, exc_info=True)
            error = "Sorry, something went wrong while processing your message."

        answer = (error or (final if final is not None else "".join(tokens))).strip()
        if not answer:
            answer = "I processed your message but had no response."
        chunks = self._split_message(answer)
        async with self._send_lock:
            if draft_msg is not None:
                try:
                    await draft_msg.edit(content=chunks[0])
                    chunks = chunks[1:]
                except Exception as e:
                    logger.warning("[Discord] draft edit failed (%s) — sending fresh", e)
            for i, chunk in enumerate(chunks):
                await message.reply(chunk)
                if i + 1 < len(chunks):
                    await asyncio.sleep(0.3)

    async def _handle_query(self, query: str, user_id: int) -> str:
        """Run query through think() and collect the final response (non-streaming
        fallback; on_message uses _stream_reply for live draft delivery)."""
        from app.core.brain import think

        conv_id = await self._get_conversation_id(user_id)
        try:
            tokens = []
            async for event in think(query=query, conversation_id=conv_id, channel="discord"):
                if event.type == EventType.REVISION:
                    # stream-first refine: the final answer replaces the draft
                    tokens = [event.data.get("text", "")]
                elif event.type == EventType.TOKEN:
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

    async def send_alert(self, message: str) -> bool:
        """Send a message to the default channel.

        Returns True only when the message was actually delivered — callers
        record delivery, so a swallowed failure here becomes a permanently
        lost digest (audit 2026-07-08)."""
        if not self.default_channel_id:
            logger.warning("[Discord] Skipping alert — no default channel configured")
            return False
        if not self._client.is_ready():
            # Wait briefly for the client to become ready (e.g. during startup)
            for _ in range(10):
                await asyncio.sleep(1)
                if self._client.is_ready():
                    break
            if not self._client.is_ready():
                logger.warning("[Discord] Skipping alert — client not ready after 10s wait")
                return False
        try:
            channel = self._client.get_channel(int(self.default_channel_id))
            if channel:
                # Reserve headroom for a continuation marker so split chunks 2..n
                # aren't orphaned fragments (a reader otherwise can't tell a
                # continuation from a new monitor's message).
                chunks = self._split_message(message, limit=1960)
                n = len(chunks)
                async with self._send_lock:
                    for i, chunk in enumerate(chunks):
                        if n > 1 and i > 0:
                            chunk = f"_(cont. {i + 1}/{n})_\n{chunk}"
                        await channel.send(chunk)
                        if i + 1 < n:
                            await asyncio.sleep(0.3)  # pace multi-part posts
                return True
            logger.warning("[Discord] Skipping alert — channel %s not found", self.default_channel_id)
            return False
        except Exception as e:
            logger.error("[Discord] Alert send failed: %s", e)
            return False

    async def start(self):
        """Start the Discord bot with reconnection and exponential backoff."""
        if not self.token:
            logger.warning("[Discord] No token configured, skipping")
            return

        import time
        _INITIAL_BACKOFF = 5.0
        _MAX_BACKOFF = 60.0
        _STABLE_UPTIME_S = 300.0   # 5 min of uptime resets the backoff
        backoff = _INITIAL_BACKOFF

        while True:
            # discord.py cannot restart a closed Client — a second .start() on
            # it returns immediately, which used to hit the silent `return`
            # below and leave the primary delivery channel dead until app
            # restart (audit 2026-08-19). Rebuild client + handlers per retry,
            # the way the telegram adapter does.
            if self._client.is_closed():
                intents = discord.Intents.default()
                intents.message_content = True
                self._client = discord.Client(intents=intents)
                self._setup_events()
            connect_started = time.monotonic()
            try:
                await self._client.start(self.token)
                if self._shutdown:
                    return  # Clean exit — close() during app shutdown
                logger.warning(
                    "[Discord] gateway loop exited without exception (client "
                    "closed) — reconnecting in %.0fs", backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)
                continue
            except discord.LoginFailure as e:
                logger.error("[Discord] Authentication failed (check DISCORD_TOKEN): %s", e)
                return  # Don't retry auth failures
            except asyncio.CancelledError:
                logger.info("[Discord] Bot shutting down")
                return
            except (discord.ConnectionClosed, Exception) as e:
                # Reset backoff if the connection was stable for ≥5 min before
                # dropping — otherwise a long-running bot that occasionally
                # blips at the gateway would compound backoff to MAX_BACKOFF
                # permanently and reconnect slowly.
                uptime = time.monotonic() - connect_started
                if uptime >= _STABLE_UPTIME_S:
                    if backoff > _INITIAL_BACKOFF:
                        logger.info(
                            "[Discord] Connection was stable for %.0fs — resetting backoff %.0f → %.0fs",
                            uptime, backoff, _INITIAL_BACKOFF,
                        )
                    backoff = _INITIAL_BACKOFF
                level = "warning" if isinstance(e, discord.ConnectionClosed) else "error"
                getattr(logger, level)(
                    "[Discord] %s after %.0fs uptime: %s — reconnecting in %.0fs",
                    type(e).__name__, uptime, e, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def close(self):
        """Gracefully close the Discord connection."""
        self._shutdown = True
        if self._client and not self._client.is_closed():
            await self._client.close()
