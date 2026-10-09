"""Discord channel adapter — connects Nova to Discord via discord.py."""

from __future__ import annotations

import asyncio
import logging

import discord

from app.config import config

logger = logging.getLogger(__name__)


class DiscordBot:
    """Discord delivery adapter: posts Nova's briefings to the configured channel.

    Inbound chat was removed 2026-10-09 (40 user messages ever, all in August;
    the brain stays reachable through the HTTP chat API). This client only
    sends; it registers no message handler and needs no message-content intent."""

    def __init__(self):
        self.token = config.DISCORD_TOKEN
        self.default_channel_id = config.DISCORD_CHANNEL_ID

        self._client = discord.Client(intents=discord.Intents.default())
        self._shutdown = False  # set by close(); distinguishes app shutdown from a dead client
        # Serializes ALL multi-chunk sends. Slow monitors run concurrently
        # (heartbeat gather), so two alerts that each split into several 2000-char
        # Discord messages used to interleave their chunks ("crossing paths").
        # Every send path holds this lock so one message's chunks post fully
        # before the next begins; a tiny inter-chunk delay also respects Discord's
        # per-channel rate limit.
        self._send_lock = asyncio.Lock()
        self._setup_events()

    def _setup_events(self):
        @self._client.event
        async def on_ready():
            guilds = [g.name for g in self._client.guilds]
            logger.info(
                "[Discord] Connected as %s to %d guild(s): %s",
                self._client.user, len(guilds), ", ".join(guilds),
            )

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
                # Discord is the dialect everything else converts FROM, which is
                # why it was the one channel with no converter of its own — and
                # why a bare <URL> shipped with the whole URL as its visible
                # text. On a feed row whose body is one line, a 100-character
                # filing URL wraps to three blue lines and the item reads as a
                # link (owner, 2026-09-04). Telegram was fixed on 2026-08-31.
                from app.channels.format_for_channel import to_discord
                message = to_discord(message)
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
                self._client = discord.Client(intents=discord.Intents.default())
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
