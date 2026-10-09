"""Telegram channel adapter — connects Nova to Telegram via python-telegram-bot."""

from __future__ import annotations

import asyncio
import logging
import re

from telegram.ext import Application

from app.config import config

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram delivery adapter: posts Nova's briefings to the default chat.

    Inbound chat and the /start, /status, /help commands were removed
    2026-10-09; the bot sends only, so it never polls for updates. The brain
    stays reachable through the HTTP chat API."""

    def __init__(self):
        self.token = config.TELEGRAM_TOKEN
        self.default_chat_id = config.TELEGRAM_CHAT_ID
        # Serializes multi-chunk sends so concurrent monitor alerts don't
        # interleave their parts (see Discord adapter for the full rationale).
        self._send_lock = asyncio.Lock()
        self._app: Application | None = None
        self._stop_event = asyncio.Event()

    # Telegram's HTML parser accepts only this tag set and rejects the whole
    # message on any imbalance ("unmatched end tag" → plain-text fallback,
    # ~4x/48h observed 2026-08-26). Two sources of imbalance: _split_message
    # cutting across an open tag (chunk 1 unclosed + chunk 2 orphan closer —
    # BOTH chunks bounce), and the model emitting a stray closer.
    _TG_TAG_RE = re.compile(
        r"<(/?)(b|strong|i|em|u|ins|s|strike|del|code|pre|a|tg-spoiler|blockquote)"
        r"(\s[^>]*)?>",
        re.IGNORECASE,
    )

    @classmethod
    def _balance_html_chunks(cls, chunks: list[str]) -> list[str]:
        """Deterministic tag repair across split chunks.

        Carries tags still open at a chunk boundary into the next chunk
        (close at the end, reopen at the start), strips orphan closers, and
        closes anything left dangling — so every chunk is independently
        valid Telegram HTML."""
        out: list[str] = []
        carried: list[tuple[str, str]] = []  # (tag_name, full_open_tag)
        for chunk in chunks:
            prefix = "".join(open_tag for _, open_tag in carried)
            stack = list(carried)
            rebuilt: list[str] = [prefix]
            pos = 0
            for m in cls._TG_TAG_RE.finditer(chunk):
                rebuilt.append(chunk[pos:m.start()])
                pos = m.end()
                is_close, name = bool(m.group(1)), m.group(2).lower()
                if not is_close:
                    stack.append((name, m.group(0)))
                    rebuilt.append(m.group(0))
                elif any(n == name for n, _ in stack):
                    # Close improperly-nested inner tags first, pop through
                    # the match, then reopen the inner ones (standard fixup).
                    inner: list[tuple[str, str]] = []
                    while stack and stack[-1][0] != name:
                        inner.append(stack.pop())
                        rebuilt.append(f"</{inner[-1][0]}>")
                    stack.pop()
                    rebuilt.append(f"</{name}>")
                    for n, open_tag in reversed(inner):
                        stack.append((n, open_tag))
                        rebuilt.append(open_tag)
                # else: orphan closer — drop it
            rebuilt.append(chunk[pos:])
            rebuilt.append("".join(f"</{n}>" for n, _ in reversed(stack)))
            out.append("".join(rebuilt))
            carried = stack
        return out

    @staticmethod
    def _split_message(text: str, limit: int = 4096) -> list[str]:
        """Split a message into chunks that fit Telegram's character limit."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = text.rfind(" ", 0, limit)
            if split_at == -1:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        return chunks

    async def send_alert(self, message: str) -> bool:
        """Send a message to the default chat. Converts Discord-style markdown
        to Telegram HTML so **bold**, *italic*, `code`, and <URL> render
        correctly instead of showing as raw asterisks/brackets."""
        if not self.default_chat_id or not self._app:
            return False
        try:
            from app.channels.format_for_channel import to_telegram_html
            html_message = to_telegram_html(message)
            _chunks = self._balance_html_chunks(self._split_message(html_message))
            async with self._send_lock:
              for _ci, chunk in enumerate(_chunks):
                if _ci:
                    await asyncio.sleep(0.3)  # pace multi-part posts
                try:
                    await self._app.bot.send_message(
                        chat_id=int(self.default_chat_id),
                        text=chunk,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except Exception as html_err:
                    # If HTML parsing fails (e.g. rare escaping edge case),
                    # fall back to plain text so the alert still goes through.
                    logger.warning("[Telegram] HTML send failed (%s) — sending plain", html_err)
                    await self._app.bot.send_message(
                        chat_id=int(self.default_chat_id), text=chunk,
                    )
            return True
        except Exception as e:
            logger.error("[Telegram] Alert send failed: %s", e)
            return False

    async def start(self):
        """Start the Telegram bot (polling mode) with outer reconnect loop."""
        if not self.token:
            logger.warning("[Telegram] No token configured, skipping")
            return

        import time
        _INITIAL_BACKOFF = 5.0
        _MAX_BACKOFF = 60.0
        _STABLE_UPTIME_S = 300.0
        backoff = _INITIAL_BACKOFF

        while not self._stop_event.is_set():
            connect_started = time.monotonic()
            try:
                self._app = Application.builder().token(self.token).build()
                await self._app.initialize()
                await self._app.start()
                logger.info("[Telegram] Bot started (delivery only, no polling)")

                if self.default_chat_id:
                    try:
                        chat = await self._app.bot.get_chat(chat_id=int(self.default_chat_id))
                        logger.info("[Telegram] Alert chat validated: %s (type=%s)", chat.title or chat.first_name or chat.id, chat.type)
                    except Exception as e:
                        logger.error(
                            "[Telegram] TELEGRAM_CHAT_ID=%s is invalid (%s). "
                            "Send /start to the bot, then check logs or use: "
                            "curl https://api.telegram.org/bot<TOKEN>/getUpdates",
                            self.default_chat_id, e,
                        )

                # Block until stop event — clean shutdown path
                await self._stop_event.wait()
                return  # clean exit

            except asyncio.CancelledError:
                logger.info("[Telegram] Bot shutting down")
                return
            except Exception as e:
                # Reset backoff if the connection was stable for ≥5 min so a
                # long-running bot's occasional polling blip doesn't compound
                # to MAX_BACKOFF permanently.
                uptime = time.monotonic() - connect_started
                if uptime >= _STABLE_UPTIME_S and backoff > _INITIAL_BACKOFF:
                    logger.info(
                        "[Telegram] Stable for %.0fs before failure — resetting backoff %.0f → %.0fs",
                        uptime, backoff, _INITIAL_BACKOFF,
                    )
                    backoff = _INITIAL_BACKOFF
                logger.error(
                    "[Telegram] Bot failed after %.0fs uptime: %s — reconnecting in %.0fs",
                    uptime, e, backoff,
                )
                # Best-effort tear-down of the half-initialized Application
                # before retry; ignore errors since the connection may already
                # be unrecoverable.
                if self._app:
                    try:
                        await self._app.stop()
                        await self._app.shutdown()
                    except Exception:
                        pass
                    self._app = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

        # _stop_event tripped before entering the retry loop body
        if self._app:
            try:
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass

    async def close(self):
        """Gracefully close the Telegram bot."""
        self._stop_event.set()
