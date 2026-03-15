"""Telegram channel adapter — connects Nova_ to Telegram via python-telegram-bot."""

from __future__ import annotations

import asyncio
import collections
import logging

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import config
from app.schema import EventType

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot that calls think() directly for each user message."""

    def __init__(self):
        self.token = config.TELEGRAM_TOKEN
        self.default_chat_id = config.TELEGRAM_CHAT_ID
        self._allowed_users = self._parse_allowed_users()
        self._conversations: collections.OrderedDict[int, str] = collections.OrderedDict()  # telegram user_id → conv_id
        self._app: Application | None = None

    @staticmethod
    def _parse_allowed_users() -> set[int]:
        """Parse comma-separated user IDs from config."""
        raw = config.TELEGRAM_ALLOWED_USERS
        if not raw:
            return set()
        try:
            return {int(uid.strip()) for uid in raw.split(",") if uid.strip()}
        except ValueError:
            logger.warning("[Telegram] Invalid TELEGRAM_ALLOWED_USERS: %s", raw)
            return set()

    def _is_allowed(self, user_id: int) -> bool:
        """Check if user is in the allowlist. Empty list = allow all."""
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(
            "Hello! I'm Nova_, your personal AI assistant. Send me a message to get started."
        )

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://nova:{config.PORT}/api/health", timeout=5)
                data = resp.json()
                status = (
                    f"Status: {data.get('status', 'unknown')}\n"
                    f"Model: {data.get('model', 'unknown')}\n"
                    f"Ollama: {'connected' if data.get('ollama_connected') else 'disconnected'}\n"
                    f"DB: {'connected' if data.get('db_connected') else 'disconnected'}"
                )
        except httpx.ConnectError:
            status = "Could not reach health endpoint (connection refused)"
        except httpx.TimeoutException:
            status = "Could not reach health endpoint (timed out)"
        except Exception:
            status = "Could not reach health endpoint"
        await update.message.reply_text(status)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text(
            "Commands:\n"
            "/start — Introduction\n"
            "/status — Check system health\n"
            "/help — This message\n\n"
            "Just send any message to chat with me."
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages."""
        if not update.message or not update.message.text:
            return

        user_id = update.effective_user.id
        if not self._is_allowed(user_id):
            await update.message.reply_text("Sorry, you're not authorized to use this bot.")
            return

        query = update.message.text.strip()
        if not query:
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        answer = await self._handle_query(query, user_id)

        for chunk in self._split_message(answer):
            await update.message.reply_text(chunk)

    async def _handle_query(self, query: str, user_id: int) -> str:
        """Run query through think() and collect the response."""
        from app.core.brain import think, get_services

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
            async for event in think(query=query, conversation_id=conv_id, channel="telegram"):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
                elif event.type == EventType.ERROR:
                    return f"Error: {event.data.get('message', 'unknown error')}"

            answer = "".join(tokens).strip()
            return answer if answer else "I processed your message but had no response."

        except Exception as e:
            logger.error("[Telegram] Query failed: %s", e, exc_info=True)
            return "Sorry, something went wrong while processing your message."

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

    async def send_alert(self, message: str):
        """Send a message to the default chat."""
        if not self.default_chat_id or not self._app:
            return
        try:
            for chunk in self._split_message(message):
                await self._app.bot.send_message(chat_id=int(self.default_chat_id), text=chunk)
        except Exception as e:
            logger.error("[Telegram] Alert send failed: %s", e)

    async def start(self):
        """Start the Telegram bot (polling mode)."""
        if not self.token:
            logger.warning("[Telegram] No token configured, skipping")
            return

        try:
            self._app = Application.builder().token(self.token).build()
            self._app.add_handler(CommandHandler("start", self._handle_start))
            self._app.add_handler(CommandHandler("status", self._handle_status))
            self._app.add_handler(CommandHandler("help", self._handle_help))
            self._app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )

            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
            logger.info("[Telegram] Bot started, polling for messages")

            # Keep running until cancelled
            while True:
                await asyncio.sleep(3600)

        except asyncio.CancelledError:
            logger.info("[Telegram] Bot shutting down")
        except Exception as e:
            logger.error("[Telegram] Bot failed: %s", e, exc_info=True)
        finally:
            if self._app:
                try:
                    await self._app.updater.stop()
                    await self._app.stop()
                    await self._app.shutdown()
                except Exception:
                    pass

    async def close(self):
        """Gracefully close the Telegram bot."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass
