"""WhatsApp channel adapter — connects Nova to WhatsApp via Business API webhooks."""

from __future__ import annotations

import asyncio
import collections
import json
import logging

import hmac

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse

from app.config import config
from app.schema import EventType

logger = logging.getLogger(__name__)


class WhatsAppBot:
    """WhatsApp bot using the Business API (webhook-based)."""

    def __init__(self):
        self.api_url = config.WHATSAPP_API_URL
        self.api_token = config.WHATSAPP_API_TOKEN
        self.verify_token = config.WHATSAPP_VERIFY_TOKEN
        self.phone_number_id = config.WHATSAPP_PHONE_ID
        self.default_chat_id = config.WHATSAPP_CHAT_ID
        self._allowed_users = self._parse_allowed_users()
        self._conversations: collections.OrderedDict[str, str] = collections.OrderedDict()  # whatsapp phone → conv_id
        self._conv_store = None  # lazy-init DB store
        self._conv_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=30)
        self._dedup_db = None  # lazy-init SQLite dedup

    @staticmethod
    def _parse_allowed_users() -> set[str]:
        """Parse comma-separated phone numbers from config."""
        raw = config.WHATSAPP_ALLOWED_USERS
        if not raw:
            return set()
        try:
            return {uid.strip() for uid in raw.split(",") if uid.strip()}
        except Exception:
            logger.warning("[WhatsApp] Invalid WHATSAPP_ALLOWED_USERS: %s", raw)
            return set()

    def _is_allowed(self, phone: str) -> bool:
        """Check if phone is in the allowlist. Empty list = allow all."""
        if not self._allowed_users:
            return True
        return phone in self._allowed_users

    def _get_dedup_db(self):
        """Lazy-init SQLite dedup table."""
        if self._dedup_db is None:
            from app.database import get_db
            self._dedup_db = get_db()
            self._dedup_db.execute(
                "CREATE TABLE IF NOT EXISTS channel_dedup "
                "(msg_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
        return self._dedup_db

    def _check_dedup(self, msg_id: str) -> bool:
        """Return True if msg_id was already processed."""
        db = self._get_dedup_db()
        row = db.fetchone("SELECT 1 FROM channel_dedup WHERE msg_id = ?", (msg_id,))
        return row is not None

    def _record_dedup(self, msg_id: str) -> None:
        """Record a processed msg_id and clean up old entries."""
        db = self._get_dedup_db()
        db.execute("INSERT OR IGNORE INTO channel_dedup (msg_id) VALUES (?)", (msg_id,))
        # Periodic cleanup of entries older than 24 hours
        db.execute(
            "DELETE FROM channel_dedup WHERE created_at < datetime('now', '-24 hours')"
        )

    async def _handle_query(self, query: str, sender_phone: str) -> str:
        """Run query through think() and collect the response."""
        from app.core.brain import think, get_services

        # Get or create conversation for this user
        async with self._conv_lock:
            conv_id = self._conversations.get(sender_phone)
            if conv_id:
                self._conversations.move_to_end(sender_phone)
            else:
                # Try DB recovery
                if self._conv_store is None:
                    from app.database import get_db, ChannelConversationStore
                    self._conv_store = ChannelConversationStore(get_db())
                conv_id = self._conv_store.get("whatsapp", str(sender_phone))
                if not conv_id:
                    svc = get_services()
                    conv_id = svc.conversations.create_conversation()
                    self._conv_store.set("whatsapp", str(sender_phone), conv_id)
                self._conversations[sender_phone] = conv_id
                while len(self._conversations) > 1000:  # LRU cap for personal bot
                    self._conversations.popitem(last=False)

        try:
            tokens = []
            async for event in think(query=query, conversation_id=conv_id, channel="whatsapp"):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
                elif event.type == EventType.ERROR:
                    return f"Error: {event.data.get('message', 'unknown error')}"

            answer = "".join(tokens).strip()
            return answer if answer else "I processed your message but had no response."

        except Exception as e:
            logger.error("[WhatsApp] Query failed: %s", e, exc_info=True)
            return "Sorry, something went wrong while processing your message."

    @staticmethod
    def _split_message(text: str, limit: int = 4096) -> list[str]:
        """Split a message into chunks that fit WhatsApp's character limit."""
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

    async def _send_message(self, to: str, text: str) -> None:
        """Send a text message via the WhatsApp Business API."""
        if not self.api_url or not self.api_token:
            logger.warning("[WhatsApp] API URL or token not configured, cannot send")
            return

        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }

        try:
            resp = await self._client.post(
                self.api_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_token}"},
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    "[WhatsApp] Send failed (status=%d): %s",
                    resp.status_code, resp.text[:500],
                )
        except Exception as e:
            logger.error("[WhatsApp] Send failed: %s", e)

    async def send_alert(self, message: str) -> None:
        """Send a message to the default chat."""
        if not self.default_chat_id:
            return
        try:
            for chunk in self._split_message(message):
                await self._send_message(self.default_chat_id, chunk)
        except Exception as e:
            logger.error("[WhatsApp] Alert send failed: %s", e)

    def get_router(self) -> APIRouter:
        """Return a FastAPI router with webhook endpoints."""
        router = APIRouter(prefix="/api/channels/whatsapp", tags=["whatsapp"])

        @router.get("/webhook")
        async def verify_webhook(
            hub_mode: str = Query(None, alias="hub.mode"),
            hub_verify_token: str = Query(None, alias="hub.verify_token"),
            hub_challenge: str = Query(None, alias="hub.challenge"),
        ):
            """WhatsApp webhook verification (hub challenge)."""
            if not self.verify_token:
                return PlainTextResponse("Verify token not configured", status_code=403)
            if hub_mode == "subscribe" and hmac.compare_digest(hub_verify_token or "", self.verify_token):
                logger.info("[WhatsApp] Webhook verified")
                return PlainTextResponse(hub_challenge)
            logger.warning("[WhatsApp] Webhook verification failed")
            return PlainTextResponse("Verification failed", status_code=403)

        @router.post("/webhook")
        async def receive_message(request: Request):
            """Handle incoming WhatsApp messages."""
            # Fail-closed: reject all webhooks if app secret is not configured
            if not config.WHATSAPP_APP_SECRET:
                return JSONResponse({"status": "webhook not configured"}, status_code=503)

            # HMAC-SHA256 signature verification
            import hashlib
            import hmac as _hmac
            raw_body = await request.body()
            signature = request.headers.get("x-hub-signature-256", "")
            if not signature:
                return JSONResponse({"status": "missing signature"}, status_code=403)
            expected = "sha256=" + _hmac.new(
                config.WHATSAPP_APP_SECRET.encode(),
                raw_body,
                hashlib.sha256,
            ).hexdigest()
            if not _hmac.compare_digest(signature, expected):
                logger.warning("[WhatsApp] Invalid webhook signature")
                return JSONResponse({"status": "invalid signature"}, status_code=403)
            try:
                body = json.loads(raw_body)
            except Exception:
                return JSONResponse({"status": "invalid body"}, status_code=400)

            # Parse the nested WhatsApp webhook structure
            try:
                entry = body.get("entry", [])
                if not entry:
                    return JSONResponse({"status": "ok"})

                changes = entry[0].get("changes", [])
                if not changes:
                    return JSONResponse({"status": "ok"})

                value = changes[0].get("value", {})
                messages = value.get("messages", [])
                if not messages:
                    return JSONResponse({"status": "ok"})

                msg = messages[0]
                msg_id = msg.get("id", "")
                msg_type = msg.get("type", "")
                sender_phone = msg.get("from", "")

                # Dedup — WhatsApp may send the same webhook multiple times (persistent via SQLite)
                if self._check_dedup(msg_id):
                    return JSONResponse({"status": "ok"})
                self._record_dedup(msg_id)

                # Only handle text messages
                if msg_type != "text":
                    return JSONResponse({"status": "ok"})

                text_body = msg.get("text", {}).get("body", "").strip()
                if not text_body:
                    return JSONResponse({"status": "ok"})

                # Allowlist check
                if not self._is_allowed(sender_phone):
                    await self._send_message(
                        sender_phone,
                        "Sorry, you're not authorized to use this bot.",
                    )
                    return JSONResponse({"status": "ok"})

                # Process the message
                answer = await self._handle_query(text_body, sender_phone)

                for chunk in self._split_message(answer):
                    await self._send_message(sender_phone, chunk)

            except Exception as e:
                logger.error("[WhatsApp] Webhook handler error: %s", e, exc_info=True)

            return JSONResponse({"status": "ok"})

        return router

    async def start(self) -> None:
        """No-op — webhook-based adapter, no polling needed."""
        logger.info("[WhatsApp] Bot ready (webhook mode)")

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
