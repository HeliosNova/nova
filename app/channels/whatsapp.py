"""WhatsApp channel adapter — connects Nova_ to WhatsApp via Business API webhooks."""

from __future__ import annotations

import collections
import logging

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
        self._client = httpx.AsyncClient(timeout=30)
        self._processed_ids: collections.OrderedDict[str, None] = collections.OrderedDict()  # dedup

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

    async def _handle_query(self, query: str, sender_phone: str) -> str:
        """Run query through think() and collect the response."""
        from app.core.brain import think, get_services

        # Get or create conversation for this user
        conv_id = self._conversations.get(sender_phone)
        if conv_id:
            # Move to end so it's marked as recently used
            self._conversations.move_to_end(sender_phone)
        else:
            svc = get_services()
            conv_id = svc.conversations.create_conversation()
            self._conversations[sender_phone] = conv_id
            # LRU eviction: cap at 1000 entries
            while len(self._conversations) > 1000:
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
            if hub_mode == "subscribe" and hub_verify_token == self.verify_token:
                logger.info("[WhatsApp] Webhook verified")
                return PlainTextResponse(hub_challenge)
            logger.warning("[WhatsApp] Webhook verification failed")
            return PlainTextResponse("Verification failed", status_code=403)

        @router.post("/webhook")
        async def receive_message(request: Request):
            """Handle incoming WhatsApp messages."""
            try:
                body = await request.json()
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

                # Dedup — WhatsApp may send the same webhook multiple times
                if msg_id in self._processed_ids:
                    return JSONResponse({"status": "ok"})
                self._processed_ids[msg_id] = None

                # Cap dedup dict to prevent unbounded growth — pop oldest entries
                while len(self._processed_ids) > 10000:
                    self._processed_ids.popitem(last=False)

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
            await self._client.aclose()
