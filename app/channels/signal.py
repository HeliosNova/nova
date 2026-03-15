"""Signal channel adapter — connects Nova_ to Signal via signal-cli REST API."""

from __future__ import annotations

import asyncio
import collections
import logging

import httpx

from app.config import config
from app.schema import EventType

logger = logging.getLogger(__name__)


class SignalBot:
    """Signal bot that polls signal-cli REST API and calls think() for each message."""

    def __init__(self):
        self.api_url = config.SIGNAL_API_URL.rstrip("/")
        self.phone_number = config.SIGNAL_PHONE_NUMBER
        self.default_recipient = config.SIGNAL_CHAT_ID
        self._allowed_users = self._parse_allowed_users()
        self._conversations: collections.OrderedDict[str, str] = collections.OrderedDict()  # sender phone → conv_id
        self._processed_ids: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._running = False
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _parse_allowed_users() -> set[str]:
        """Parse comma-separated phone numbers from config."""
        raw = config.SIGNAL_ALLOWED_USERS
        if not raw:
            return set()
        try:
            return {uid.strip() for uid in raw.split(",") if uid.strip()}
        except Exception:
            logger.warning("[Signal] Invalid SIGNAL_ALLOWED_USERS: %s", raw)
            return set()

    def _is_allowed(self, sender: str) -> bool:
        """Check if sender is in the allowlist. Empty list = allow all."""
        if not self._allowed_users:
            return True
        return sender in self._allowed_users

    async def _poll_messages(self) -> None:
        """Poll signal-cli REST API for incoming messages."""
        while self._running:
            try:
                resp = await self._client.get(
                    f"{self.api_url}/v1/receive/{self.phone_number}",
                    timeout=30,
                )
                if resp.status_code == 200:
                    messages = resp.json()
                    for msg in messages:
                        await self._process_envelope(msg)
                elif resp.status_code != 204:
                    logger.warning("[Signal] Poll returned status %d", resp.status_code)
            except httpx.ConnectError:
                logger.warning("[Signal] Cannot reach signal-cli at %s (retrying)", self.api_url)
            except httpx.TimeoutException:
                pass  # Normal on long poll with no messages
            except Exception as e:
                logger.error("[Signal] Poll error: %s", e, exc_info=True)

            await asyncio.sleep(config.SIGNAL_POLL_INTERVAL)

    async def _process_envelope(self, msg: dict) -> None:
        """Process a single message envelope from signal-cli."""
        try:
            envelope = msg.get("envelope", {})
            source = envelope.get("sourceNumber") or envelope.get("source", "")
            data_message = envelope.get("dataMessage")

            if not data_message or not source:
                return

            text = data_message.get("message", "")
            if not text or not text.strip():
                return

            # Dedup using timestamp + source
            timestamp = str(data_message.get("timestamp", ""))
            dedup_key = f"{timestamp}:{source}"
            if dedup_key in self._processed_ids:
                return
            self._processed_ids[dedup_key] = None
            while len(self._processed_ids) > 10000:
                self._processed_ids.popitem(last=False)

            if not self._is_allowed(source):
                await self._send_message(source, "Sorry, you're not authorized to use this bot.")
                return

            query = text.strip()
            answer = await self._handle_query(query, source)

            for chunk in self._split_message(answer):
                await self._send_message(source, chunk)

        except Exception as e:
            logger.error("[Signal] Error processing message: %s", e, exc_info=True)

    async def _handle_query(self, query: str, sender: str) -> str:
        """Run query through think() and collect the response."""
        from app.core.brain import think, get_services

        # Get or create conversation for this sender
        conv_id = self._conversations.get(sender)
        if conv_id:
            # Move to end so it's marked as recently used
            self._conversations.move_to_end(sender)
        else:
            svc = get_services()
            conv_id = svc.conversations.create_conversation()
            self._conversations[sender] = conv_id
            # LRU eviction: cap at 1000 entries
            while len(self._conversations) > 1000:
                self._conversations.popitem(last=False)

        try:
            tokens = []
            async for event in think(query=query, conversation_id=conv_id, channel="signal"):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
                elif event.type == EventType.ERROR:
                    return f"Error: {event.data.get('message', 'unknown error')}"

            answer = "".join(tokens).strip()
            return answer if answer else "I processed your message but had no response."

        except Exception as e:
            logger.error("[Signal] Query failed: %s", e, exc_info=True)
            return "Sorry, something went wrong while processing your message."

    @staticmethod
    def _split_message(text: str, limit: int = 4096) -> list[str]:
        """Split a message into chunks that fit Signal's practical character limit."""
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

    async def _send_message(self, recipient: str, text: str) -> None:
        """Send a message via signal-cli REST API."""
        try:
            resp = await self._client.post(
                f"{self.api_url}/v2/send",
                json={
                    "message": text,
                    "number": self.phone_number,
                    "recipients": [recipient],
                },
                timeout=30,
            )
            if resp.status_code not in (200, 201):
                logger.warning("[Signal] Send returned status %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("[Signal] Send failed to %s: %s", recipient, e)

    async def send_alert(self, message: str) -> None:
        """Send a message to the default recipient."""
        if not self.default_recipient or not self._client:
            return
        try:
            for chunk in self._split_message(message):
                await self._send_message(self.default_recipient, chunk)
        except Exception as e:
            logger.error("[Signal] Alert send failed: %s", e)

    async def start(self) -> None:
        """Start the Signal bot (polling mode, blocks until cancelled/closed)."""
        if not self.api_url or not self.phone_number:
            logger.warning("[Signal] No API URL or phone number configured, skipping")
            return

        try:
            self._client = httpx.AsyncClient()
            self._running = True
            logger.info(
                "[Signal] Bot started, polling %s every %ds",
                self.api_url, config.SIGNAL_POLL_INTERVAL,
            )
            await self._poll_messages()

        except asyncio.CancelledError:
            logger.info("[Signal] Bot shutting down")
        except Exception as e:
            logger.error("[Signal] Bot failed: %s", e, exc_info=True)
        finally:
            self._running = False
            if self._client:
                await self._client.aclose()
                self._client = None

    async def close(self) -> None:
        """Gracefully close the Signal bot."""
        self._running = False
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
