"""Base channel class — shared logic for all channel adapters.

Extracts common patterns: query handling via brain.think() and message splitting.
"""

from __future__ import annotations

import logging

from app.schema import EventType

logger = logging.getLogger(__name__)


class BaseChannel:
    """Shared base for channel adapters (Discord, Telegram, Signal, WhatsApp)."""

    async def _handle_query(self, query: str, user_id: str, conversation_id: str | None = None) -> str:
        """Run query through think() and collect the response.

        Args:
            query: The user's message text.
            user_id: Channel-specific user identifier.
            conversation_id: Optional existing conversation ID.

        Returns:
            The assistant's response text.
        """
        from app.core.brain import think

        try:
            tokens = []
            async for event in think(query=query, conversation_id=conversation_id):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        tokens.append(text)
                elif event.type == EventType.ERROR:
                    return f"Error: {event.data.get('message', 'unknown error')}"

            answer = "".join(tokens).strip()
            return answer if answer else "I processed your message but had no response."

        except Exception as e:
            logger.error("[%s] Query failed: %s", self.__class__.__name__, e, exc_info=True)
            return "Sorry, something went wrong while processing your message."

    @staticmethod
    def _split_message(text: str, max_length: int = 2000) -> list[str]:
        """Split long messages at word boundaries.

        Args:
            text: The text to split.
            max_length: Maximum length per chunk.

        Returns:
            List of text chunks, each within max_length.
        """
        if len(text) <= max_length:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            # Try to split at a newline first
            split_at = text.rfind("\n", 0, max_length)
            if split_at == -1:
                # Fall back to word boundary
                split_at = text.rfind(" ", 0, max_length)
            if split_at == -1:
                # Hard split as last resort
                split_at = max_length
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        return chunks
