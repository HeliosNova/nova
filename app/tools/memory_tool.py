"""Memory tool — search past conversations and manage user facts."""

from __future__ import annotations

import logging

from app.core.text_utils import normalize_words, STOP_WORDS
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = "Search past conversations and archival memory."
    parameters = "query: str"

    def __init__(self, conversations=None, user_facts=None):
        self._conversations = conversations
        self._user_facts = user_facts

    async def execute(self, *, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(output="", success=False, error="No query provided")

        results = []

        # Search user facts using word-overlap matching
        if self._user_facts:
            facts = self._user_facts.get_all()
            query_words = normalize_words(query)
            non_stop_query = query_words - STOP_WORDS
            # If all query words are stop words (e.g. "what do you know about me"), return all facts
            return_all = len(non_stop_query) == 0
            matched_facts = []
            for f in facts:
                if return_all:
                    matched_facts.append(f)
                else:
                    fact_words = normalize_words(f.key + " " + f.value) - STOP_WORDS
                    if non_stop_query & fact_words:
                        matched_facts.append(f)
            # Fallback: substring matching if word overlap found nothing
            if not matched_facts and not return_all:
                query_lower = query.lower()
                for f in facts:
                    combined = (f.key + " " + f.value).lower()
                    if query_lower in combined or any(w in combined for w in non_stop_query):
                        matched_facts.append(f)
            if matched_facts:
                results.append("## Matching User Facts")
                for f in matched_facts:
                    results.append(f"- {f.key}: {f.value}")

        # Search conversation messages by content
        if self._conversations:
            matches = self._conversations.search_messages(query, limit=10)
            if matches:
                results.append("\n## Matching Conversations")
                for m in matches:
                    title = m.get("conversation_title", "Untitled")
                    role = m["role"]
                    content = m["content"][:300]
                    results.append(f"\n**{title}** [{role}]:\n{content}")

        if not results:
            return ToolResult(
                output="No matching memories found.",
                success=True,
            )

        return ToolResult(output="\n".join(results), success=True)
