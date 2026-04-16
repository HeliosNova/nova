"""Active Memory Tool — lets Nova deliberately manage its own memory.

AgeMem pattern: memory operations exposed as callable tools during conversation.
The agent actively decides what to remember, forget, search, and summarize.
Separate from memory_search (read-only conversation search).

References: AgeMem (arXiv 2601.01885), Letta/MemGPT memory blocks
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import config
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

MAX_ACTIVE_MEMORIES = 500


class ActiveMemoryTool(BaseTool):
    name = "active_memory"
    description = (
        "Manage your long-term memory. Use this to deliberately store important information, "
        "search for past memories, update or delete entries, and summarize related memories. "
        "Actions: add, search, update, delete, list, summarize."
    )
    parameters = "action: str, content: str, category: str, id: int, query: str"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "update", "delete", "list", "summarize"],
                "description": "Memory action to perform.",
            },
            "content": {
                "type": "string",
                "description": "Memory content (for add/update).",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "preference", "decision", "pattern", "correction"],
                "description": "Category of memory (for add). Defaults to 'fact'.",
            },
            "id": {
                "type": "integer",
                "description": "Memory ID (for update/delete).",
            },
            "query": {
                "type": "string",
                "description": "Search query (for search/summarize).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, db: Any = None):
        self._db = db
        self._collection = None
        self._ensure_table()

    def _ensure_table(self):
        if self._db is None:
            return
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS active_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'fact',
                tags TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                last_accessed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                self._collection = client.get_or_create_collection(
                    name="active_memory",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("Failed to init active_memory ChromaDB collection: %s", e)
                return None
        return self._collection

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "").lower()

        if self._db is None:
            return ToolResult(output="", success=False, error="Memory system not initialized",
                              error_category=ErrorCategory.INTERNAL)

        if action == "add":
            return self._add(kwargs)
        elif action == "search":
            return self._search(kwargs)
        elif action == "update":
            return self._update(kwargs)
        elif action == "delete":
            return self._delete(kwargs)
        elif action == "list":
            return self._list(kwargs)
        elif action == "summarize":
            return self._summarize(kwargs)
        else:
            return ToolResult(
                output="",
                success=False,
                error=f"Unknown action '{action}'. Use: add, search, update, delete, list, summarize",
                error_category=ErrorCategory.VALIDATION,
            )

    def _add(self, kwargs: dict) -> ToolResult:
        content = kwargs.get("content", "").strip()
        if not content:
            return ToolResult(output="", success=False, error="No content provided",
                              error_category=ErrorCategory.VALIDATION)

        category = kwargs.get("category", "fact")
        now = datetime.now(timezone.utc).isoformat()

        # Check limit
        count = self._db.fetchone("SELECT COUNT(*) as c FROM active_memories")
        if count and count["c"] >= MAX_ACTIVE_MEMORIES:
            return ToolResult(output="", success=False,
                              error=f"Memory limit reached ({MAX_ACTIVE_MEMORIES}). Delete old memories first.",
                              error_category=ErrorCategory.VALIDATION)

        self._db.execute(
            "INSERT INTO active_memories (content, category, last_accessed_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (content, category, now, now, now),
        )
        row = self._db.fetchone("SELECT id FROM active_memories ORDER BY id DESC LIMIT 1")
        mem_id = row["id"] if row else 0

        # Add to vector store
        collection = self._get_collection()
        if collection:
            try:
                collection.upsert(
                    ids=[str(mem_id)],
                    documents=[content],
                    metadatas=[{"category": category}],
                )
            except Exception:
                pass

        return ToolResult(output=f"Memory stored (id={mem_id}, category={category}): {content[:100]}", success=True)

    def _search(self, kwargs: dict) -> ToolResult:
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolResult(output="", success=False, error="No search query provided",
                              error_category=ErrorCategory.VALIDATION)

        results = []

        # Vector search
        collection = self._get_collection()
        if collection and collection.count() > 0:
            try:
                res = collection.query(query_texts=[query], n_results=min(5, collection.count()))
                if res and res["ids"] and res["ids"][0]:
                    for rid, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
                        score = 1.0 - (dist / 2.0)
                        row = self._db.fetchone("SELECT * FROM active_memories WHERE id = ?", (int(rid),))
                        if row:
                            results.append((score, row))
                            self._db.execute(
                                "UPDATE active_memories SET access_count = access_count + 1, "
                                "last_accessed_at = datetime('now') WHERE id = ?",
                                (int(rid),),
                            )
            except Exception as e:
                logger.debug("Active memory vector search failed: %s", e)

        # Keyword fallback
        if not results:
            rows = self._db.fetchall(
                "SELECT * FROM active_memories ORDER BY created_at DESC LIMIT 50"
            )
            query_words = set(query.lower().split())
            for row in rows:
                content_words = set(row["content"].lower().split())
                overlap = len(query_words & content_words)
                if overlap >= 2:
                    results.append((overlap / 10.0, row))

        if not results:
            return ToolResult(output="No memories found matching that query.", success=True)

        results.sort(key=lambda x: -x[0])
        lines = []
        for score, row in results[:5]:
            lines.append(f"[{row['id']}] ({row['category']}) {row['content'][:200]}")

        return ToolResult(output="\n".join(lines), success=True)

    def _update(self, kwargs: dict) -> ToolResult:
        mem_id = kwargs.get("id")
        content = kwargs.get("content", "").strip()
        if not mem_id:
            return ToolResult(output="", success=False, error="No memory ID provided",
                              error_category=ErrorCategory.VALIDATION)
        if not content:
            return ToolResult(output="", success=False, error="No new content provided",
                              error_category=ErrorCategory.VALIDATION)

        existing = self._db.fetchone("SELECT id FROM active_memories WHERE id = ?", (mem_id,))
        if not existing:
            return ToolResult(output="", success=False, error=f"Memory {mem_id} not found",
                              error_category=ErrorCategory.NOT_FOUND)

        self._db.execute(
            "UPDATE active_memories SET content = ?, updated_at = datetime('now') WHERE id = ?",
            (content, mem_id),
        )

        collection = self._get_collection()
        if collection:
            try:
                collection.upsert(ids=[str(mem_id)], documents=[content])
            except Exception:
                pass

        return ToolResult(output=f"Memory {mem_id} updated.", success=True)

    def _delete(self, kwargs: dict) -> ToolResult:
        mem_id = kwargs.get("id")
        if not mem_id:
            return ToolResult(output="", success=False, error="No memory ID provided",
                              error_category=ErrorCategory.VALIDATION)

        existing = self._db.fetchone("SELECT id FROM active_memories WHERE id = ?", (mem_id,))
        if not existing:
            return ToolResult(output="", success=False, error=f"Memory {mem_id} not found",
                              error_category=ErrorCategory.NOT_FOUND)

        self._db.execute("DELETE FROM active_memories WHERE id = ?", (mem_id,))

        collection = self._get_collection()
        if collection:
            try:
                collection.delete(ids=[str(mem_id)])
            except Exception:
                pass

        return ToolResult(output=f"Memory {mem_id} deleted.", success=True)

    def _list(self, kwargs: dict) -> ToolResult:
        category = kwargs.get("category")
        if category:
            rows = self._db.fetchall(
                "SELECT * FROM active_memories WHERE category = ? ORDER BY created_at DESC LIMIT 20",
                (category,),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM active_memories ORDER BY created_at DESC LIMIT 20"
            )

        if not rows:
            return ToolResult(output="No memories stored.", success=True)

        lines = []
        for row in rows:
            lines.append(f"[{row['id']}] ({row['category']}) {row['content'][:150]}")

        total = self._db.fetchone("SELECT COUNT(*) as c FROM active_memories")
        lines.append(f"\nTotal memories: {total['c'] if total else len(rows)}")

        return ToolResult(output="\n".join(lines), success=True)

    def _summarize(self, kwargs: dict) -> ToolResult:
        query = kwargs.get("query", "").strip()
        # First search for related memories
        search_result = self._search({"query": query or "all"})
        if not search_result.success or search_result.output == "No memories found matching that query.":
            return ToolResult(output="No memories to summarize.", success=True)

        return ToolResult(
            output=f"Related memories:\n{search_result.output}\n\nSummarize these memories to answer the query.",
            success=True,
        )
