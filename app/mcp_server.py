"""Nova MCP Server — expose Nova's intelligence as MCP tools.

Allows external MCP clients (Cursor, VS Code, and others) to query Nova's
long-term memory, knowledge graph, lessons, and document store.

This module defines the MCP server and its tools. It does NOT start the
server — that's done by scripts/mcp_server_runner.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import Resource, TextContent, Tool, CallToolResult

from app.config import config
from app.core.kg import KnowledgeGraph
from app.core.learning import LearningEngine
from app.core.memory import ConversationStore, UserFactStore
from app.core.retriever import Retriever
from app.database import SafeDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for each tool's inputSchema)
# ---------------------------------------------------------------------------

_TOOLS = [
    Tool(
        name="nova_memory_query",
        description=(
            "Search Nova's long-term memory for user facts and past conversation excerpts. "
            "Returns matching facts (key, value, category, confidence) and conversation snippets "
            "ranked by keyword relevance. Use for: recalling user preferences, finding past "
            "discussions, checking what Nova knows about the user. Prefer nova_document_search "
            "for ingested document content. Prefer nova_knowledge_graph for structured entity "
            "relationships. Limit default: 5, max: 20."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for memory lookup",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_knowledge_graph",
        description=(
            "Query Nova's temporal knowledge graph for structured facts about entities and "
            "their relationships. Returns subject-predicate-object triples with confidence "
            "scores, source provenance, and traversal depth. Use for: looking up entity facts, "
            "exploring connections between concepts, checking what Nova has learned from research. "
            "Set hops=2 to include neighbors of neighbors. Prefer nova_memory_query for "
            "user-specific facts. Prefer nova_document_search for full-text document content."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity name to look up in the knowledge graph",
                },
                "hops": {
                    "type": "integer",
                    "description": "Number of hops for graph traversal (1=direct, 2=neighbors of neighbors)",
                    "default": 1,
                },
            },
            "required": ["entity"],
        },
    ),
    Tool(
        name="nova_lessons",
        description=(
            "Retrieve lessons Nova has learned from user corrections, relevant to a query topic. "
            "Returns lesson details including topic, correct/wrong answers, lesson text, "
            "confidence, and retrieval/helpfulness counts. Use for: understanding how Nova was "
            "corrected on similar topics, checking if Nova has learned from past mistakes. "
            "Limit default: 5."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or question to find relevant lessons for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lessons to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_document_search",
        description=(
            "Search Nova's ingested documents using hybrid retrieval (vector similarity + BM25 "
            "keyword matching with Reciprocal Rank Fusion). Returns ranked document chunks with "
            "content, relevance score, source, and title. Use for: finding information in "
            "uploaded files and documents. Prefer nova_memory_query for user facts and "
            "conversation history. Top_k default: 5."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for document retrieval",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of document chunks to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_facts_about",
        description=(
            "Get all stored facts about the user/owner, optionally filtered by category. "
            "Returns facts with key, value, category (fact/preference/capability/constraint), "
            "and confidence score. Use for: getting a complete picture of known user attributes. "
            "Omit category parameter to retrieve all facts across all categories."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category: fact, preference, capability, or constraint. Omit for all.",
                    "enum": ["fact", "preference", "capability", "constraint"],
                },
            },
            "required": [],
        },
    ),
    # Knowing tier (2026-09-02): the durable understanding, not just the facts.
    Tool(
        name="nova_dossiers",
        description=(
            "Search Nova's living dossiers — its standing, revisable understanding of a "
            "domain, entity or story thread, distilled from verified intelligence digests. "
            "Returns the best-matching dossiers with their 'Current understanding' and "
            "'Open questions' excerpts. Use for: what does Nova currently understand about X, "
            "what does it still not know. Prefer nova_knowledge_graph for single facts. "
            "Limit default: 2, max: 5."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic, entity or question"},
                "limit": {"type": "integer", "description": "Max dossiers", "default": 2},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_storylines",
        description=(
            "Active story threads Nova is tracking that match a query: title, current "
            "state, how many times the thread moved, last update. Use for: what is the "
            "current state of an ongoing situation. Limit default: 3, max: 10."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Story topic or entity"},
                "limit": {"type": "integer", "description": "Max threads", "default": 3},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_forecasts",
        description=(
            "Nova's falsifiable forecasts and its self-graded track record. Filter by "
            "status (open, hit, miss, unresolvable, restated) and an optional claim "
            "substring. Returns the forecasts plus accuracy and calibration. "
            "Limit default: 10, max: 50."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "hit", "miss", "unresolvable", "restated"]},
                "query": {"type": "string", "description": "Substring the claim must contain"},
                "limit": {"type": "integer", "description": "Max forecasts", "default": 10},
            },
            "required": [],
        },
    ),
]


# ---------------------------------------------------------------------------
# Structured error helper
# ---------------------------------------------------------------------------


def _mcp_error(message: str, category: str, is_retryable: bool = False) -> CallToolResult:
    """Return a structured MCP error response with isError=True."""
    return CallToolResult(
        content=[TextContent(
            type="text",
            text=json.dumps({
                "error": message,
                "error_category": category,
                "is_retryable": is_retryable,
            }),
        )],
        isError=True,
    )


# ---------------------------------------------------------------------------
# MCP Server factory
# ---------------------------------------------------------------------------

def create_mcp_server(
    db: SafeDB,
    *,
    user_facts: UserFactStore | None = None,
    conversations: ConversationStore | None = None,
    learning: LearningEngine | None = None,
    kg: KnowledgeGraph | None = None,
    retriever: Retriever | None = None,
) -> Server:
    """Create and configure a Nova MCP server with all tool handlers.

    Callers must provide a SafeDB instance (already init_schema'd).
    Service instances are created lazily if not provided.
    """
    server = Server(config.MCP_SERVER_NAME)

    # Lazily build services from the db if not injected
    _user_facts = user_facts or UserFactStore(db)
    _conversations = conversations or ConversationStore(db)
    _learning = learning or LearningEngine(db)
    _kg = kg or KnowledgeGraph(db)

    # Retriever may fail (ChromaDB not available) — that's OK
    _retriever = retriever
    if _retriever is None:
        try:
            _retriever = Retriever(db)
        except Exception as e:
            logger.warning("Retriever unavailable (document search disabled): %s", e)

    # ------------------------------------------------------------------
    # list_tools handler
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Knowing-tier handlers (2026-09-02)
    # ------------------------------------------------------------------

    def _clip(rows: list[dict], cap: int = 2500) -> list[dict]:
        return [{k: (v[:cap] if isinstance(v, str) else v) for k, v in dict(r).items()} for r in rows]

    async def _handle_dossiers(arguments: dict[str, Any]):
        import asyncio as _aio
        query = str(arguments.get("query", "") or "").strip()
        if not query:
            return _mcp_error("query is required", "invalid_argument", False)
        limit = max(1, min(int(arguments.get("limit", 2) or 2), 5))
        from app.core.dossiers import get_relevant_dossiers
        rows = await _aio.to_thread(get_relevant_dossiers, db, query, limit=limit, open_questions=True)
        return [TextContent(type="text", text=json.dumps(
            {"query": query, "dossiers": _clip(rows)}, default=str))]

    async def _handle_storylines(arguments: dict[str, Any]):
        import asyncio as _aio
        query = str(arguments.get("query", "") or "").strip()
        if not query:
            return _mcp_error("query is required", "invalid_argument", False)
        limit = max(1, min(int(arguments.get("limit", 3) or 3), 10))
        from app.core.storylines import get_relevant_storylines
        rows = await _aio.to_thread(get_relevant_storylines, db, query, limit=limit)
        return [TextContent(type="text", text=json.dumps(
            {"query": query, "storylines": _clip(rows, 1500)}, default=str))]

    async def _handle_forecasts(arguments: dict[str, Any]):
        import asyncio as _aio
        status = str(arguments.get("status", "") or "").strip().lower() or None
        if status is not None and status not in ("open", "hit", "miss", "unresolvable", "restated"):
            return _mcp_error("status must be open|hit|miss|unresolvable|restated", "invalid_argument", False)
        needle = str(arguments.get("query", "") or "").strip()
        limit = max(1, min(int(arguments.get("limit", 10) or 10), 50))

        def _rows():
            clauses, params = [], []
            if status:
                clauses.append("status = ?")
                params.append(status)
            if needle:
                clauses.append("claim LIKE ?")
                params.append(f"%{needle}%")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            out = db.fetchall(
                f"SELECT id, claim, confidence, status, resolves_at, resolved_at, resolution, "
                f"source_monitor, created_at FROM forecasts {where} "
                f"ORDER BY created_at DESC LIMIT ?", (*params, limit))
            from app.core.forecasts import accuracy, calibration
            return [dict(r) for r in out], accuracy(db), calibration(db, min_n=5)

        rows, acc, cal = await _aio.to_thread(_rows)
        return [TextContent(type="text", text=json.dumps(
            {"forecasts": rows, "track_record": acc, "calibration": cal}, default=str))]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _TOOLS

    # ------------------------------------------------------------------
    # call_tool handler
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        try:
            if name == "nova_memory_query":
                content = await _handle_memory_query(arguments)
            elif name == "nova_knowledge_graph":
                content = await _handle_knowledge_graph(arguments)
            elif name == "nova_lessons":
                content = await _handle_lessons(arguments)
            elif name == "nova_document_search":
                content = await _handle_document_search(arguments)
            elif name == "nova_facts_about":
                content = await _handle_facts_about(arguments)
            elif name == "nova_dossiers":
                content = await _handle_dossiers(arguments)
            elif name == "nova_storylines":
                content = await _handle_storylines(arguments)
            elif name == "nova_forecasts":
                content = await _handle_forecasts(arguments)
            else:
                return _mcp_error(f"Unknown tool: {name}", "not_found", False)
            # Wrap list[TextContent] in CallToolResult for consistent return type
            if isinstance(content, list):
                return CallToolResult(content=content, isError=False)
            return content  # Already a CallToolResult (from _mcp_error)
        except Exception as e:
            logger.exception("MCP tool '%s' failed", name)
            return _mcp_error(str(e), "internal", True)

    # ------------------------------------------------------------------
    # Resource handlers
    # ------------------------------------------------------------------

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="nova://facts",
                name="User Facts",
                description="User fact categories and counts",
                mimeType="application/json",
            ),
            Resource(
                uri="nova://lessons",
                name="Learned Lessons",
                description="Lesson topic index and counts",
                mimeType="application/json",
            ),
            Resource(
                uri="nova://documents",
                name="Ingested Documents",
                description="Document metadata catalog",
                mimeType="application/json",
            ),
            Resource(
                uri="nova://knowledge-graph/entities",
                name="Knowledge Graph Entities",
                description="Top entities and their fact counts",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri) -> list[ReadResourceContents]:
        uri_str = str(uri)

        if uri_str == "nova://facts":
            all_facts = _user_facts.get_all()
            categories: dict[str, int] = {}
            for f in all_facts:
                categories[f.category] = categories.get(f.category, 0) + 1
            return [ReadResourceContents(
                content=json.dumps({"total": len(all_facts), "by_category": categories}),
                mime_type="application/json",
            )]

        elif uri_str == "nova://lessons":
            lessons = _learning.get_all_lessons(limit=100)
            topics = [lesson.topic for lesson in lessons]
            return [ReadResourceContents(
                content=json.dumps({"total": len(topics), "topics": topics[:50]}),
                mime_type="application/json",
            )]

        elif uri_str == "nova://documents":
            if _retriever is None:
                return [ReadResourceContents(
                    content=json.dumps({"error": "Document store unavailable", "total": 0}),
                    mime_type="application/json",
                )]
            try:
                docs = _retriever.list_documents(limit=50)
                return [ReadResourceContents(
                    content=json.dumps({
                        "total": len(docs),
                        "documents": [
                            {"id": d.get("id", ""), "title": d.get("title", ""), "source": d.get("source", "")}
                            for d in docs
                        ],
                        "note": "Use nova_document_search to query content",
                    }),
                    mime_type="application/json",
                )]
            except Exception:
                return [ReadResourceContents(
                    content=json.dumps({"total": 0, "note": "Document metadata unavailable"}),
                    mime_type="application/json",
                )]

        elif uri_str == "nova://knowledge-graph/entities":
            try:
                top_entities = _kg.get_top_entities(limit=50)
                entity_counts = {r["subject"]: r["cnt"] for r in top_entities}
                stats = _kg.get_stats()
                return [ReadResourceContents(
                    content=json.dumps({
                        "total": stats.get("unique_entities", 0),
                        "top_entities": entity_counts,
                    }),
                    mime_type="application/json",
                )]
            except Exception:
                return [ReadResourceContents(
                    content=json.dumps({"total": 0, "entities": {}}),
                    mime_type="application/json",
                )]

        else:
            return [ReadResourceContents(
                content=json.dumps({"error": f"Unknown resource: {uri_str}"}),
                mime_type="application/json",
            )]

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _handle_memory_query(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        limit = min(max(1, int(args.get("limit", 5))), 20)

        if not query:
            return _mcp_error("query is required", "validation", False)

        # Gather user facts — word-overlap scoring instead of naive substring
        from app.core.text_utils import normalize_words
        all_facts = _user_facts.get_all()
        query_words = normalize_words(query, min_length=2)
        scored_facts = []
        for f in all_facts:
            fact_words = normalize_words(f.key, min_length=2) | normalize_words(f.value, min_length=2)
            overlap = len(query_words & fact_words)
            if overlap >= 1:
                scored_facts.append((overlap, f))
        scored_facts.sort(key=lambda x: -x[0])
        matching_facts = [
            {"key": f.key, "value": f.value, "category": f.category, "confidence": f.confidence}
            for _, f in scored_facts
        ]

        # Search conversation messages
        message_results = _conversations.search_messages(query, limit=limit)

        result = {
            "matching_facts": matching_facts[:limit],
            "conversation_excerpts": message_results[:limit],
            "total_facts_checked": len(all_facts),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_knowledge_graph(args: dict) -> list[TextContent]:
        entity = args.get("entity", "")
        hops = min(max(1, int(args.get("hops", 1))), 5)

        if not entity:
            return _mcp_error("entity is required", "validation", False)

        facts = _kg.query(entity, hops=hops)
        # Never expose quarantined (uncorroborated web-derived) facts to
        # external MCP clients — same poisoning surface as the chat prompt.
        facts = [f for f in facts if not f.get('quarantined')]
        result = {
            "entity": entity,
            "hops": hops,
            "facts": [
                {
                    "subject": f["subject"],
                    "predicate": f["predicate"],
                    "object": f["object"],
                    "confidence": f["confidence"],
                    "source": f.get("source", ""),
                    "depth": f.get("depth", 0),
                }
                for f in facts
            ],
            "total": len(facts),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_lessons(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        limit = min(max(1, int(args.get("limit", 5))), 20)

        if not query:
            return _mcp_error("query is required", "validation", False)

        lessons = _learning.get_relevant_lessons(query, limit=limit)
        result = {
            "query": query,
            "lessons": [
                {
                    "id": lesson.id,
                    "topic": lesson.topic,
                    "correct_answer": lesson.correct_answer,
                    "lesson_text": lesson.lesson_text or "",
                    "confidence": lesson.confidence,
                    "times_retrieved": lesson.times_retrieved,
                    "times_helpful": lesson.times_helpful,
                }
                for lesson in lessons
            ],
            "total": len(lessons),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_document_search(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        top_k = min(max(1, int(args.get("top_k", 5))), 50)

        if not query:
            return _mcp_error("query is required", "validation", False)

        if _retriever is None:
            return _mcp_error(
                "Document search is unavailable (ChromaDB not initialized)",
                "transient",
                True,
            )

        chunks = await _retriever.search(query, top_k=top_k)
        result = {
            "query": query,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "score": round(c.score, 4),
                    "source": c.source,
                    "title": c.title,
                }
                for c in chunks
            ],
            "total": len(chunks),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_facts_about(args: dict) -> list[TextContent]:
        category = args.get("category")

        all_facts = _user_facts.get_all()

        if category:
            facts = [f for f in all_facts if f.category == category]
        else:
            facts = all_facts

        result = {
            "facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "category": f.category,
                    "confidence": f.confidence,
                }
                for f in facts
            ],
            "total": len(facts),
        }
        if category:
            result["filter"] = category

        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server
