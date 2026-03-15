"""Chat API — SSE streaming and synchronous endpoints.

POST /api/chat/stream — Server-Sent Events streaming response
POST /api/chat        — Synchronous (full response at once)
GET  /api/chat/conversations — List recent conversations
GET  /api/chat/conversations/search — Search conversations by content
GET  /api/chat/conversations/{id} — Get conversation with messages
DELETE /api/chat/conversations/{id} — Delete a conversation
POST /api/chat/facts   — Create/update a user fact
GET  /api/chat/facts    — List all user facts
DELETE /api/chat/facts/{key} — Delete a user fact
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.auth import require_auth
from app.core.brain import get_services, think
from app.schema import (
    ChatRequest,
    ChatResponse,
    EventType,
    StreamEvent,
    UserFactCreate,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"], dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------------------
# SSE Streaming — POST /chat/stream
# ---------------------------------------------------------------------------

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream a chat response via Server-Sent Events."""

    async def event_generator():
        try:
            async for event in think(
                query=request.query,
                conversation_id=request.conversation_id,
                image=request.image_base64,
            ):
                yield event.to_sse()
        except Exception:
            logger.exception("Error in chat stream")
            error_event = StreamEvent(
                type=EventType.ERROR,
                data={"message": "An internal error occurred while processing your request"},
            )
            yield error_event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Synchronous — POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Synchronous chat — collects the full response and returns it."""
    tokens: list[str] = []
    conversation_id = request.conversation_id
    tool_results: list[dict] = []
    sources: list[dict] = []
    lessons_used = 0
    skill_used = None

    try:
        async for event in think(
            query=request.query,
            conversation_id=request.conversation_id,
            image=request.image_base64,
        ):
            if event.type == EventType.TOKEN:
                tokens.append(event.data.get("text", ""))
            elif event.type == EventType.DONE:
                conversation_id = event.data.get("conversation_id", conversation_id)
                lessons_used = event.data.get("lessons_used", 0)
                skill_used = event.data.get("skill_used")
            elif event.type == EventType.TOOL_USE and event.data.get("status") == "complete":
                tool_results.append({
                    "tool": event.data.get("tool"),
                    "result": event.data.get("result", ""),
                })
            elif event.type == EventType.SOURCES:
                sources = event.data.get("sources", [])
            elif event.type == EventType.ERROR:
                raise HTTPException(status_code=500, detail=event.data.get("message", "Unknown error"))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in sync chat")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request")

    return ChatResponse(
        answer="".join(tokens),
        conversation_id=conversation_id or "",
        sources=sources,
        tool_results=tool_results,
        lessons_used=lessons_used,
        skill_used=skill_used,
    )


# ---------------------------------------------------------------------------
# Conversations — CRUD
# ---------------------------------------------------------------------------

@router.get("/chat/conversations")
async def list_conversations(limit: int = Query(default=50, ge=1, le=500)):
    """List recent conversations."""
    svc = get_services()
    return svc.conversations.list_conversations(limit=limit)


@router.get("/chat/conversations/search")
async def search_conversations(q: str = Query(max_length=1_000), limit: int = Query(default=20, ge=1, le=500)):
    """Search across all conversations by message content."""
    svc = get_services()
    if not q.strip():
        return []
    return svc.conversations.search_conversations(q, limit=limit)


@router.get("/chat/messages/search")
async def search_messages(q: str = Query(max_length=1_000), limit: int = Query(default=20, ge=1, le=500)):
    """Search all messages across conversations. Returns individual messages with context."""
    svc = get_services()
    if not q.strip():
        return []
    return svc.conversations.search_messages(q, limit=limit)


@router.get("/chat/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a conversation with its messages."""
    svc = get_services()
    conv = svc.conversations.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = svc.conversations.get_history(conv_id, limit=100)
    return {
        **conv,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_name": m.tool_name,
                "sources": m.sources,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    }


@router.patch("/chat/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: dict):
    """Rename a conversation."""
    svc = get_services()
    conv = svc.conversations.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    if len(title) > 500:
        raise HTTPException(status_code=400, detail="Title too long (max 500 chars)")
    svc.conversations.update_title(conv_id, title)
    return {"status": "ok", "conversation_id": conv_id, "title": title}


@router.delete("/chat/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    svc = get_services()
    conv = svc.conversations.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    svc.conversations.delete_conversation(conv_id)
    return {"status": "deleted", "conversation_id": conv_id}


# ---------------------------------------------------------------------------
# User Facts — CRUD
# ---------------------------------------------------------------------------

@router.get("/chat/facts")
async def list_facts():
    """List all user facts."""
    svc = get_services()
    facts = svc.user_facts.get_all()
    return [
        {"id": f.id, "key": f.key, "value": f.value, "source": f.source, "confidence": f.confidence, "category": f.category}
        for f in facts
    ]


@router.post("/chat/facts")
async def create_fact(fact: UserFactCreate):
    """Create or update a user fact."""
    svc = get_services()
    svc.user_facts.set(fact.key, fact.value, source=fact.source, category=fact.category)
    return {"status": "ok", "key": fact.key}


@router.delete("/chat/facts/{key}")
async def delete_fact(key: str):
    """Delete a user fact."""
    svc = get_services()
    deleted = svc.user_facts.delete(key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Fact '{key}' not found")
    return {"status": "deleted", "key": key}
