"""Documents API — ingest, list, delete documents.

POST /api/documents/ingest — Ingest text or URL
GET  /api/documents — List all documents
GET  /api/documents/{id} — Get document details
DELETE /api/documents/{id} — Delete a document
POST /api/documents/search — Search documents
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import require_auth
from app.core.brain import get_services
from app.schema import IngestRequest
from app.tools.http_fetch import _is_safe_url, _safe_url_with_pinned_ip

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_auth)])


@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    """Ingest a document (text or URL)."""
    svc = get_services()

    if not svc.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    text = request.text
    source = "direct_text"

    # If URL provided, fetch it (with SSRF protection + DNS pinning)
    if request.url and not text:
        pin_result = _safe_url_with_pinned_ip(request.url)
        if pin_result is None:
            raise HTTPException(status_code=400, detail="URL blocked: internal/private addresses not allowed")
        _orig_url, pinned_url, original_host = pin_result
        try:
            import httpx
            _MAX_FETCH_BYTES = 10 * 1024 * 1024  # 10 MB
            _fetch_headers = {"Host": original_host} if original_host else {}
            async with httpx.AsyncClient(timeout=15.0) as client:
                async with client.stream("GET", pinned_url, headers=_fetch_headers) as resp:
                    resp.raise_for_status()
                    chunks = []
                    total = 0
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > _MAX_FETCH_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Response too large (>{_MAX_FETCH_BYTES // (1024*1024)}MB)",
                            )
                        chunks.append(chunk)
                    text = b"".join(chunks).decode("utf-8", errors="replace")
                source = request.url
        except HTTPException:
            raise
        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail="URL fetch timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=400, detail=f"URL returned HTTP {e.response.status_code}")
        except Exception:
            logger.exception("URL fetch failed for document ingest")
            raise HTTPException(status_code=400, detail="Failed to fetch URL")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text content to ingest")

    doc_id, chunk_count = await svc.retriever.ingest(
        text,
        source=source,
        title=request.title or source,
    )

    return {
        "status": "ok",
        "document_id": doc_id,
        "chunk_count": chunk_count,
        "title": request.title or source,
    }


@router.get("")
async def list_documents(limit: int = Query(default=50, ge=1, le=500)):
    """List all ingested documents."""
    svc = get_services()
    if not svc.retriever:
        return []
    return svc.retriever.list_documents(limit=limit)


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """Get a document's metadata."""
    svc = get_services()
    if not svc.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    doc = svc.retriever.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    svc = get_services()
    if not svc.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    deleted = svc.retriever.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": doc_id}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)


@router.post("/search")
async def search_documents(body: SearchRequest):
    """Search ingested documents."""
    query = body.query
    svc = get_services()
    if not svc.retriever:
        return []

    chunks = await svc.retriever.search(query)
    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "content": c.content[:500],
            "score": round(c.score, 4),
            "title": c.title,
        }
        for c in chunks
    ]
