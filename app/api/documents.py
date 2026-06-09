"""Documents API — ingest, list, delete documents.

POST /api/documents/ingest — Ingest text or URL
POST /api/documents/upload — Upload a file (txt/md/pdf/docx) for indexing
GET  /api/documents — List all documents
GET  /api/documents/{id} — Get document details
DELETE /api/documents/{id} — Delete a document
POST /api/documents/search — Search documents
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.auth import require_auth
from app.core.brain import get_services
from app.schema import IngestRequest
from app.tools.http_fetch import _is_safe_url, _safe_url_with_pinned_ip

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_auth)])

# Upload limits — keep small enough not to block the worker, big enough for real PDFs
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
SUPPORTED_UPLOAD_EXTS = {".txt", ".md", ".markdown", ".rst", ".log",
                          ".pdf", ".docx", ".html", ".htm", ".json", ".csv"}


def _extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded file. Returns "" on unsupported.

    PDFs use pypdf, DOCX uses python-docx, HTML strips tags via stdlib.
    Plain text formats are decoded directly.
    """
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md", ".markdown", ".rst", ".log", ".json", ".csv"}:
        return raw.decode("utf-8", errors="replace")
    if ext in {".html", ".htm"}:
        try:
            from html.parser import HTMLParser

            class _Stripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts: list[str] = []

                def handle_data(self, data):
                    self.parts.append(data)

            stripper = _Stripper()
            stripper.feed(raw.decode("utf-8", errors="replace"))
            return " ".join(s.strip() for s in stripper.parts if s.strip())
        except Exception as e:
            logger.warning("HTML extract failed: %s", e)
            return raw.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="PDF support requires pypdf. Install with: pip install pypdf",
            )
        except Exception as e:
            logger.warning("PDF extract failed: %s", e)
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")
    if ext == ".docx":
        try:
            import io
            from docx import Document as _DocxDocument
            doc = _DocxDocument(io.BytesIO(raw))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="DOCX support requires python-docx. Install with: pip install python-docx",
            )
        except Exception as e:
            logger.warning("DOCX extract failed: %s", e)
            raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {e}")
    return ""


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


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Query(default="", max_length=300),
):
    """Upload a file (txt/md/pdf/docx/html) and index it into ChromaDB + FTS5.

    Personal RAG ingestion path. The retriever already supports search; this
    endpoint adds the missing piece: getting your own files into the index.
    """
    svc = get_services()
    if not svc.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext}. Supported: {sorted(SUPPORTED_UPLOAD_EXTS)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes, max {MAX_UPLOAD_BYTES})",
        )

    try:
        text = _extract_text(file.filename, raw)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Text extraction failed for %s", file.filename)
        raise HTTPException(status_code=400, detail=f"Text extraction failed: {e}")

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from this file")

    doc_title = (title or file.filename).strip()[:300]
    doc_id, chunk_count = await svc.retriever.ingest(
        text,
        source=f"upload:{file.filename}",
        title=doc_title,
    )

    return {
        "status": "ok",
        "document_id": doc_id,
        "chunk_count": chunk_count,
        "title": doc_title,
        "filename": file.filename,
        "bytes": len(raw),
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
