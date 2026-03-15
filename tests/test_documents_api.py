"""Tests for documents API — list, ingest, get, delete.

Item 10 coverage: documents API endpoints had zero tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


@pytest.fixture
def mock_retriever():
    """A mock retriever that behaves like the real one."""
    retriever = MagicMock()
    retriever.list_documents = MagicMock(return_value=[])
    retriever.get_document = MagicMock(return_value=None)
    retriever.delete_document = MagicMock(return_value=False)
    retriever.ingest = AsyncMock(return_value=("doc-123", 5))
    retriever.search = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def client(db, mock_retriever):
    """FastAPI test client with services + mock retriever."""
    from fastapi.testclient import TestClient
    from app.main import app

    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
        retriever=mock_retriever,
    )
    set_services(svc)
    return TestClient(app)


class TestDocumentsAPI:
    def test_list_empty(self, client):
        """GET /api/documents should return empty list when no docs."""
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_ingest_text(self, client, mock_retriever):
        """POST /api/documents/ingest should accept text and return doc info."""
        response = client.post("/api/documents/ingest", json={
            "text": "This is a test document with enough content to be valid.",
            "title": "Test Doc",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["chunk_count"] == 5
        mock_retriever.ingest.assert_called_once()

    def test_ingest_empty_rejected(self, client):
        """POST /api/documents/ingest with empty text should return 400."""
        response = client.post("/api/documents/ingest", json={
            "text": "",
            "title": "Empty",
        })
        assert response.status_code == 400

    def test_get_document(self, client, mock_retriever):
        """GET /api/documents/{id} should return doc when found."""
        mock_retriever.get_document.return_value = {
            "id": "doc-123",
            "title": "Test Doc",
            "source": "direct_text",
            "chunk_count": 5,
            "created_at": "2026-01-01T00:00:00",
        }
        response = client.get("/api/documents/doc-123")
        assert response.status_code == 200
        assert response.json()["id"] == "doc-123"

    def test_get_document_not_found(self, client):
        """GET /api/documents/{id} should return 404 when not found."""
        response = client.get("/api/documents/nonexistent")
        assert response.status_code == 404

    def test_delete_document(self, client, mock_retriever):
        """DELETE /api/documents/{id} should return success when found."""
        mock_retriever.delete_document.return_value = True
        response = client.delete("/api/documents/doc-123")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_document_not_found(self, client):
        """DELETE /api/documents/{id} should return 404 when not found."""
        response = client.delete("/api/documents/nonexistent")
        assert response.status_code == 404


class TestDocumentsNoRetriever:
    def test_list_without_retriever_returns_empty(self, db):
        """When retriever is None, list should return empty list."""
        from fastapi.testclient import TestClient
        from app.main import app

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=None,
        )
        set_services(svc)
        client = TestClient(app)

        response = client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_ingest_without_retriever_returns_503(self, db):
        """When retriever is None, ingest should return 503."""
        from fastapi.testclient import TestClient
        from app.main import app

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=None,
        )
        set_services(svc)
        client = TestClient(app)

        response = client.post("/api/documents/ingest", json={
            "text": "some text content here for testing",
            "title": "Test",
        })
        assert response.status_code == 503


class TestDocumentsEdgeCases:
    """Edge cases for document ingestion."""

    @pytest.fixture
    def client(self, db):
        from fastapi.testclient import TestClient
        from app.main import app
        mock_retriever = MagicMock()
        mock_retriever.list_documents = MagicMock(return_value=[])
        mock_retriever.get_document = MagicMock(return_value=None)
        mock_retriever.delete_document = MagicMock(return_value=False)
        mock_retriever.ingest = AsyncMock(return_value=("doc-edge", 3))
        mock_retriever.search = AsyncMock(return_value=[])
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=mock_retriever,
        )
        set_services(svc)
        return TestClient(app)

    def test_ingest_whitespace_only(self, client):
        """Whitespace-only text should be rejected."""
        response = client.post("/api/documents/ingest", json={
            "text": "   \n\t   ",
        })
        assert response.status_code == 400

    def test_ingest_no_text_no_url(self, client):
        """Missing both text and url should be rejected."""
        response = client.post("/api/documents/ingest", json={
            "title": "Empty Doc",
        })
        assert response.status_code == 400

    def test_ingest_with_title(self, client):
        """Title should be accepted and passed through."""
        response = client.post("/api/documents/ingest", json={
            "text": "Valid text content for ingestion testing purposes.",
            "title": "My Custom Title",
        })
        assert response.status_code == 200

    def test_search_with_empty_query(self, client):
        """Search with empty query should return empty results."""
        response = client.post("/api/documents/search?query=")
        assert response.status_code == 200
        assert response.json() == []

    def test_delete_nonexistent_document(self, client):
        """Deleting nonexistent doc should return 404."""
        response = client.delete("/api/documents/99999")
        assert response.status_code == 404

    def test_list_returns_expected_fields(self, db):
        """Listed documents should have required fields."""
        from fastapi.testclient import TestClient
        from app.main import app
        mock_retriever = MagicMock()
        mock_retriever.list_documents = MagicMock(return_value=[
            {
                "id": "doc-1",
                "title": "Test",
                "source": "direct_text",
                "chunk_count": 5,
                "created_at": "2026-01-01T00:00:00",
            }
        ])
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=mock_retriever,
        )
        set_services(svc)
        client = TestClient(app)

        response = client.get("/api/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert "title" in docs[0]
        assert "chunk_count" in docs[0]
