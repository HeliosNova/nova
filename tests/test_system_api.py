"""Tests for system API endpoints: health, status, export, import.

R7: Test coverage for previously untested API routes.
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


@pytest.fixture
def client(db):
    """FastAPI test client with services — no API_KEY set (dev mode)."""
    import importlib, app.config, app.auth
    importlib.reload(app.config)
    importlib.reload(app.auth)

    from fastapi.testclient import TestClient
    from app.main import app, _rate_limit_requests

    _rate_limit_requests.clear()

    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
    )
    set_services(svc)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /api/health should always return 200."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        """Health response should have required fields (minimal: status + timestamp only)."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "ok"
        # Verify no sensitive info is leaked
        assert "model" not in data
        assert "version" not in data
        assert "provider" not in data
        assert "db_connected" not in data
        assert "llm_connected" not in data


class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        """GET /api/status should return 200 with counts."""
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_status_response_shape(self, client):
        """Status response should have all expected count fields."""
        resp = client.get("/api/status")
        data = resp.json()
        for field in ("conversations", "messages", "user_facts", "lessons", "skills", "documents", "training_examples"):
            assert field in data
            assert isinstance(data[field], int)

    def test_status_reflects_data(self, client, db):
        """Status counts should reflect actual data in the database."""
        # Insert a conversation
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("conv-test", "Test Conversation"),
        )
        resp = client.get("/api/status")
        data = resp.json()
        assert data["conversations"] >= 1


class TestExportEndpoint:
    def test_export_returns_json(self, client):
        """GET /api/export should return JSON with version field."""
        resp = client.get("/api/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["version"] == "1.0"

    def test_export_has_all_sections(self, client):
        """Export should include all data sections."""
        resp = client.get("/api/export")
        data = resp.json()
        for section in ("conversations", "user_facts", "lessons", "skills", "training_data"):
            assert section in data
            assert isinstance(data[section], list)

    def test_export_includes_conversations(self, client, db):
        """Exported data should include conversations with messages."""
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("conv-export", "Export Test"),
        )
        db.execute(
            """INSERT INTO messages (id, conversation_id, role, content)
               VALUES (?, ?, ?, ?)""",
            ("msg-1", "conv-export", "user", "Hello Nova"),
        )

        resp = client.get("/api/export")
        data = resp.json()
        assert len(data["conversations"]) >= 1
        conv = next(c for c in data["conversations"] if c["id"] == "conv-export")
        assert conv["title"] == "Export Test"
        assert len(conv["messages"]) == 1
        assert conv["messages"][0]["content"] == "Hello Nova"

    def test_export_content_disposition(self, client):
        """Export should set Content-Disposition header for download."""
        resp = client.get("/api/export")
        assert "content-disposition" in resp.headers
        assert "nova_export.json" in resp.headers["content-disposition"]


class TestImportEndpoint:
    def test_import_valid_json(self, client, db):
        """POST /api/import should accept valid export JSON."""
        export_data = {
            "version": "1.0",
            "conversations": [],
            "user_facts": [],
            "lessons": [],
            "skills": [],
            "training_data": [],
        }
        file_content = json.dumps(export_data).encode()
        resp = client.post(
            "/api/import",
            files={"file": ("export.json", io.BytesIO(file_content), "application/json")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "imported"

    def test_import_rejects_invalid_json(self, client):
        """POST /api/import with invalid JSON should return 400."""
        resp = client.post(
            "/api/import",
            files={"file": ("bad.json", io.BytesIO(b"not json"), "application/json")},
        )
        assert resp.status_code == 400

    def test_import_rejects_missing_version(self, client):
        """POST /api/import without version field should return 400."""
        file_content = json.dumps({"conversations": []}).encode()
        resp = client.post(
            "/api/import",
            files={"file": ("no_ver.json", io.BytesIO(file_content), "application/json")},
        )
        assert resp.status_code == 400
        assert "version" in resp.json()["detail"].lower()

    def test_import_creates_data(self, client, db):
        """Import should insert conversations and lessons into the database."""
        export_data = {
            "version": "1.0",
            "conversations": [
                {
                    "id": "imported-conv-1",
                    "title": "Imported",
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "messages": [
                        {
                            "id": "imported-msg-1",
                            "role": "user",
                            "content": "Hello from import",
                            "tool_calls": None,
                            "tool_name": None,
                            "sources": None,
                            "created_at": "2026-01-01T00:00:00",
                        }
                    ],
                }
            ],
            "user_facts": [{"key": "test_fact", "value": "imported", "source": "test", "confidence": 1.0}],
            "lessons": [
                {
                    "id": 999,
                    "topic": "imported lesson",
                    "wrong_answer": "wrong",
                    "correct_answer": "right",
                    "lesson_text": "test",
                    "context": "test",
                    "confidence": 0.9,
                }
            ],
            "skills": [],
            "training_data": [],
        }
        file_content = json.dumps(export_data).encode()
        resp = client.post(
            "/api/import",
            files={"file": ("data.json", io.BytesIO(file_content), "application/json")},
        )
        assert resp.status_code == 200
        stats = resp.json()["stats"]
        assert stats["conversations"] == 1
        assert stats["messages"] == 1
        assert stats["user_facts"] == 1
        assert stats["lessons"] == 1

    def test_import_skips_duplicates(self, client, db):
        """Importing the same conversation twice should skip duplicates."""
        export_data = {
            "version": "1.0",
            "conversations": [
                {"id": "dup-conv", "title": "Dup", "created_at": None, "updated_at": None, "messages": []}
            ],
            "user_facts": [],
            "lessons": [],
            "skills": [],
            "training_data": [],
        }
        file_content = json.dumps(export_data).encode()

        # First import
        client.post("/api/import", files={"file": ("a.json", io.BytesIO(file_content), "application/json")})

        # Second import — should skip
        resp = client.post(
            "/api/import",
            files={"file": ("b.json", io.BytesIO(file_content), "application/json")},
        )
        assert resp.status_code == 200
        assert resp.json()["stats"]["conversations"] == 0
