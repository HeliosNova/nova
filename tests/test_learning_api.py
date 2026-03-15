"""Tests for learning API endpoints: metrics, lessons, skills, toggle, delete.

R7: Test coverage for previously untested API routes.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.learning import LearningEngine
from app.core.memory import ConversationStore, UserFactStore
from app.core.skills import SkillStore


@pytest.fixture
def client(db):
    """FastAPI test client with full services — no API_KEY set (dev mode)."""
    import importlib, app.config, app.auth
    importlib.reload(app.config)
    importlib.reload(app.auth)

    from fastapi.testclient import TestClient
    from app.main import app, _rate_limit_requests

    _rate_limit_requests.clear()

    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
        learning=LearningEngine(db),
        skills=SkillStore(db),
    )
    set_services(svc)
    return TestClient(app)


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        """GET /api/learning/metrics should return 200."""
        resp = client.get("/api/learning/metrics")
        assert resp.status_code == 200

    def test_metrics_response_shape(self, client):
        """Metrics should include expected fields."""
        resp = client.get("/api/learning/metrics")
        data = resp.json()
        assert "total_lessons" in data
        assert "total_skills" in data
        assert "total_corrections" in data
        assert "training_examples" in data

    def test_metrics_reflects_data(self, client, db):
        """Metrics should reflect actual lesson count."""
        db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, context, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            ("test topic", "wrong", "right", "context", 0.8),
        )
        resp = client.get("/api/learning/metrics")
        data = resp.json()
        assert data["total_lessons"] >= 1


class TestListLessonsEndpoint:
    def test_list_lessons_empty(self, client):
        """GET /api/learning/lessons with no data should return empty list."""
        resp = client.get("/api/learning/lessons")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_lessons_returns_data(self, client, db):
        """GET /api/learning/lessons should return inserted lessons."""
        db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, context, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("Python", "Java", "Guido", "Lesson about Python", "test", 0.9),
        )
        resp = client.get("/api/learning/lessons")
        assert resp.status_code == 200
        lessons = resp.json()
        assert len(lessons) >= 1
        assert lessons[0]["topic"] == "Python"
        assert lessons[0]["confidence"] == 0.9

    def test_list_lessons_respects_limit(self, client, db):
        """Limit parameter should cap the number of returned lessons."""
        for i in range(5):
            db.execute(
                """INSERT INTO lessons (topic, wrong_answer, correct_answer, context, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (f"topic_{i}", "wrong", "right", "ctx", 0.8),
            )
        resp = client.get("/api/learning/lessons?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestDeleteLessonEndpoint:
    def test_delete_existing_lesson(self, client, db):
        """DELETE /api/learning/lessons/{id} should remove the lesson."""
        cursor = db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, context, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            ("to_delete", "wrong", "right", "ctx", 0.8),
        )
        lesson_id = cursor.lastrowid

        resp = client.delete(f"/api/learning/lessons/{lesson_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify gone
        row = db.fetchone("SELECT id FROM lessons WHERE id = ?", (lesson_id,))
        assert row is None

    def test_delete_nonexistent_lesson(self, client):
        """DELETE /api/learning/lessons/{id} for missing ID should return 404."""
        resp = client.delete("/api/learning/lessons/99999")
        assert resp.status_code == 404


class TestListSkillsEndpoint:
    def test_list_skills_empty(self, client):
        """GET /api/learning/skills with no data should return empty list."""
        resp = client.get("/api/learning/skills")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_skills_returns_data(self, client, db):
        """GET /api/learning/skills should return inserted skills."""
        db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, enabled) VALUES (?, ?, ?, ?)""",
            ("greet", r"\bhello\b", "[]", 1),
        )
        resp = client.get("/api/learning/skills")
        assert resp.status_code == 200
        skills = resp.json()
        assert len(skills) >= 1
        assert skills[0]["name"] == "greet"
        assert skills[0]["enabled"] is True

    def test_list_skills_respects_limit(self, client, db):
        """Limit parameter should cap the number of returned skills."""
        for i in range(5):
            db.execute(
                """INSERT INTO skills (name, trigger_pattern, steps, enabled)
                   VALUES (?, ?, ?, ?)""",
                (f"skill_{i}", rf"\bskill{i}\b", "[]", 1),
            )
        resp = client.get("/api/learning/skills?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3


class TestToggleSkillEndpoint:
    def test_toggle_disable(self, client, db):
        """POST /api/learning/skills/{id}/toggle?enabled=false should disable."""
        cursor = db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, enabled, times_used, success_rate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("toggle_test", r"\btest\b", "[]", 1, 5, 0.8),
        )
        skill_id = cursor.lastrowid

        resp = client.post(f"/api/learning/skills/{skill_id}/toggle?enabled=false")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_toggle_enable_resets_stats(self, client, db):
        """POST /api/learning/skills/{id}/toggle?enabled=true should reset stats."""
        cursor = db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, enabled, times_used, success_rate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("reset_test", r"\breset\b", "[]", 0, 10, 0.3),
        )
        skill_id = cursor.lastrowid

        resp = client.post(f"/api/learning/skills/{skill_id}/toggle?enabled=true")
        assert resp.status_code == 200

        # Verify stats were reset
        row = db.fetchone("SELECT times_used, success_rate FROM skills WHERE id = ?", (skill_id,))
        assert row["times_used"] == 0
        assert row["success_rate"] == 0.7

    def test_toggle_nonexistent(self, client):
        """Toggle a missing skill should return 404."""
        resp = client.post("/api/learning/skills/99999/toggle?enabled=true")
        assert resp.status_code == 404


class TestDeleteSkillEndpoint:
    def test_delete_existing_skill(self, client, db):
        """DELETE /api/learning/skills/{id} should remove the skill."""
        cursor = db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, enabled)
               VALUES (?, ?, ?, ?)""",
            ("del_skill", r"\bdel\b", "[]", 1),
        )
        skill_id = cursor.lastrowid

        resp = client.delete(f"/api/learning/skills/{skill_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        row = db.fetchone("SELECT id FROM skills WHERE id = ?", (skill_id,))
        assert row is None

    def test_delete_nonexistent_skill(self, client):
        """DELETE /api/learning/skills/{id} for missing ID should return 404."""
        resp = client.delete("/api/learning/skills/99999")
        assert resp.status_code == 404
