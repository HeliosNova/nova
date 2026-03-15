"""Security hardening tests — Sprint 3.

Covers:
- Query length validation (too long -> 400/422)
- conversation_id format validation
- Rate limit headers present in responses
- Security headers present in responses
- API key timing-safe comparison (hmac.compare_digest)
- Monitor input validation
- Action endpoint validation
- Document ingest validation
"""

from __future__ import annotations

import hmac
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# Query length validation
# ---------------------------------------------------------------------------

class TestQueryLengthValidation:
    def test_query_too_long_rejected(self, client):
        """Query exceeding 10,000 chars should be rejected with 422."""
        resp = client.post("/api/chat", json={"query": "x" * 10_001})
        assert resp.status_code == 422

    def test_query_at_max_length_accepted(self, client):
        """Query at exactly 10,000 chars should be accepted (not a validation error)."""
        # This will fail at the brain level (no LLM), but should pass validation
        resp = client.post("/api/chat", json={"query": "x" * 10_000})
        # 422 would mean validation failed; anything else means it passed validation
        assert resp.status_code != 422

    def test_empty_query_rejected(self, client):
        """Empty query should be rejected with 422."""
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# conversation_id format validation
# ---------------------------------------------------------------------------

class TestConversationIdValidation:
    def test_valid_uuid_accepted(self, client):
        """Valid UUID-style conversation_id should pass validation."""
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": "abc123-def-456",
        })
        # Should not be a 422 validation error
        assert resp.status_code != 422

    def test_valid_alphanumeric_accepted(self, client):
        """Simple alphanumeric conversation_id should pass validation."""
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": "conv_test_123",
        })
        assert resp.status_code != 422

    def test_null_conversation_id_accepted(self, client):
        """Null conversation_id should be accepted."""
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": None,
        })
        assert resp.status_code != 422

    def test_special_chars_rejected(self, client):
        """conversation_id with path traversal chars should be rejected."""
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": "../../../etc/passwd",
        })
        assert resp.status_code == 422

    def test_too_long_rejected(self, client):
        """conversation_id exceeding 100 chars should be rejected."""
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": "a" * 101,
        })
        assert resp.status_code == 422

    def test_sql_injection_rejected(self, client):
        """conversation_id with SQL injection should be rejected."""
        resp = client.post("/api/chat", json={
            "query": "hello",
            "conversation_id": "'; DROP TABLE conversations; --",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Rate limit headers
# ---------------------------------------------------------------------------

class TestRateLimitHeaders:
    def test_rate_limit_headers_present(self, client):
        """Responses should include X-RateLimit-* headers."""
        resp = client.get("/api/status")
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers
        assert "x-ratelimit-reset" in resp.headers

    def test_rate_limit_limit_value(self, client):
        """X-RateLimit-Limit should be the configured max (60)."""
        resp = client.get("/api/status")
        assert resp.headers["x-ratelimit-limit"] == "60"

    def test_rate_limit_remaining_decreases(self, client):
        """X-RateLimit-Remaining should decrease with each request."""
        resp1 = client.get("/api/status")
        remaining1 = int(resp1.headers["x-ratelimit-remaining"])

        resp2 = client.get("/api/status")
        remaining2 = int(resp2.headers["x-ratelimit-remaining"])

        assert remaining2 < remaining1

    def test_rate_limit_reset_is_integer(self, client):
        """X-RateLimit-Reset should be a valid integer timestamp."""
        resp = client.get("/api/status")
        reset = resp.headers["x-ratelimit-reset"]
        assert reset.isdigit()

    def test_rate_limit_429_has_headers(self, client):
        """429 responses should also include rate limit headers."""
        # Exhaust the rate limit
        for _ in range(60):
            client.get("/api/status")

        resp = client.get("/api/status")
        assert resp.status_code == 429
        assert "x-ratelimit-limit" in resp.headers
        assert resp.headers["x-ratelimit-remaining"] == "0"

    def test_health_no_rate_limit_headers(self, client):
        """Health endpoint is exempt from rate limiting, so no rate limit headers."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        # Health is exempt from rate limit middleware, so no rate limit headers
        assert "x-ratelimit-limit" not in resp.headers


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    def test_x_content_type_options(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("x-frame-options") == "DENY"

    def test_x_xss_protection(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("x-xss-protection") == "1; mode=block"

    def test_content_security_policy(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("content-security-policy") == "default-src 'self'; connect-src 'self'"

    def test_referrer_policy(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_security_headers_on_all_endpoints(self, client):
        """Security headers should be present on non-health endpoints too."""
        resp = client.get("/api/status")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_security_headers_on_error_responses(self, client):
        """Security headers should be present even on error responses."""
        resp = client.get("/api/chat/conversations/nonexistent-id")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"


# ---------------------------------------------------------------------------
# API key timing-safe comparison
# ---------------------------------------------------------------------------

class TestApiKeyTimingSafe:
    def test_auth_uses_hmac_compare_digest(self):
        """Verify that auth.py uses hmac.compare_digest, not == for API key comparison."""
        import inspect
        from app import auth
        source = inspect.getsource(auth.require_auth)
        assert "hmac.compare_digest" in source
        # Ensure we don't use simple equality
        assert "credentials ==" not in source
        assert "== config.API_KEY" not in source

    def test_hmac_imported_in_auth(self):
        """Verify hmac module is imported in auth module."""
        import app.auth
        assert hasattr(app.auth, "hmac")


# ---------------------------------------------------------------------------
# Monitor input validation
# ---------------------------------------------------------------------------

class TestMonitorValidation:
    def test_invalid_check_type_rejected(self, client, db):
        """Invalid check_type should be rejected."""
        from app.monitors.heartbeat import MonitorStore
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        from app.monitors.heartbeat import MonitorStore
        svc.monitor_store = MonitorStore(db)
        set_services(svc)

        resp = client.post("/api/monitors", json={
            "name": "Test Monitor",
            "check_type": "dangerous_type",
        })
        assert resp.status_code == 422

    def test_invalid_monitor_name_rejected(self, client, db):
        """Monitor name with special chars should be rejected."""
        from app.monitors.heartbeat import MonitorStore
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        svc.monitor_store = MonitorStore(db)
        set_services(svc)

        resp = client.post("/api/monitors", json={
            "name": "<script>alert(1)</script>",
            "check_type": "search",
        })
        assert resp.status_code == 422

    def test_schedule_too_low_rejected(self, client, db):
        """Schedule seconds below minimum should be rejected."""
        from app.monitors.heartbeat import MonitorStore
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        svc.monitor_store = MonitorStore(db)
        set_services(svc)

        resp = client.post("/api/monitors", json={
            "name": "Fast Monitor",
            "check_type": "search",
            "schedule_seconds": 1,  # below minimum of 10
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Action endpoint validation
# ---------------------------------------------------------------------------

class TestActionValidation:
    def test_invalid_action_type_rejected(self, client, db):
        """action_type with special chars should be rejected."""
        resp = client.get("/api/actions?action_type='; DROP TABLE--")
        assert resp.status_code == 400

    def test_hours_too_high_rejected(self, client, db):
        """hours parameter above max should be rejected."""
        resp = client.get("/api/actions?hours=9999")
        assert resp.status_code == 422

    def test_limit_too_high_rejected(self, client, db):
        """limit parameter above max should be rejected."""
        resp = client.get("/api/actions?limit=9999")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Document ingest validation
# ---------------------------------------------------------------------------

class TestDocumentValidation:
    def test_invalid_url_scheme_rejected(self, client, db):
        """URL with non-http scheme should be rejected."""
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=MagicMock(),
        )
        set_services(svc)

        resp = client.post("/api/documents/ingest", json={
            "url": "ftp://evil.com/file",
        })
        assert resp.status_code == 422

    def test_title_too_long_rejected(self, client, db):
        """Title exceeding max length should be rejected."""
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=MagicMock(),
        )
        set_services(svc)

        resp = client.post("/api/documents/ingest", json={
            "text": "Some content",
            "title": "x" * 501,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# User fact validation
# ---------------------------------------------------------------------------

class TestUserFactValidation:
    def test_empty_key_rejected(self, client):
        """Empty fact key should be rejected."""
        resp = client.post("/api/chat/facts", json={
            "key": "",
            "value": "some value",
        })
        assert resp.status_code == 422

    def test_value_too_long_rejected(self, client):
        """Fact value exceeding max should be rejected."""
        resp = client.post("/api/chat/facts", json={
            "key": "test_key",
            "value": "x" * 5_001,
        })
        assert resp.status_code == 422
