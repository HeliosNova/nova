"""Tests for WhatsApp webhook HMAC-SHA256 signature verification.

Security-critical path: verifies that only messages signed with
the WhatsApp app secret are processed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


APP_SECRET = "test_whatsapp_app_secret_key"
VERIFY_TOKEN = "test_verify_token"


def _sign_body(body: bytes, secret: str = APP_SECRET) -> str:
    """Compute the WhatsApp-style HMAC-SHA256 signature."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _make_webhook_body(text: str = "hello", sender: str = "15551234567") -> dict:
    """Build a minimal WhatsApp webhook message body."""
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "123456",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"phone_number_id": "test_phone_id"},
                    "messages": [{
                        "from": sender,
                        "id": f"wamid.{int(time.time())}",
                        "timestamp": str(int(time.time())),
                        "type": "text",
                        "text": {"body": text},
                    }],
                },
                "field": "messages",
            }],
        }],
    }


@pytest.fixture
def client(db, monkeypatch):
    """Create test client with WhatsApp webhook routes."""
    monkeypatch.setenv("WHATSAPP_API_URL", "https://graph.facebook.com/v18.0")
    monkeypatch.setenv("WHATSAPP_API_TOKEN", "test_token")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", VERIFY_TOKEN)
    monkeypatch.setenv("WHATSAPP_PHONE_ID", "test_phone_id")
    monkeypatch.setenv("WHATSAPP_CHAT_ID", "test_chat")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", APP_SECRET)

    import importlib
    import app.config
    importlib.reload(app.config)

    from app.channels.whatsapp import WhatsAppBot
    from app.main import app, _rate_limit_requests
    _rate_limit_requests.clear()

    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
    )
    set_services(svc)

    bot = WhatsAppBot()
    app.include_router(bot.get_router())

    yield TestClient(app)

    # Cleanup: remove the whatsapp routes to avoid conflicts in other tests
    app.routes[:] = [r for r in app.routes if not getattr(r, "path", "").startswith("/api/channels/whatsapp")]


# ===========================================================================
# GET /webhook — Hub verification
# ===========================================================================

class TestWebhookVerification:
    def test_valid_verify_token(self, client):
        """Valid verify token returns the challenge."""
        resp = client.get("/api/channels/whatsapp/webhook", params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "challenge_string_123",
        })
        assert resp.status_code == 200
        assert resp.text == "challenge_string_123"

    def test_invalid_verify_token(self, client):
        """Invalid verify token is rejected."""
        resp = client.get("/api/channels/whatsapp/webhook", params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "challenge_string_123",
        })
        assert resp.status_code == 403

    def test_missing_verify_token(self, client):
        """Missing verify token is rejected."""
        resp = client.get("/api/channels/whatsapp/webhook", params={
            "hub.mode": "subscribe",
            "hub.challenge": "challenge_string_123",
        })
        assert resp.status_code == 403


# ===========================================================================
# POST /webhook — HMAC signature verification
# ===========================================================================

class TestHMACSignatureVerification:
    def test_valid_signature_accepted(self, client):
        """Request with valid HMAC-SHA256 signature is accepted."""
        body = _make_webhook_body("test message")
        raw = json.dumps(body).encode()
        sig = _sign_body(raw)

        with patch("app.core.brain.think") as mock_think:
            mock_think.return_value = _empty_gen()
            resp = client.post(
                "/api/channels/whatsapp/webhook",
                content=raw,
                headers={
                    "content-type": "application/json",
                    "x-hub-signature-256": sig,
                },
            )
        assert resp.status_code == 200

    def test_invalid_signature_rejected(self, client):
        """Request with wrong signature is rejected with 403."""
        body = _make_webhook_body("test message")
        raw = json.dumps(body).encode()
        wrong_sig = _sign_body(raw, secret="wrong_secret")

        resp = client.post(
            "/api/channels/whatsapp/webhook",
            content=raw,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": wrong_sig,
            },
        )
        assert resp.status_code == 403
        assert "invalid signature" in resp.json().get("status", "").lower()

    def test_missing_signature_rejected(self, client):
        """Request with no signature header is rejected with 403."""
        body = _make_webhook_body("test message")
        raw = json.dumps(body).encode()

        resp = client.post(
            "/api/channels/whatsapp/webhook",
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 403
        assert "missing signature" in resp.json().get("status", "").lower()

    def test_tampered_body_rejected(self, client):
        """Signature computed on original body rejects tampered body."""
        body = _make_webhook_body("original message")
        raw = json.dumps(body).encode()
        sig = _sign_body(raw)  # Signature for original

        # Tamper with the body
        tampered = _make_webhook_body("tampered message")
        tampered_raw = json.dumps(tampered).encode()

        resp = client.post(
            "/api/channels/whatsapp/webhook",
            content=tampered_raw,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": sig,  # Original signature
            },
        )
        assert resp.status_code == 403

    def test_empty_signature_rejected(self, client):
        """Empty string signature is rejected."""
        body = _make_webhook_body("test")
        raw = json.dumps(body).encode()

        resp = client.post(
            "/api/channels/whatsapp/webhook",
            content=raw,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": "",
            },
        )
        assert resp.status_code == 403

    def test_malformed_signature_rejected(self, client):
        """Signature without sha256= prefix is rejected."""
        body = _make_webhook_body("test")
        raw = json.dumps(body).encode()
        sig_no_prefix = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()

        resp = client.post(
            "/api/channels/whatsapp/webhook",
            content=raw,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": sig_no_prefix,  # Missing sha256= prefix
            },
        )
        assert resp.status_code == 403


# ===========================================================================
# POST /webhook — App secret not configured
# ===========================================================================

class TestAppSecretNotConfigured:
    def test_no_app_secret_rejects_all(self, db, monkeypatch):
        """Without WHATSAPP_APP_SECRET, all webhooks are rejected (fail-closed)."""
        monkeypatch.setenv("WHATSAPP_API_URL", "https://graph.facebook.com/v18.0")
        monkeypatch.setenv("WHATSAPP_API_TOKEN", "test_token")
        monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", VERIFY_TOKEN)
        monkeypatch.setenv("WHATSAPP_PHONE_ID", "test_phone_id")
        monkeypatch.setenv("WHATSAPP_CHAT_ID", "test_chat")
        monkeypatch.setenv("WHATSAPP_APP_SECRET", "")

        import importlib
        import app.config
        importlib.reload(app.config)

        from app.channels.whatsapp import WhatsAppBot
        from app.main import app as _app, _rate_limit_requests
        _rate_limit_requests.clear()

        svc = Services(conversations=ConversationStore(db), user_facts=UserFactStore(db))
        set_services(svc)

        bot = WhatsAppBot()
        _app.include_router(bot.get_router())

        test_client = TestClient(_app)
        body = _make_webhook_body("hello")
        raw = json.dumps(body).encode()

        resp = test_client.post(
            "/api/channels/whatsapp/webhook",
            content=raw,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": _sign_body(raw),
            },
        )
        assert resp.status_code == 503

        # Cleanup routes
        _app.routes[:] = [r for r in _app.routes if not getattr(r, "path", "").startswith("/api/channels/whatsapp")]


# ===========================================================================
# Helpers
# ===========================================================================

async def _empty_gen():
    """Empty async generator for mocking think()."""
    from app.schema import EventType, StreamEvent
    yield StreamEvent(type=EventType.DONE, data={"conversation_id": "test"})
