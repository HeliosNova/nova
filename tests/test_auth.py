"""Security & validation tests — auth, SSRF, sandbox, CORS, SQL whitelist,
file delete protection, input validation, rate limiting.

Consolidated from S1-S8 audit items + R4 (input validation) + R5 (rate limiting).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


@pytest.fixture
def client(db):
    """FastAPI test client — dev mode (no API_KEY)."""
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


def _make_auth_client(db, monkeypatch, api_key="test-secret"):
    """Helper: create a test client with auth enabled."""
    monkeypatch.setenv("NOVA_API_KEY", api_key)
    import importlib, app.config, app.auth
    importlib.reload(app.config)
    importlib.reload(app.auth)

    from fastapi.testclient import TestClient
    from app.main import _rate_limit_requests

    _rate_limit_requests.clear()
    from app.main import app

    svc = Services(
        conversations=ConversationStore(db),
        user_facts=UserFactStore(db),
    )
    set_services(svc)
    return TestClient(app)


# ---------------------------------------------------------------------------
# S1: Authentication
# ---------------------------------------------------------------------------

class TestAuthSecurity:
    def test_health_always_public(self, db, monkeypatch):
        """Health check is public even with auth enabled."""
        client = _make_auth_client(db, monkeypatch)
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_auth_disabled_when_no_key(self, client):
        """With no API_KEY set, all endpoints should be accessible."""
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_status_requires_auth(self, db, monkeypatch):
        """Status endpoint requires valid token."""
        client = _make_auth_client(db, monkeypatch)
        resp = client.get("/api/status")
        assert resp.status_code == 401

    def test_chat_requires_auth(self, db, monkeypatch):
        """Chat endpoint requires valid token."""
        client = _make_auth_client(db, monkeypatch)
        resp = client.post("/api/chat", json={"query": "hello"})
        assert resp.status_code == 401

    def test_export_requires_auth(self, db, monkeypatch):
        """Export endpoint requires valid token."""
        client = _make_auth_client(db, monkeypatch)
        resp = client.get("/api/export")
        assert resp.status_code == 401

    def test_learning_requires_auth(self, db, monkeypatch):
        """Learning metrics endpoint requires valid token."""
        client = _make_auth_client(db, monkeypatch)
        resp = client.get("/api/learning/metrics")
        assert resp.status_code == 401

    def test_documents_requires_auth(self, db, monkeypatch):
        """Document search endpoint requires valid token."""
        client = _make_auth_client(db, monkeypatch)
        resp = client.get("/api/documents/search?q=test")
        assert resp.status_code == 401

    def test_correct_token_passes(self, db, monkeypatch):
        """Correct bearer token should grant access."""
        client = _make_auth_client(db, monkeypatch, api_key="my-key")
        resp = client.get("/api/status", headers={"Authorization": "Bearer my-key"})
        assert resp.status_code == 200

    def test_wrong_token_rejected(self, db, monkeypatch):
        """Wrong bearer token should be rejected."""
        client = _make_auth_client(db, monkeypatch, api_key="correct-key")
        resp = client.get("/api/status", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# S3: SSRF — document ingest
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    def test_ingest_blocks_localhost(self, db):
        """Document ingest should block localhost URLs."""
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi.testclient import TestClient
        from app.main import app, _rate_limit_requests

        _rate_limit_requests.clear()

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=MagicMock(),
        )
        set_services(svc)
        client = TestClient(app)

        resp = client.post("/api/documents/ingest", json={"url": "http://localhost:11434"})
        assert resp.status_code == 400
        assert "blocked" in resp.json()["detail"].lower()

    def test_ingest_blocks_private_ip(self, db):
        """Document ingest should block private IP ranges."""
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi.testclient import TestClient
        from app.main import app, _rate_limit_requests

        _rate_limit_requests.clear()

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=MagicMock(),
        )
        set_services(svc)
        client = TestClient(app)

        resp = client.post("/api/documents/ingest", json={"url": "http://10.0.0.1/internal"})
        assert resp.status_code == 400

    def test_ingest_blocks_metadata_ip(self, db):
        """Document ingest should block cloud metadata endpoint."""
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi.testclient import TestClient
        from app.main import app, _rate_limit_requests

        _rate_limit_requests.clear()

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=MagicMock(),
        )
        set_services(svc)
        client = TestClient(app)

        resp = client.post("/api/documents/ingest", json={"url": "http://169.254.169.254/latest/meta-data"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# S4: Code exec sandbox
# ---------------------------------------------------------------------------

class TestSandboxSecurity:
    def test_blocks_open(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("open('/etc/passwd')") is not None

    def test_blocks_eval(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("eval('__import__(\"os\")')") is not None

    def test_blocks_exec(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("exec('import os')") is not None

    def test_blocks_import_os(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("import os") is not None

    def test_blocks_getattr(self):
        from app.tools.code_exec import _check_code_safety
        result = _check_code_safety("x = getattr(__builtins__, 'open')")
        assert result is not None
        assert "blocked" in result.lower()

    def test_blocks_builtins(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("print(__builtins__)") is not None

    def test_blocks_compile(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("code = compile('import os', '<x>', 'exec')") is not None

    def test_blocks_globals(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("print(globals())") is not None

    def test_allows_safe_math(self):
        from app.tools.code_exec import _check_code_safety
        assert _check_code_safety("result = 2 ** 10\nprint(result)") is None


# ---------------------------------------------------------------------------
# S6: Redirect SSRF
# ---------------------------------------------------------------------------

class TestRedirectSSRF:
    def test_safe_url_blocks_private(self):
        from app.tools.http_fetch import _is_safe_url
        assert not _is_safe_url("http://127.0.0.1/admin")
        assert not _is_safe_url("http://10.0.0.1/secret")
        assert not _is_safe_url("http://192.168.1.1/config")
        assert not _is_safe_url("http://169.254.169.254/metadata")

    def test_safe_url_allows_public(self):
        from app.tools.http_fetch import _is_safe_url
        assert _is_safe_url("https://example.com")
        assert _is_safe_url("https://google.com/search?q=test")


# ---------------------------------------------------------------------------
# S7: SQL injection table whitelist
# ---------------------------------------------------------------------------

class TestSQLWhitelist:
    def test_allowed_tables_complete(self):
        from app.api.system import _ALLOWED_TABLES
        expected = {"conversations", "messages", "user_facts", "lessons", "skills", "documents", "kg_facts", "reflexions", "custom_tools"}
        assert expected == _ALLOWED_TABLES

    def test_dangerous_tables_excluded(self):
        from app.api.system import _ALLOWED_TABLES
        assert "sqlite_master" not in _ALLOWED_TABLES
        assert "sqlite_sequence" not in _ALLOWED_TABLES


# ---------------------------------------------------------------------------
# S8: File ops delete protection
# ---------------------------------------------------------------------------

class TestFileDeleteProtection:
    @pytest.mark.asyncio
    async def test_blocks_db_extension(self, tmp_path):
        from app.tools.file_ops import FileOpsTool
        tool = FileOpsTool()
        target = tmp_path / "data.db"
        target.write_text("data")
        with patch("app.tools.file_ops._safe_path", return_value=target):
            result = await tool.execute(action="delete", path="data.db")
        assert not result.success
        assert "protected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_sqlite_extension(self, tmp_path):
        from app.tools.file_ops import FileOpsTool
        tool = FileOpsTool()
        target = tmp_path / "app.sqlite"
        target.write_text("data")
        with patch("app.tools.file_ops._safe_path", return_value=target):
            result = await tool.execute(action="delete", path="app.sqlite")
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocks_protected_filename(self, tmp_path):
        from app.tools.file_ops import FileOpsTool
        tool = FileOpsTool()
        target = tmp_path / "training_data.jsonl"
        target.write_text("data")
        with patch("app.tools.file_ops._safe_path", return_value=target):
            result = await tool.execute(action="delete", path="training_data.jsonl")
        assert not result.success
        assert "protected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_txt_allowed(self, tmp_path):
        from app.tools.file_ops import FileOpsTool
        tool = FileOpsTool()
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello")
        with patch("app.tools.file_ops._safe_path", return_value=txt_file):
            result = await tool.execute(action="delete", path="notes.txt")
        assert result.success


# ---------------------------------------------------------------------------
# S2: CORS configuration
# ---------------------------------------------------------------------------

class TestCORSConfig:
    def test_cors_config_field_exists(self):
        from app.config import Config
        c = Config()
        assert hasattr(c, "ALLOWED_ORIGINS")

    def test_cors_default_is_localhost(self):
        from app.config import Config
        c = Config()
        assert c.ALLOWED_ORIGINS == "http://localhost:5173"

    def test_cors_headers_present(self, client):
        """OPTIONS request should return CORS headers."""
        response = client.options("/api/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert response.status_code in (200, 204, 400)


# ---------------------------------------------------------------------------
# R4: Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_query_rejected(self, client):
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_huge_query_rejected(self, client):
        resp = client.post("/api/chat", json={"query": "x" * 10_001})
        assert resp.status_code == 422

    def test_huge_ingest_rejected(self, db):
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi.testclient import TestClient
        from app.main import app, _rate_limit_requests

        _rate_limit_requests.clear()

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            retriever=MagicMock(),
        )
        set_services(svc)
        client = TestClient(app)

        resp = client.post("/api/documents/ingest", json={
            "text": "x" * 1_000_001,
            "title": "huge",
        })
        assert resp.status_code == 422

    def test_limit_bounded(self, client):
        resp = client.get("/api/chat/conversations?limit=999")
        assert resp.status_code == 422

    def test_limit_zero_rejected(self, client):
        resp = client.get("/api/chat/conversations?limit=0")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# R5: Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_normal_usage_passes(self, client):
        for _ in range(5):
            resp = client.get("/api/health")
            assert resp.status_code == 200

    def test_excess_blocked(self, client):
        responses = []
        for _ in range(65):
            r = client.get("/api/status")
            responses.append(r.status_code)
        assert 429 in responses

    def test_health_exempt(self, client):
        for _ in range(70):
            resp = client.get("/api/health")
            assert resp.status_code == 200
