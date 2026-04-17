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
        resp = client.post("/api/chat", json={"query": "x" * 50_001})
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

    def test_ratelimit_headers_present_on_normal_request(self, client):
        """Normal requests must include X-RateLimit-Limit/Remaining/Reset headers."""
        resp = client.get("/api/status")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers
        assert int(resp.headers["X-RateLimit-Limit"]) >= 1

    def test_ratelimit_headers_present_on_429(self, client):
        """429 response must also carry rate-limit headers with Remaining=0."""
        resp = None
        for _ in range(65):
            resp = client.get("/api/status")
            if resp.status_code == 429:
                break
        assert resp is not None and resp.status_code == 429
        assert resp.headers.get("X-RateLimit-Remaining") == "0"
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_remaining_decrements(self, client):
        """X-RateLimit-Remaining should decrease with each request."""
        remaining_values = []
        for _ in range(3):
            r = client.get("/api/status")
            if r.status_code == 200:
                remaining_values.append(int(r.headers.get("X-RateLimit-Remaining", -1)))
        assert len(remaining_values) >= 2
        assert remaining_values[0] > remaining_values[-1]

    def test_custom_limit_respected(self, db, monkeypatch):
        """RATE_LIMIT_RPM config drives the effective per-IP limit."""
        monkeypatch.setenv("RATE_LIMIT_RPM", "3")
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi.testclient import TestClient
        from app.main import app, _rate_limit_requests
        _rate_limit_requests.clear()

        from app.core.brain import Services, set_services
        from app.core.memory import ConversationStore, UserFactStore
        svc = Services(conversations=ConversationStore(db), user_facts=UserFactStore(db))
        set_services(svc)

        with patch("app.main.config") as mock_cfg:
            mock_cfg.RATE_LIMIT_RPM = 3
            mock_cfg.API_KEY = ""
            mock_cfg.REQUIRE_AUTH = False
            mock_cfg.TRUSTED_PROXY = None
            client_low = TestClient(app)
            statuses = [client_low.get("/api/status").status_code for _ in range(6)]

        assert 429 in statuses, "Low limit (3 rpm) should trigger 429 within 6 requests"


# ---------------------------------------------------------------------------
# Auth failure dict eviction (bounded growth)
# ---------------------------------------------------------------------------

class TestAuthFailureEviction:
    """Verify auth failure tracking dicts are bounded and evict properly."""

    @pytest.fixture(autouse=True)
    def _clear_auth_state(self):
        """Reset auth state between tests."""
        import app.auth as auth_mod
        auth_mod._auth_failures.clear()
        auth_mod._lockouts.clear()
        yield
        auth_mod._auth_failures.clear()
        auth_mod._lockouts.clear()

    def test_failures_dict_is_regular_dict(self):
        """_auth_failures must be a regular dict, not defaultdict."""
        import app.auth as auth_mod
        from collections import defaultdict
        assert type(auth_mod._auth_failures) is dict
        assert not isinstance(auth_mod._auth_failures, defaultdict)

    def test_check_rate_limit_no_phantom_entries(self):
        """_check_rate_limit should not create entries for IPs with no failures."""
        import app.auth as auth_mod
        auth_mod._check_rate_limit("192.168.1.100")
        # IP with no failures should not appear in the dict
        assert "192.168.1.100" not in auth_mod._auth_failures

    def test_record_failure_creates_entry(self):
        """_record_failure should create an entry for the IP."""
        import app.auth as auth_mod
        auth_mod._record_failure("10.0.0.1")
        assert "10.0.0.1" in auth_mod._auth_failures

    def test_eviction_under_max_cap(self):
        """When dict exceeds max IPs, oldest entries are evicted."""
        import app.auth as auth_mod
        import time

        max_ips = 5
        now = time.time()

        # Fill with max_ips + 3 entries
        for i in range(max_ips + 3):
            ip = f"10.0.0.{i}"
            auth_mod._auth_failures[ip] = [now]

        assert len(auth_mod._auth_failures) == max_ips + 3

        # Eviction should trim to max_ips
        auth_mod._evict_oldest(auth_mod._auth_failures, max_ips)
        assert len(auth_mod._auth_failures) <= max_ips

        # The oldest entries (lowest IPs) should have been evicted
        assert "10.0.0.0" not in auth_mod._auth_failures
        assert "10.0.0.1" not in auth_mod._auth_failures
        assert "10.0.0.2" not in auth_mod._auth_failures

    def test_eviction_runs_on_every_auth_check(self):
        """_check_rate_limit calls _evict_oldest to cap dict size."""
        import app.auth as auth_mod
        import time

        now = time.time()
        # Add many entries, simulating many failing IPs
        for i in range(20):
            auth_mod._auth_failures[f"10.0.0.{i}"] = [now]

        # Temporarily lower the cap by patching the config module in auth
        with patch("app.auth.config") as mock_cfg:
            mock_cfg.AUTH_MAX_TRACKED_IPS = 10
            mock_cfg.AUTH_LOCKOUT_SECONDS = 300
            auth_mod._check_rate_limit("192.168.1.1")

        assert len(auth_mod._auth_failures) <= 10

    def test_lockouts_dict_also_evicted(self):
        """Lockout dict is also bounded by _evict_oldest."""
        import app.auth as auth_mod
        import time

        now = time.time()
        for i in range(15):
            auth_mod._lockouts[f"10.0.0.{i}"] = now + 600  # locked for 10 min

        auth_mod._evict_oldest(auth_mod._lockouts, 5)
        assert len(auth_mod._lockouts) <= 5

    def test_expired_entries_cleaned_up(self):
        """_cleanup_expired_entries removes stale failure entries."""
        import app.auth as auth_mod
        import time

        old = time.time() - 1000  # Way past the lockout window
        auth_mod._auth_failures["stale_ip"] = [old]
        auth_mod._lockouts["expired_ip"] = old  # Already expired

        auth_mod._cleanup_expired_entries()

        assert "stale_ip" not in auth_mod._auth_failures
        assert "expired_ip" not in auth_mod._lockouts

    def test_record_failure_triggers_lockout(self):
        """After enough failures, IP gets locked out and failures cleared."""
        import app.auth as auth_mod
        from app.config import config as app_config

        max_failures = app_config.AUTH_MAX_FAILURES
        for _ in range(max_failures):
            auth_mod._record_failure("attacker")

        # Should be locked out
        assert "attacker" in auth_mod._lockouts
        # Failures should be cleared (not in dict anymore)
        assert "attacker" not in auth_mod._auth_failures


# ---------------------------------------------------------------------------
# Access Tiers (from test_access_tiers)
# ---------------------------------------------------------------------------

from pathlib import Path

from app.core.access_tiers import (
    get_blocked_builtins,
    get_blocked_imports,
    get_blocked_shell_commands,
    is_path_allowed,
)


def _with_tier(tier: str):
    """Patch config to use a specific access tier."""
    return patch("app.core.access_tiers.config",
                 type("C", (), {"SYSTEM_ACCESS_LEVEL": tier})())


class TestShellTiers:
    def test_sandboxed_blocks_interpreters(self):
        with _with_tier("sandboxed"):
            blocked = get_blocked_shell_commands()
            assert "python" in blocked
            assert "python3" in blocked
            assert "node" in blocked

    def test_standard_allows_interpreters(self):
        with _with_tier("standard"):
            blocked = get_blocked_shell_commands()
            assert "python" not in blocked
            assert "python3" not in blocked

    def test_standard_blocks_system(self):
        with _with_tier("standard"):
            blocked = get_blocked_shell_commands()
            assert "shutdown" in blocked
            assert "systemctl" in blocked

    def test_full_only_blocks_escape(self):
        with _with_tier("full"):
            blocked = get_blocked_shell_commands()
            assert "docker" in blocked
            assert "nsenter" in blocked
            assert "chroot" in blocked
            assert "shutdown" not in blocked
            assert "python" not in blocked

    def test_container_escape_always_blocked(self):
        for tier in ("sandboxed", "standard", "full"):
            with _with_tier(tier):
                blocked = get_blocked_shell_commands()
                assert "docker" in blocked
                assert "nsenter" in blocked


class TestFilesystemTiers:
    def test_sandboxed_allows_data(self):
        with _with_tier("sandboxed"):
            assert is_path_allowed(Path("/data/test.txt"), write=False)
            assert is_path_allowed(Path("/data/test.txt"), write=True)

    def test_sandboxed_blocks_etc(self):
        with _with_tier("sandboxed"):
            assert not is_path_allowed(Path("/etc/hosts"), write=False)
            assert not is_path_allowed(Path("/etc/hosts"), write=True)

    def test_standard_reads_anywhere(self):
        with _with_tier("standard"):
            assert is_path_allowed(Path("/tmp/test.txt"), write=False)
            assert is_path_allowed(Path("/usr/bin/env"), write=False)

    def test_standard_writes_limited(self):
        with _with_tier("standard"):
            assert is_path_allowed(Path("/data/test.txt"), write=True)
            assert is_path_allowed(Path("/tmp/test.txt"), write=True)
            assert is_path_allowed(Path("/home/nova/test.txt"), write=True)
            assert not is_path_allowed(Path("/etc/hosts"), write=True)

    def test_full_writes_most(self):
        with _with_tier("full"):
            assert is_path_allowed(Path("/data/test.txt"), write=True)
            assert is_path_allowed(Path("/tmp/test.txt"), write=True)

    def test_shadow_always_protected(self):
        for tier in ("sandboxed", "standard", "full"):
            with _with_tier(tier):
                assert not is_path_allowed(Path("/etc/shadow"), write=True)

    def test_ssh_always_protected(self):
        for tier in ("sandboxed", "standard", "full"):
            with _with_tier(tier):
                assert not is_path_allowed(Path("/root/.ssh/id_rsa"), write=True)


class TestCodeExecTiers:
    def test_sandboxed_blocks_os(self):
        with _with_tier("sandboxed"):
            blocked = get_blocked_imports()
            assert "os" in blocked
            assert "subprocess" in blocked
            assert "pathlib" in blocked

    def test_standard_allows_os_pathlib(self):
        with _with_tier("standard"):
            blocked = get_blocked_imports()
            assert "os" not in blocked
            assert "pathlib" not in blocked
            assert "glob" not in blocked
            assert "sys" not in blocked
            assert "subprocess" in blocked  # still blocked

    def test_full_minimal_blocks(self):
        with _with_tier("full"):
            blocked = get_blocked_imports()
            assert "ctypes" in blocked
            assert "multiprocessing" in blocked
            assert "os" not in blocked
            assert "subprocess" not in blocked

    def test_sandboxed_blocks_builtins(self):
        with _with_tier("sandboxed"):
            blocked = get_blocked_builtins()
            assert "eval(" in blocked
            assert "exec(" in blocked
            assert "__import__" in blocked

    def test_full_minimal_builtins(self):
        with _with_tier("full"):
            blocked = get_blocked_builtins()
            assert "__import__" in blocked  # always blocked
            assert "eval(" not in blocked   # allowed at full


class TestAccessTierConfigValidation:
    def test_invalid_access_level_warns(self):
        from app.config import Config
        cfg = Config(SYSTEM_ACCESS_LEVEL="stanard")
        warnings = cfg.validate()
        assert any("SYSTEM_ACCESS_LEVEL" in w for w in warnings)

    def test_valid_access_levels_no_warning(self):
        from app.config import Config
        for tier in ("sandboxed", "standard", "full"):
            cfg = Config(SYSTEM_ACCESS_LEVEL=tier)
            warnings = cfg.validate()
            assert not any("SYSTEM_ACCESS_LEVEL" in w for w in warnings)

    def test_invalid_tier_falls_back_to_sandboxed(self):
        with _with_tier("typo"):
            blocked = get_blocked_shell_commands()
            # Falls back to sandboxed, which blocks interpreters
            assert "python" in blocked


# ---------------------------------------------------------------------------
# Finding #10: Config orphan cleanup / unimplemented-provider warning
# ---------------------------------------------------------------------------

class TestConfigOrphansAndValidation:
    """Verify removed orphan fields are gone and validate() catches silent failures."""

    def test_temperature_internal_removed(self):
        """TEMPERATURE_INTERNAL was unused (llm.py hard-codes defaults) — must not exist."""
        from app.config import Config
        assert not hasattr(Config(), "TEMPERATURE_INTERNAL"), (
            "TEMPERATURE_INTERNAL is an orphan — llm.py never reads it. Remove from Config."
        )

    def test_temperature_reflexion_removed(self):
        """TEMPERATURE_REFLEXION was unused — must not exist."""
        from app.config import Config
        assert not hasattr(Config(), "TEMPERATURE_REFLEXION")

    def test_openai_use_completion_tokens_removed(self):
        """OPENAI_USE_COMPLETION_TOKENS has no reader in app code — must not exist."""
        from app.config import Config
        assert not hasattr(Config(), "OPENAI_USE_COMPLETION_TOKENS")

    def test_anthropic_api_version_removed(self):
        """ANTHROPIC_API_VERSION has no reader in app code — must not exist."""
        from app.config import Config
        assert not hasattr(Config(), "ANTHROPIC_API_VERSION")

    def test_anthropic_beta_header_removed(self):
        """ANTHROPIC_BETA_HEADER has no reader in app code — must not exist."""
        from app.config import Config
        assert not hasattr(Config(), "ANTHROPIC_BETA_HEADER")

    def test_validate_warns_on_unimplemented_provider_anthropic(self):
        """validate() must warn when LLM_PROVIDER=anthropic but the module doesn't exist."""
        from app.config import Config
        cfg = Config(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-fake")
        warnings = cfg.validate()
        assert any("anthropic" in w.lower() and "not exist" in w.lower() for w in warnings), (
            f"Expected warning about missing anthropic module. Got: {warnings}"
        )

    def test_validate_warns_on_unimplemented_provider_openai(self):
        """validate() must warn when LLM_PROVIDER=openai but the module doesn't exist."""
        from app.config import Config
        cfg = Config(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-fake")
        warnings = cfg.validate()
        assert any("openai" in w.lower() and "not exist" in w.lower() for w in warnings)

    def test_validate_no_warning_for_ollama(self):
        """validate() must NOT warn about provider when LLM_PROVIDER=ollama (implemented)."""
        from app.config import Config
        cfg = Config(LLM_PROVIDER="ollama")
        warnings = cfg.validate()
        assert not any("not exist" in w.lower() for w in warnings)

    def test_base_urls_still_present(self):
        """Provider base URLs must be kept — they'll be needed when providers are implemented."""
        from app.config import Config
        cfg = Config()
        assert hasattr(cfg, "OPENAI_BASE_URL")
        assert hasattr(cfg, "ANTHROPIC_BASE_URL")
        assert hasattr(cfg, "GOOGLE_BASE_URL")
