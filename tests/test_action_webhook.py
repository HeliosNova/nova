"""Tests for action_webhook — URL allowlist, SSRF, redirects, errors."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.tools.action_webhook import (
    WebhookTool,
    _is_url_allowed,
    _check_rate_limit,
    _webhook_timestamps,
    _WEBHOOK_RATE_LIMIT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_webhook_rate_limit():
    """Clear rate limit state between tests."""
    _webhook_timestamps.clear()
    yield
    _webhook_timestamps.clear()


def _enabled_config():
    """Return a mock config with webhooks enabled."""
    mock_cfg = MagicMock()
    mock_cfg.ENABLE_WEBHOOKS = True
    mock_cfg.WEBHOOK_ALLOWED_URLS = "https://hooks.example.com,https://api.myserver.com"
    mock_cfg.DB_PATH = ":memory:"
    return mock_cfg


# ===========================================================================
# URL allowlist enforcement
# ===========================================================================


class TestURLAllowlist:
    def test_allowed_url_matches_prefix(self):
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://hooks.example.com,https://api.myserver.com"
            assert _is_url_allowed("https://hooks.example.com/services/T00/B00")
            assert _is_url_allowed("https://api.myserver.com/deploy")

    def test_disallowed_url_rejected(self):
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://hooks.example.com"
            assert not _is_url_allowed("https://evil.com/hook")

    def test_empty_allowlist_blocks_all(self):
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.WEBHOOK_ALLOWED_URLS = ""
            assert not _is_url_allowed("https://hooks.example.com/anything")

    def test_path_boundary_prevents_prefix_bypass(self):
        """'/api' should not match '/api-internal' (path boundary check)."""
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://myserver.com/api"
            assert _is_url_allowed("https://myserver.com/api/deploy")
            assert _is_url_allowed("https://myserver.com/api")
            assert not _is_url_allowed("https://myserver.com/api-internal/secret")

    @pytest.mark.asyncio
    async def test_execute_blocks_non_allowlisted_url(self):
        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://myserver.com"
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="https://evil.com/hook")
        assert not result.success
        assert "not in the allowed" in result.error


# ===========================================================================
# SSRF blocking (private IP ranges)
# ===========================================================================


class TestSSRFBlocking:
    @pytest.mark.asyncio
    async def test_blocks_localhost(self):
        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "http://127.0.0.1"
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="http://127.0.0.1/admin")
        assert not result.success
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_private_10_range(self):
        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            # Allowlist matches the URL so the SSRF check is the one that blocks
            mock_cfg.WEBHOOK_ALLOWED_URLS = "http://10.0.0.1"
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="http://10.0.0.1/api")
        assert not result.success
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocks_192_168_range(self):
        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "http://192.168.1.1"
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="http://192.168.1.1/secret")
        assert not result.success
        assert "blocked" in result.error.lower()


# ===========================================================================
# Redirect following
# ===========================================================================


class TestRedirectFollowing:
    @pytest.mark.asyncio
    async def test_301_switches_to_get(self):
        """301/302/303 redirects should switch the method to GET."""
        # First response: 301 redirect
        redirect_resp = MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.status_code = 301
        redirect_resp.headers = {"location": "https://hooks.example.com/new-endpoint"}
        redirect_resp.url = httpx.URL("https://hooks.example.com/old-endpoint")

        # Final response after redirect
        final_resp = MagicMock()
        final_resp.is_redirect = False
        final_resp.status_code = 200
        final_resp.text = "OK"
        final_resp.reason_phrase = "OK"

        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=[redirect_resp, final_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.action_webhook._is_safe_url", return_value=True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await WebhookTool._call(
                    "https://hooks.example.com/old-endpoint",
                    "POST", {}, {"data": "test"},
                )

        # Second call should be GET (method changed by 301)
        second_call = mock_client.request.call_args_list[1]
        assert second_call[0][0] == "GET"
        assert "200" in result

    @pytest.mark.asyncio
    async def test_307_preserves_method(self):
        """307/308 redirects should preserve the original HTTP method."""
        redirect_resp = MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.status_code = 307
        redirect_resp.headers = {"location": "https://hooks.example.com/new-endpoint"}
        redirect_resp.url = httpx.URL("https://hooks.example.com/old-endpoint")

        final_resp = MagicMock()
        final_resp.is_redirect = False
        final_resp.status_code = 200
        final_resp.text = '{"ok": true}'
        final_resp.reason_phrase = "OK"

        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=[redirect_resp, final_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.action_webhook._is_safe_url", return_value=True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await WebhookTool._call(
                    "https://hooks.example.com/old-endpoint",
                    "PUT", {}, {"data": "test"},
                )

        # Second call should preserve PUT (307 preserves method)
        second_call = mock_client.request.call_args_list[1]
        assert second_call[0][0] == "PUT"

    @pytest.mark.asyncio
    async def test_308_preserves_method(self):
        """308 permanent redirect also preserves the original method."""
        redirect_resp = MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.status_code = 308
        redirect_resp.headers = {"location": "https://hooks.example.com/v2"}
        redirect_resp.url = httpx.URL("https://hooks.example.com/v1")

        final_resp = MagicMock()
        final_resp.is_redirect = False
        final_resp.status_code = 200
        final_resp.text = "done"
        final_resp.reason_phrase = "OK"

        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=[redirect_resp, final_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.action_webhook._is_safe_url", return_value=True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await WebhookTool._call(
                    "https://hooks.example.com/v1",
                    "PATCH", {}, {"key": "val"},
                )

        second_call = mock_client.request.call_args_list[1]
        assert second_call[0][0] == "PATCH"


# ===========================================================================
# Max redirect limit
# ===========================================================================


class TestMaxRedirects:
    @pytest.mark.asyncio
    async def test_too_many_redirects_raises(self):
        """Exceeding _max_redirects should raise ValueError."""
        redirect_resp = MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.status_code = 302
        redirect_resp.headers = {"location": "https://hooks.example.com/loop"}
        redirect_resp.url = httpx.URL("https://hooks.example.com/loop")

        mock_client = MagicMock()
        # Every request returns another redirect
        mock_client.request = AsyncMock(return_value=redirect_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.tools.action_webhook._is_safe_url", return_value=True):
            with patch("httpx.AsyncClient", return_value=mock_client):
                with pytest.raises(ValueError, match="Too many redirects"):
                    await WebhookTool._call(
                        "https://hooks.example.com/loop",
                        "GET", {}, None, _max_redirects=3,
                    )

    @pytest.mark.asyncio
    async def test_redirect_to_private_ip_blocked(self):
        """A redirect to a private IP should be blocked (SSRF via redirect)."""
        redirect_resp = MagicMock()
        redirect_resp.is_redirect = True
        redirect_resp.status_code = 302
        redirect_resp.headers = {"location": "http://169.254.169.254/latest/meta-data"}
        redirect_resp.url = httpx.URL("https://hooks.example.com/start")

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=redirect_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # First URL is safe, redirect target is not
        def safe_check(url):
            return "169.254" not in url

        with patch("app.tools.action_webhook._is_safe_url", side_effect=safe_check):
            with patch("httpx.AsyncClient", return_value=mock_client):
                with pytest.raises(ValueError, match="blocked"):
                    await WebhookTool._call(
                        "https://hooks.example.com/start",
                        "GET", {}, None,
                    )


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Connection errors should be caught and returned as transient failures."""
        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://hooks.example.com"

            with patch("app.tools.action_webhook._is_safe_url", return_value=True):
                with patch.object(
                    tool, "_call", new_callable=AsyncMock,
                    side_effect=httpx.ConnectError("Connection refused"),
                ):
                    with patch("app.tools.action_logging.get_db") as mock_db:
                        mock_db.return_value = MagicMock()
                        result = await tool.execute(
                            action="call",
                            url="https://hooks.example.com/down",
                        )
        assert not result.success
        assert "failed" in result.error.lower()
        assert result.error_category is not None
        assert result.error_category.value == "transient"

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Timeouts should be caught and returned as transient failures."""
        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://hooks.example.com"

            with patch("app.tools.action_webhook._is_safe_url", return_value=True):
                with patch.object(
                    tool, "_call", new_callable=AsyncMock,
                    side_effect=httpx.ReadTimeout("Read timed out"),
                ):
                    with patch("app.tools.action_logging.get_db") as mock_db:
                        mock_db.return_value = MagicMock()
                        result = await tool.execute(
                            action="call",
                            url="https://hooks.example.com/slow",
                        )
        assert not result.success
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_webhooks_disabled_returns_permission_error(self):
        """When ENABLE_WEBHOOKS is false, the tool should reject immediately."""
        tool = WebhookTool()
        # Default config has ENABLE_WEBHOOKS=false
        result = await tool.execute(action="call", url="https://hooks.example.com/hook")
        assert not result.success
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Filling the rate limit bucket should block further calls."""
        now = time.time()
        _webhook_timestamps.extend([now] * _WEBHOOK_RATE_LIMIT)

        tool = WebhookTool()
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://hooks.example.com"

            with patch("app.tools.action_webhook._is_safe_url", return_value=True):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(
                        action="call",
                        url="https://hooks.example.com/hook",
                    )
        assert not result.success
        assert "rate limit" in result.error.lower()


# ===========================================================================
# Rate limit helper
# ===========================================================================


class TestRateLimitHelper:
    def test_under_limit_passes(self):
        timestamps: list[float] = []
        assert _check_rate_limit(timestamps, 10, 3600) is True

    def test_at_limit_blocks(self):
        now = time.time()
        timestamps = [now] * 10
        assert _check_rate_limit(timestamps, 10, 3600) is False

    def test_expired_entries_pruned(self):
        old = time.time() - 7200  # 2 hours ago
        timestamps = [old] * 30
        assert _check_rate_limit(timestamps, 30, 3600) is True
        assert len(timestamps) == 0  # All pruned
