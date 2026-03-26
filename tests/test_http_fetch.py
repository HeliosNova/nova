"""Tests for the full HTTP client (http_fetch tool)."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.http_fetch import (
    HttpFetchTool,
    _build_auth_header,
    _is_safe_url,
    _is_private_ip,
    _sanitize_for_log,
    _get_client,
)


class TestSSRF:
    def test_blocks_localhost(self):
        assert not _is_safe_url("http://localhost/secret")

    def test_blocks_internal_ip(self):
        assert not _is_safe_url("http://10.0.0.1/api")
        assert not _is_safe_url("http://192.168.1.1/")

    def test_blocks_172_16(self):
        assert not _is_safe_url("http://172.16.0.1/")
        assert not _is_safe_url("http://172.17.0.1/")

    def test_blocks_172_19_through_31(self):
        """RFC 1918: 172.16.0.0/12 covers 172.16-31.*"""
        assert not _is_safe_url("http://172.19.0.1/")
        assert not _is_safe_url("http://172.24.0.1/")
        assert not _is_safe_url("http://172.31.255.254/")

    def test_allows_172_outside_range(self):
        """172.32.x.x is NOT private."""
        assert _is_safe_url("http://172.32.0.1/")

    def test_blocks_metadata(self):
        assert not _is_safe_url("http://169.254.169.254/")

    def test_blocks_loopback_range(self):
        assert not _is_safe_url("http://127.0.0.2/")
        assert not _is_safe_url("http://127.255.255.254/")

    def test_allows_external(self):
        """External IPs should pass."""
        # Use a literal public IP to avoid DNS
        assert _is_safe_url("https://8.8.8.8/")

    def test_blocks_non_http(self):
        assert not _is_safe_url("ftp://example.com/file")
        assert not _is_safe_url("file:///etc/passwd")

    def test_private_ip_check(self):
        assert _is_private_ip("10.0.0.1")
        assert _is_private_ip("172.20.0.1")
        assert _is_private_ip("192.168.0.1")
        assert _is_private_ip("127.0.0.1")
        assert not _is_private_ip("8.8.8.8")


class TestAuthHeaders:
    def test_bearer(self):
        h = _build_auth_header({"type": "bearer", "token": "abc123"})
        assert h == {"Authorization": "Bearer abc123"}

    def test_basic(self):
        h = _build_auth_header({"type": "basic", "username": "u", "password": "p"})
        expected = base64.b64encode(b"u:p").decode()
        assert h == {"Authorization": f"Basic {expected}"}

    def test_api_key(self):
        h = _build_auth_header({"type": "api_key", "header": "X-Token", "key": "k"})
        assert h == {"X-Token": "k"}

    def test_none_auth(self):
        assert _build_auth_header(None) == {}

    def test_unknown_type(self):
        assert _build_auth_header({"type": "oauth"}) == {}


class TestSanitizeLog:
    def test_masks_tokens(self):
        assert _sanitize_for_log({"type": "bearer", "token": "secret"}) == "bearer(***)"

    def test_none(self):
        assert _sanitize_for_log(None) == "none"


class TestHttpFetchTool:
    @pytest.fixture
    def tool(self):
        return HttpFetchTool()

    @pytest.mark.asyncio
    async def test_no_url(self, tool):
        r = await tool.execute()
        assert not r.success
        assert "No URL" in r.error

    @pytest.mark.asyncio
    async def test_bad_method(self, tool):
        r = await tool.execute(url="https://example.com", method="TRACE")
        assert not r.success
        assert "not allowed" in r.error

    @pytest.mark.asyncio
    async def test_ssrf_blocked(self, tool):
        r = await tool.execute(url="http://127.0.0.1/admin")
        assert not r.success
        assert "blocked" in r.error.lower()

    @pytest.mark.asyncio
    async def test_get_success(self, tool):
        mock_resp = MagicMock()
        mock_resp.text = "Hello World"
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_resp.headers = {}
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool.execute(url="https://example.com")
        assert r.success
        assert r.output == "Hello World"

    @pytest.mark.asyncio
    async def test_post_with_json_body(self, tool):
        mock_resp = MagicMock()
        mock_resp.text = '{"ok": true}'
        mock_resp.status_code = 200
        mock_resp.url = "https://api.example.com/data"
        mock_resp.headers = {}
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool.execute(
                url="https://api.example.com/data",
                method="POST",
                body={"key": "value"},
                auth={"type": "bearer", "token": "test"},
            )
        assert r.success
        # Verify the request was made with json body
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs.get("json") == {"key": "value"} or \
               call_kwargs[1].get("json") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_http_error_returns_body(self, tool):
        mock_resp = MagicMock()
        mock_resp.text = '{"error": "not found"}'
        mock_resp.status_code = 404
        mock_resp.url = "https://api.example.com/missing"
        mock_resp.headers = {}
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool.execute(url="https://api.example.com/missing")
        assert not r.success
        assert "404" in r.error
        assert "not found" in r.output

    @pytest.mark.asyncio
    async def test_response_truncation(self, tool):
        mock_resp = MagicMock()
        mock_resp.text = "x" * 60_000
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com/big"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool.execute(url="https://example.com/big")
        assert r.success
        assert len(r.output) < 55_000
        assert "truncated" in r.output

    @pytest.mark.asyncio
    async def test_backward_compatible_get_only(self, tool):
        """Old-style call with just url= should still work (GET, no auth)."""
        mock_resp = MagicMock()
        mock_resp.text = "ok"
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com"
        mock_resp.headers = {}
        mock_resp.is_redirect = False

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool.execute(url="https://example.com")
        assert r.success
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs.get("method") == "GET" or \
               call_kwargs[1].get("method") == "GET"
