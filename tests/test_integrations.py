"""Tests for integration templates and integration tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations.registry import IntegrationRegistry, Integration, Endpoint
from app.tools.integration import IntegrationTool, set_registry


@pytest.fixture
def templates_dir(tmp_path):
    """Create a temp dir with test integration templates."""
    d = tmp_path / "templates"
    d.mkdir()

    github = {
        "name": "github",
        "base_url": "https://api.github.com",
        "auth_type": "bearer",
        "auth_env_var": "GITHUB_TOKEN",
        "endpoints": [
            {"name": "list_repos", "method": "GET", "path": "/user/repos"},
            {"name": "create_issue", "method": "POST",
             "path": "/repos/{owner}/{repo}/issues",
             "required_params": ["owner", "repo", "title"]},
        ],
    }
    (d / "github.json").write_text(json.dumps(github))

    noauth = {
        "name": "public_api",
        "base_url": "https://api.example.com",
        "auth_type": "none",
        "auth_env_var": "",
        "endpoints": [
            {"name": "status", "method": "GET", "path": "/status"},
        ],
    }
    (d / "public_api.json").write_text(json.dumps(noauth))

    weather = {
        "name": "weather",
        "base_url": "https://api.openweathermap.org",
        "auth_type": "api_key",
        "auth_env_var": "OWM_API_KEY",
        "endpoints": [
            {"name": "get_weather", "method": "GET", "path": "/data/2.5/weather",
             "required_params": ["q"]},
        ],
    }
    (d / "weather.json").write_text(json.dumps(weather))

    return d


class TestRegistry:
    def test_loads_templates(self, templates_dir):
        reg = IntegrationRegistry(templates_dir)
        assert len(reg.get_all()) == 3

    def test_get_by_name(self, templates_dir):
        reg = IntegrationRegistry(templates_dir)
        gh = reg.get("github")
        assert gh is not None
        assert gh.name == "github"
        assert len(gh.endpoints) == 2

    def test_configured_without_token(self, templates_dir, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        reg = IntegrationRegistry(templates_dir)
        configured = reg.get_configured()
        # Only public_api should be configured (no token needed)
        names = [i.name for i in configured]
        assert "public_api" in names
        assert "github" not in names

    def test_configured_with_token(self, templates_dir, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        reg = IntegrationRegistry(templates_dir)
        configured = reg.get_configured()
        names = [i.name for i in configured]
        assert "github" in names
        assert "public_api" in names

    def test_format_for_prompt(self, templates_dir, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        reg = IntegrationRegistry(templates_dir)
        text = reg.format_for_prompt()
        assert "github" in text
        assert "integration" in text.lower()

    def test_format_empty_when_none_configured(self, templates_dir, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        # Remove the public_api to test truly empty
        (templates_dir / "public_api.json").unlink()
        reg = IntegrationRegistry(templates_dir)
        assert reg.format_for_prompt() == ""

    def test_invalid_json_skipped(self, templates_dir):
        (templates_dir / "bad.json").write_text("not json")
        reg = IntegrationRegistry(templates_dir)
        # Should still load the 3 valid ones
        assert len(reg.get_all()) == 3


class TestIntegrationTool:
    @pytest.fixture
    def tool_with_registry(self, templates_dir, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        reg = IntegrationRegistry(templates_dir)
        set_registry(reg)
        return IntegrationTool()

    @pytest.mark.asyncio
    async def test_missing_service(self, tool_with_registry):
        r = await tool_with_registry.execute(service="", action="list")
        assert not r.success

    @pytest.mark.asyncio
    async def test_unknown_service(self, tool_with_registry):
        r = await tool_with_registry.execute(service="nonexistent", action="list")
        assert not r.success
        assert "Unknown service" in r.error

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool_with_registry):
        r = await tool_with_registry.execute(service="github", action="nonexistent")
        assert not r.success
        assert "Unknown action" in r.error

    @pytest.mark.asyncio
    async def test_missing_required_params(self, tool_with_registry):
        r = await tool_with_registry.execute(
            service="github", action="create_issue", params={"owner": "me"}
        )
        assert not r.success
        assert "Missing required" in r.error

    @pytest.mark.asyncio
    async def test_successful_call(self, tool_with_registry):
        mock_resp = MagicMock()
        mock_resp.text = '[{"name": "my-repo"}]'
        mock_resp.status_code = 200
        mock_resp.url = "https://api.github.com/user/repos"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool_with_registry.execute(service="github", action="list_repos")

        assert r.success
        assert "my-repo" in r.output

    @pytest.mark.asyncio
    async def test_path_params_replaced(self, tool_with_registry):
        mock_resp = MagicMock()
        mock_resp.text = '{"id": 1}'
        mock_resp.status_code = 201
        mock_resp.url = "https://api.github.com/repos/me/proj/issues"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool_with_registry.execute(
                service="github",
                action="create_issue",
                params={"owner": "me", "repo": "proj", "title": "Bug"},
            )

        assert r.success
        # Verify URL had path params replaced
        call_kwargs = mock_client.request.call_args
        used_url = call_kwargs.kwargs.get("url") or call_kwargs[1].get("url")
        assert "me" in used_url
        assert "proj" in used_url
        assert "{owner}" not in used_url

    @pytest.mark.asyncio
    async def test_get_query_params(self, templates_dir, monkeypatch):
        """Non-path params should be appended as query string for GET requests."""
        monkeypatch.setenv("OWM_API_KEY", "test_key")
        reg = IntegrationRegistry(templates_dir)
        set_registry(reg)
        tool = IntegrationTool()

        mock_resp = MagicMock()
        mock_resp.text = '{"temp": 20}'
        mock_resp.status_code = 200
        mock_resp.url = "https://api.openweathermap.org/data/2.5/weather?q=London"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.tools.http_fetch._get_client", return_value=mock_client):
            r = await tool.execute(
                service="weather",
                action="get_weather",
                params={"q": "London", "units": "metric"},
            )

        assert r.success
        call_kwargs = mock_client.request.call_args
        used_url = call_kwargs.kwargs.get("url") or call_kwargs[1].get("url")
        assert "q=London" in used_url
        assert "units=metric" in used_url
