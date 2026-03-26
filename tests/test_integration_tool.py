"""Tests for the integration tool — external service connections."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.base import ToolResult, ErrorCategory


@pytest.fixture(autouse=True)
def _set_tier():
    """IntegrationTool requires standard/full tier."""
    with patch("app.core.access_tiers.config") as mock_cfg:
        mock_cfg.SYSTEM_ACCESS_LEVEL = "standard"
        yield


@pytest.fixture
def integration_tool():
    from app.tools.integration import IntegrationTool
    return IntegrationTool()


async def test_integration_disabled(integration_tool, monkeypatch):
    """Integration tool returns error when disabled."""
    monkeypatch.setenv("ENABLE_INTEGRATIONS", "false")
    from app.config import reset_config
    reset_config()
    result = await integration_tool.execute(service="github", action="list_repos")
    assert not result.success


async def test_integration_missing_service(integration_tool):
    """Integration tool returns error for unknown service."""
    result = await integration_tool.execute(service="", action="list")
    assert not result.success
    assert result.error_category == ErrorCategory.VALIDATION


async def test_integration_missing_action(integration_tool):
    """Integration tool returns error for missing action."""
    result = await integration_tool.execute(service="github", action="")
    assert not result.success
    assert result.error_category == ErrorCategory.VALIDATION
