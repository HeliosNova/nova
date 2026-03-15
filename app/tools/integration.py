"""Integration tool — call external APIs via registered templates."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote, urlencode

from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Set by main.py during startup
_registry = None


def set_registry(registry) -> None:
    global _registry
    _registry = registry


class IntegrationTool(BaseTool):
    name = "integration"
    description = (
        "Call a configured external service. "
        "Use for interacting with GitHub, Slack, Todoist, Home Assistant, etc."
    )
    parameters = "service: str, action: str, params: dict = {}"

    async def execute(
        self,
        *,
        service: str = "",
        action: str = "",
        params: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if not service or not action:
            return ToolResult(
                output="", success=False,
                error="Both 'service' and 'action' are required.",
            )

        if _registry is None:
            return ToolResult(
                output="", success=False,
                error="Integration registry not initialized.",
            )

        integration = _registry.get(service)
        if integration is None:
            available = ", ".join(_registry.get_configured_names())
            return ToolResult(
                output="", success=False,
                error=f"Unknown service '{service}'. Available: {available}",
            )

        if not integration.is_configured:
            return ToolResult(
                output="", success=False,
                error=f"Service '{service}' is not configured. Set {integration.auth_env_var} env var.",
            )

        # Find endpoint
        endpoint = None
        for ep in integration.endpoints:
            if ep.name == action:
                endpoint = ep
                break

        if endpoint is None:
            ep_names = ", ".join(ep.name for ep in integration.endpoints)
            return ToolResult(
                output="", success=False,
                error=f"Unknown action '{action}' for {service}. Available: {ep_names}",
            )

        # Check required params
        params = params or {}
        missing = [p for p in endpoint.required_params if p not in params]
        if missing:
            return ToolResult(
                output="", success=False,
                error=f"Missing required params: {', '.join(missing)}",
            )

        # Build URL with path params (encode values, reject path traversal)
        path = endpoint.path
        for key, value in params.items():
            str_value = str(value)
            if ".." in str_value or "/" in str_value or "\\" in str_value:
                return ToolResult(
                    output="", success=False,
                    error=f"Path parameter '{key}' contains disallowed characters",
                )
            path = path.replace(f"{{{key}}}", quote(str_value, safe=""))

        # Check for unfilled path params
        unfilled = re.findall(r"\{(\w+)\}", path)
        if unfilled:
            return ToolResult(
                output="", success=False,
                error=f"Missing path params: {', '.join(unfilled)}",
            )

        # Separate path params from extra params
        path_params = set(re.findall(r"\{(\w+)\}", endpoint.path))
        extra_params = {k: v for k, v in params.items() if k not in path_params}

        url = integration.base_url.rstrip("/") + path

        # For GET/HEAD, append extra params as query string
        if extra_params and endpoint.method in ("GET", "HEAD"):
            url = url + ("&" if "?" in url else "?") + urlencode(extra_params)

        # Build auth dict for http_fetch
        auth = None
        token = integration.get_token()
        if token and integration.auth_type == "bearer":
            auth = {"type": "bearer", "token": token}
        elif token and integration.auth_type == "basic":
            auth = {"type": "basic", "username": token, "password": ""}
        elif token and integration.auth_type == "api_key":
            auth = {"type": "api_key", "header": "Authorization", "key": token}

        # Build body from non-path params for non-GET
        body = None
        if endpoint.method not in ("GET", "HEAD") and extra_params:
            body = extra_params

        # Delegate to http_fetch
        from app.tools.http_fetch import HttpFetchTool

        logger.info(
            "Integration call: %s.%s → %s %s",
            service, action, endpoint.method, url,
        )

        fetcher = HttpFetchTool()
        result = await fetcher.execute(
            url=url,
            method=endpoint.method,
            body=body,
            auth=auth,
        )

        return result
