"""Integration registry — loads JSON templates and checks which have env tokens."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class Endpoint:
    name: str
    method: str
    path: str
    required_params: list[str]
    optional_params: list[str]


@dataclass
class Integration:
    name: str
    base_url: str
    auth_type: str          # bearer, basic, api_key, none
    auth_env_var: str       # e.g. GITHUB_TOKEN
    endpoints: list[Endpoint]
    description: str = ""

    @property
    def is_configured(self) -> bool:
        """True if the required env var has a non-empty value."""
        if not self.auth_env_var:
            return True  # No auth needed
        return bool(os.getenv(self.auth_env_var))

    def get_token(self) -> str:
        """Read the auth token from the environment."""
        if not self.auth_env_var:
            return ""
        return os.getenv(self.auth_env_var, "")


class IntegrationRegistry:
    """Discovers and loads integration templates from JSON files."""

    def __init__(self, templates_dir: Path | None = None):
        self._dir = templates_dir or TEMPLATES_DIR
        self._integrations: dict[str, Integration] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self._dir.is_dir():
            logger.warning("Integration templates dir not found: %s", self._dir)
            return

        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                endpoints = []
                for ep in data.get("endpoints", []):
                    endpoints.append(Endpoint(
                        name=ep["name"],
                        method=ep.get("method", "GET"),
                        path=ep["path"],
                        required_params=ep.get("required_params", []),
                        optional_params=ep.get("optional_params", []),
                    ))

                integration = Integration(
                    name=data["name"],
                    base_url=data["base_url"],
                    auth_type=data.get("auth_type", "none"),
                    auth_env_var=data.get("auth_env_var", ""),
                    endpoints=endpoints,
                    description=data.get("description", ""),
                )
                self._integrations[integration.name] = integration
                logger.debug("Loaded integration template: %s", integration.name)
            except Exception as e:
                logger.warning("Failed to load integration %s: %s", path.name, e)

        logger.info(
            "Loaded %d integration template(s), %d configured",
            len(self._integrations),
            len(self.get_configured()),
        )

    def get(self, name: str) -> Integration | None:
        return self._integrations.get(name)

    def get_all(self) -> list[Integration]:
        return list(self._integrations.values())

    def get_configured(self) -> list[Integration]:
        """Return only integrations that have their auth tokens set."""
        return [i for i in self._integrations.values() if i.is_configured]

    def get_configured_names(self) -> list[str]:
        return [i.name for i in self.get_configured()]

    def format_for_prompt(self) -> str:
        """Format configured integrations for the system prompt."""
        configured = self.get_configured()
        if not configured:
            return ""
        lines = ["## Available Integrations\n"]
        lines.append("Use the `integration` tool to call these services:\n")
        for integ in configured:
            ep_names = [ep.name for ep in integ.endpoints]
            lines.append(f"- **{integ.name}**: {', '.join(ep_names)}")
        return "\n".join(lines)
