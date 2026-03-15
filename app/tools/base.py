"""Tool base class and registry.

All tools implement BaseTool. The ToolRegistry auto-generates
tool descriptions for the system prompt and dispatches execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    output: str
    success: bool = True
    error: str | None = None
    retriable: bool = False


def format_tool_error(name: str, message: str, retriable: bool = False) -> str:
    """Standard format for tool error messages."""
    tag = "retriable: yes" if retriable else "retriable: no"
    return f"[Tool error: {name}] {message} ({tag})"


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: str = ""  # Human-readable parameter description

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        ...


class ToolRegistry:
    """Registry of available tools. Generates descriptions and dispatches calls."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered — overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name. Returns the output string."""
        tool = self._tools.get(name)
        if not tool:
            return format_tool_error(name, f"Tool not found. Available: {', '.join(self._tools)}")

        try:
            result = await tool.execute(**args)
            if result.success:
                return result.output
            else:
                return format_tool_error(name, result.error or "Unknown error", retriable=result.retriable)
        except TypeError as e:
            logger.warning("Tool '%s' parameter mismatch: %s", name, e)
            return format_tool_error(name, f"Invalid parameters: {e}")
        except Exception as e:
            logger.exception("Tool '%s' raised exception", name)
            return format_tool_error(name, f"Unexpected: {e}", retriable=True)

    def get_descriptions(self) -> str:
        """Generate tool descriptions for the system prompt."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"{tool.name}({tool.parameters}) — {tool.description}")
        return "\n".join(lines)

    def get_tool_list(self) -> list[dict]:
        """Get tool metadata for validation."""
        return [{"name": t.name} for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
