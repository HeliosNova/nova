"""Tool base class and registry.

All tools implement BaseTool. The ToolRegistry auto-generates
tool descriptions for the system prompt and dispatches execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


# Canonical tool failure markers — single source of truth.
# Import from here in brain.py, critique.py, and anywhere else.
TOOL_FAILURE_MARKERS = ("failed", "timed out", "error:", "not available", "not found", "exception")


# ---------------------------------------------------------------------------
# Error categories for structured failure handling
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    """Categorizes tool errors for structured failure detection."""
    TRANSIENT = "transient"      # Network timeout, HTTP 429/5xx, connection refused
    VALIDATION = "validation"    # Missing/invalid params, bad format, unsafe input
    PERMISSION = "permission"    # Feature disabled, access tier blocked, rate limited
    NOT_FOUND = "not_found"      # File/entity/results not found
    INTERNAL = "internal"        # Unexpected exception, bug


@dataclass
class ToolResult:
    """Result of a tool execution."""
    output: str
    success: bool = True
    error: str | None = None
    retriable: bool = False
    error_category: ErrorCategory | None = None


def format_tool_error(
    name: str,
    message: str,
    retriable: bool = False,
    category: ErrorCategory | None = None,
) -> str:
    """Standard format for tool error messages."""
    tag = "retriable: yes" if retriable else "retriable: no"
    cat = f" [{category.value}]" if category else ""
    return f"[Tool error: {name}]{cat} {message} ({tag})"


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: str = ""  # Human-readable parameter description
    input_schema: dict | None = None  # JSON Schema for cloud providers

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        ...

    def trim_output(self, output: str) -> str:
        """Trim tool output for context window storage.

        Default: truncate to 2000 chars. Tools override for
        intelligent per-tool trimming.
        """
        if len(output) <= 2000:
            return output
        return output[:2000] + "\n[...truncated]"


# ---------------------------------------------------------------------------
# Tool hooks — pre/post execution callbacks
# ---------------------------------------------------------------------------

class ToolHook(Protocol):
    """Protocol for tool execution hooks (pre/post callbacks)."""

    async def pre_execute(self, tool_name: str, args: dict) -> None: ...
    async def post_execute(self, tool_name: str, args: dict, result: ToolResult) -> None: ...


class AuditLogHook:
    """Default hook that logs tool name + args on pre_execute."""

    async def pre_execute(self, tool_name: str, args: dict) -> None:
        logger.info("Tool audit: executing '%s' with args=%s", tool_name, args)

    async def post_execute(self, tool_name: str, args: dict, result: ToolResult) -> None:
        # Log tool executions to action_log for the audit trail
        try:
            from app.tools.action_logging import log_action
            output = (result.output or result.error or "")[:500]
            log_action(f"tool:{tool_name}", args, output, result.success)
        except Exception:
            pass


class ToolRegistry:
    """Registry of available tools. Generates descriptions and dispatches calls."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._hooks: list[ToolHook] = [AuditLogHook()]

    def register_hook(self, hook: ToolHook) -> None:
        """Register a tool execution hook."""
        self._hooks.append(hook)

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

        for hook in self._hooks:
            try:
                await hook.pre_execute(name, args)
            except Exception:
                pass

        try:
            result = await tool.execute(**args)
            for hook in self._hooks:
                try:
                    await hook.post_execute(name, args, result)
                except Exception:
                    pass
            if result.success:
                return result.output
            else:
                return format_tool_error(
                    name, result.error or "Unknown error",
                    retriable=result.retriable,
                    category=result.error_category,
                )
        except TypeError as e:
            logger.warning("Tool '%s' parameter mismatch: %s", name, e)
            return format_tool_error(name, f"Invalid parameters: {e}", category=ErrorCategory.VALIDATION)
        except Exception as e:
            logger.exception("Tool '%s' raised exception", name)
            return format_tool_error(name, f"Unexpected: {e}", retriable=True, category=ErrorCategory.INTERNAL)

    async def execute_full(self, name: str, args: dict) -> tuple[str, ToolResult | None]:
        """Execute a tool and return both the formatted output and the raw ToolResult.

        Used by brain.py for structured failure detection instead of substring matching.
        """
        tool = self._tools.get(name)
        if not tool:
            msg = format_tool_error(name, f"Tool not found. Available: {', '.join(self._tools)}")
            return msg, ToolResult(output="", success=False, error="Tool not found", error_category=ErrorCategory.NOT_FOUND)

        for hook in self._hooks:
            try:
                await hook.pre_execute(name, args)
            except Exception:
                pass

        try:
            result = await tool.execute(**args)
            for hook in self._hooks:
                try:
                    await hook.post_execute(name, args, result)
                except Exception:
                    pass
            if result.success:
                return result.output, result
            else:
                msg = format_tool_error(
                    name, result.error or "Unknown error",
                    retriable=result.retriable,
                    category=result.error_category,
                )
                return msg, result
        except TypeError as e:
            logger.warning("Tool '%s' parameter mismatch: %s", name, e)
            msg = format_tool_error(name, f"Invalid parameters: {e}", category=ErrorCategory.VALIDATION)
            return msg, ToolResult(output="", success=False, error=str(e), error_category=ErrorCategory.VALIDATION)
        except Exception as e:
            logger.exception("Tool '%s' raised exception", name)
            msg = format_tool_error(name, f"Unexpected: {e}", retriable=True, category=ErrorCategory.INTERNAL)
            return msg, ToolResult(output="", success=False, error=str(e), retriable=True, error_category=ErrorCategory.INTERNAL)

    def get_descriptions(self) -> str:
        """Generate tool descriptions for the system prompt."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"{tool.name}({tool.parameters}) — {tool.description}")
        return "\n".join(lines)

    def get_tool_list(self) -> list[dict]:
        """Get tool metadata for cloud provider tool calling."""
        result = []
        for t in self._tools.values():
            entry: dict = {"name": t.name, "description": t.description}
            if t.input_schema is not None:
                entry["parameters"] = t.input_schema
            result.append(entry)
        return result

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
