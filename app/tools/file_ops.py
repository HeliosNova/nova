"""File operations tool — tier-aware filesystem access."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.access_tiers import is_path_allowed
from app.database import get_db
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


def _log_action(action_type: str, params: dict, result: str, success: bool) -> None:
    """Log an action to the action_log table."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO action_log (action_type, params, result, success) VALUES (?, ?, ?, ?)",
            (action_type, json.dumps(params, default=str), result[:2000], 1 if success else 0),
        )
    except Exception as e:
        logger.warning("Failed to log action: %s", e)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _safe_path(path_str: str, *, write: bool = False) -> Path | None:
    """Resolve path and ensure it's within allowed roots for the current tier."""
    try:
        # Anchor relative paths to /data for backward compatibility
        if not path_str.startswith("/"):
            target = (Path("/data") / path_str).resolve()
        else:
            target = Path(path_str).resolve()
        if not is_path_allowed(target, write=write):
            return None
        return target
    except Exception:
        return None


class FileOpsTool(BaseTool):
    name = "file_ops"
    description = "Read/write files. Access level depends on SYSTEM_ACCESS_LEVEL."
    parameters = "action: str, path: str, content: str"

    async def execute(
        self,
        *,
        action: str = "",
        path: str = "",
        content: str = "",
        **kwargs,
    ) -> ToolResult:
        if not action or not path:
            return ToolResult(
                output="",
                success=False,
                error="Both 'action' and 'path' are required.",
            )

        action = action.lower()

        if action == "read":
            return await self._read(path)
        elif action == "write":
            return await self._write(path, content)
        elif action == "list":
            return await self._list(path)
        elif action == "delete":
            return await self._delete(path)
        else:
            return ToolResult(
                output="",
                success=False,
                error=f"Unknown action '{action}'. Use: read, write, list, delete.",
            )

    async def _read(self, path_str: str) -> ToolResult:
        safe = _safe_path(path_str, write=False)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed at current access level.")
        if not safe.exists():
            return ToolResult(output="", success=False, error=f"File not found: {safe.name}")
        if not safe.is_file():
            return ToolResult(output="", success=False, error=f"Not a file: {safe.name}")
        if safe.stat().st_size > MAX_FILE_SIZE:
            return ToolResult(output="", success=False, error="File too large (>10MB)")

        try:
            text = safe.read_text(encoding="utf-8")
            if len(text) > 10000:
                text = text[:10000] + "\n[... truncated]"
            return ToolResult(output=text, success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Read error: {e}")

    async def _write(self, path_str: str, content: str) -> ToolResult:
        if not content:
            return ToolResult(output="", success=False, error="No content to write")
        if len(content) > MAX_FILE_SIZE:
            return ToolResult(output="", success=False, error="Content too large (>10MB)")

        safe = _safe_path(path_str, write=True)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed for writing at current access level.")

        # Protect sensitive directories from LLM writes
        for part in safe.parts:
            if part.lower() in self._PROTECTED_DIRS:
                return ToolResult(output="", success=False, error=f"Cannot write to '{part}/' directory (protected)")

        try:
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")
            return ToolResult(output=f"Written {len(content)} bytes to {safe.name}", success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Write error: {e}")

    async def _list(self, path_str: str) -> ToolResult:
        safe = _safe_path(path_str, write=False)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed at current access level.")
        if not safe.exists():
            return ToolResult(output="", success=False, error=f"Directory not found: {safe.name}")
        if not safe.is_dir():
            return ToolResult(output="", success=False, error=f"Not a directory: {safe.name}")

        try:
            entries = sorted(safe.iterdir())
            lines = []
            for entry in entries[:100]:  # Limit listing
                prefix = "d " if entry.is_dir() else "f "
                size = entry.stat().st_size if entry.is_file() else 0
                lines.append(f"{prefix}{entry.name} ({size} bytes)")
            return ToolResult(output="\n".join(lines) or "(empty directory)", success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=f"List error: {e}")

    _PROTECTED_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}
    _PROTECTED_FILES = {"nova.db", "training_data.jsonl"}
    _PROTECTED_DIRS = {"mcp"}  # Prevent LLM from writing MCP server configs

    async def _delete(self, path_str: str) -> ToolResult:
        safe = _safe_path(path_str, write=True)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed for deletion at current access level.")
        if not safe.exists():
            return ToolResult(output="", success=False, error=f"Not found: {safe.name}")

        # Protect critical files from deletion
        if safe.suffix.lower() in self._PROTECTED_EXTENSIONS:
            return ToolResult(output="", success=False, error=f"Cannot delete {safe.suffix} files (protected)")
        if safe.name.lower() in self._PROTECTED_FILES:
            return ToolResult(output="", success=False, error=f"Cannot delete '{safe.name}' (protected)")

        try:
            if safe.is_file():
                safe.unlink()
                _log_action("file_delete", {"path": str(safe)}, f"Deleted {safe.name}", True)
                return ToolResult(output=f"Deleted {safe.name}", success=True)
            else:
                return ToolResult(output="", success=False, error="Cannot delete directories")
        except Exception as e:
            _log_action("file_delete", {"path": str(safe)}, f"Delete error: {e}", False)
            return ToolResult(output="", success=False, error=f"Delete error: {e}")
