"""File operations tool — tier-aware filesystem access."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.core.access_tiers import is_path_allowed
from app.tools.action_logging import log_action as _log_action
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _safe_path(path_str: str, *, write: bool = False) -> Path | None:
    """Resolve path and ensure it's within allowed roots for the current tier."""
    try:
        # Anchor relative paths to /data for backward compatibility
        # On Windows, also treat drive-letter paths (C:\...) as absolute
        if not path_str.startswith("/") and not re.match(r'^[A-Za-z]:', path_str):
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
    description = (
        "Read, write, list, or delete files on the local filesystem. "
        "Access level is controlled by SYSTEM_ACCESS_LEVEL tier. "
        "Reads return file content (truncated at 10,000 characters). Writes create parent directories automatically. "
        "Use for persistent data storage in /data/. "
        "Do NOT use for executing code (use code_exec) or running commands (use shell_exec)."
    )
    parameters = "action: str, path: str, content: str"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list", "delete"],
                "description": "File operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path. Relative paths are anchored to /data/.",
            },
            "content": {
                "type": "string",
                "description": "Content to write (required for write action).",
            },
        },
        "required": ["action", "path"],
    }

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
                error_category=ErrorCategory.VALIDATION,
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
                error_category=ErrorCategory.VALIDATION,
            )

    async def _read(self, path_str: str) -> ToolResult:
        safe = _safe_path(path_str, write=False)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed at current access level.", error_category=ErrorCategory.PERMISSION)
        for part in safe.parts:
            if part.lower() in self._PROTECTED_DIRS:
                return ToolResult(output="", success=False, error=f"Cannot access '{part}/' directory (protected)", error_category=ErrorCategory.PERMISSION)
        if not safe.exists():
            return ToolResult(output="", success=False, error=f"File not found: {safe.name}", error_category=ErrorCategory.NOT_FOUND)
        if not safe.is_file():
            return ToolResult(output="", success=False, error=f"Not a file: {safe.name}", error_category=ErrorCategory.VALIDATION)
        if safe.stat().st_size > MAX_FILE_SIZE:
            return ToolResult(output="", success=False, error="File too large (>10MB)", error_category=ErrorCategory.VALIDATION)

        try:
            text = safe.read_text(encoding="utf-8")
            if len(text) > 10000:
                text = text[:10000] + "\n[... truncated]"
            return ToolResult(output=text, success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Read error: {e}", error_category=ErrorCategory.INTERNAL)

    async def _write(self, path_str: str, content: str) -> ToolResult:
        if not content:
            return ToolResult(output="", success=False, error="No content to write", error_category=ErrorCategory.VALIDATION)
        if len(content) > MAX_FILE_SIZE:
            return ToolResult(output="", success=False, error="Content too large (>10MB)", error_category=ErrorCategory.VALIDATION)

        safe = _safe_path(path_str, write=True)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed for writing at current access level.", error_category=ErrorCategory.PERMISSION)

        # Protect sensitive directories from LLM writes
        for part in safe.parts:
            if part.lower() in self._PROTECTED_DIRS:
                return ToolResult(output="", success=False, error=f"Cannot write to '{part}/' directory (protected)", error_category=ErrorCategory.PERMISSION)

        # Protect critical file types and names from being overwritten
        if safe.suffix.lower() in self._PROTECTED_EXTENSIONS:
            return ToolResult(output="", success=False, error=f"Cannot write {safe.suffix} files (protected)", error_category=ErrorCategory.PERMISSION)
        if safe.name.lower() in self._PROTECTED_FILES:
            return ToolResult(output="", success=False, error=f"Cannot write to '{safe.name}' (protected)", error_category=ErrorCategory.PERMISSION)

        try:
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")
            return ToolResult(output=f"Written {len(content)} bytes to {safe.name}", success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=f"Write error: {e}", error_category=ErrorCategory.INTERNAL)

    async def _list(self, path_str: str) -> ToolResult:
        safe = _safe_path(path_str, write=False)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed at current access level.", error_category=ErrorCategory.PERMISSION)
        for part in safe.parts:
            if part.lower() in self._PROTECTED_DIRS:
                return ToolResult(output="", success=False, error=f"Cannot access '{part}/' directory (protected)", error_category=ErrorCategory.PERMISSION)
        if not safe.exists():
            return ToolResult(output="", success=False, error=f"Directory not found: {safe.name}", error_category=ErrorCategory.NOT_FOUND)
        if not safe.is_dir():
            return ToolResult(output="", success=False, error=f"Not a directory: {safe.name}", error_category=ErrorCategory.VALIDATION)

        try:
            entries = sorted(safe.iterdir())
            lines = []
            for entry in entries[:100]:  # Limit listing
                prefix = "d " if entry.is_dir() else "f "
                size = entry.stat().st_size if entry.is_file() else 0
                lines.append(f"{prefix}{entry.name} ({size} bytes)")
            return ToolResult(output="\n".join(lines) or "(empty directory)", success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=f"List error: {e}", error_category=ErrorCategory.INTERNAL)

    _PROTECTED_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".pk8", ".p12", ".env", ".secret", ".pfx", ".crt"}
    _PROTECTED_FILES = {"nova.db", "training_data.jsonl", "config_overrides.json", ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
    _PROTECTED_DIRS = {"mcp", ".ssh", ".gnupg"}  # Prevent LLM from writing MCP server configs / accessing sensitive dirs

    async def _delete(self, path_str: str) -> ToolResult:
        safe = _safe_path(path_str, write=True)
        if safe is None:
            return ToolResult(output="", success=False, error="Path not allowed for deletion at current access level.", error_category=ErrorCategory.PERMISSION)
        for part in safe.parts:
            if part.lower() in self._PROTECTED_DIRS:
                return ToolResult(output="", success=False, error=f"Cannot delete files in protected directory", error_category=ErrorCategory.PERMISSION)
        if not safe.exists():
            return ToolResult(output="", success=False, error=f"Not found: {safe.name}", error_category=ErrorCategory.NOT_FOUND)

        # Protect critical files from deletion
        if safe.suffix.lower() in self._PROTECTED_EXTENSIONS:
            return ToolResult(output="", success=False, error=f"Cannot delete {safe.suffix} files (protected)", error_category=ErrorCategory.PERMISSION)
        if safe.name.lower() in self._PROTECTED_FILES:
            return ToolResult(output="", success=False, error=f"Cannot delete '{safe.name}' (protected)", error_category=ErrorCategory.PERMISSION)

        try:
            if safe.is_file():
                safe.unlink()
                _log_action("file_delete", {"path": str(safe)}, f"Deleted {safe.name}", True)
                return ToolResult(output=f"Deleted {safe.name}", success=True)
            else:
                return ToolResult(output="", success=False, error="Cannot delete directories", error_category=ErrorCategory.VALIDATION)
        except Exception as e:
            _log_action("file_delete", {"path": str(safe)}, f"Delete error: {e}", False)
            return ToolResult(output="", success=False, error=f"Delete error: {e}", error_category=ErrorCategory.INTERNAL)
