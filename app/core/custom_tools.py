"""Dynamic Tool Creation — create, store, and execute user-defined tools.

Tools are Python scripts persisted in SQLite. They execute in the same
subprocess sandbox as CodeExecTool, reusing _check_code_safety.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import config
from app.tools.base import BaseTool, ToolResult
from app.tools.code_exec import _check_code_safety

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema + Data types
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS custom_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL,
    code TEXT NOT NULL,
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class CustomToolRecord:
    id: int
    name: str
    description: str
    parameters: str  # JSON string: [{"name": "x", "type": "str"}, ...]
    code: str
    times_used: int = 0
    success_rate: float = 1.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# CustomToolStore — CRUD for custom tools
# ---------------------------------------------------------------------------

class CustomToolStore:
    """SQLite-backed store for user-created tools."""

    MAX_CODE_LENGTH = 5000
    MAX_TOOLS = 50

    def __init__(self, db):
        self._db = db
        self._db.execute(_SCHEMA.strip())

    def create_tool(
        self,
        name: str,
        description: str,
        parameters: str,
        code: str,
    ) -> int:
        """Create a new custom tool. Returns tool ID, or -1 on failure."""
        name = name.strip().lower().replace(" ", "_")
        if not name or len(name) > 50:
            logger.warning("Invalid tool name: %r", name)
            return -1

        # Check name uniqueness
        existing = self._db.fetchone(
            "SELECT id FROM custom_tools WHERE name = ?", (name,)
        )
        if existing:
            logger.warning("Tool '%s' already exists", name)
            return -1

        # Validate code safety
        safety_error = _check_code_safety(code)
        if safety_error:
            logger.warning("Tool '%s' code blocked: %s", name, safety_error)
            return -1

        # Size limits
        if len(code) > self.MAX_CODE_LENGTH:
            logger.warning("Tool '%s' code too long: %d chars", name, len(code))
            return -1

        # Tool count limit
        count = self._db.fetchone("SELECT COUNT(*) as c FROM custom_tools")
        if count and count["c"] >= self.MAX_TOOLS:
            logger.warning("Tool limit reached (%d)", self.MAX_TOOLS)
            return -1

        # Validate parameters JSON
        try:
            if isinstance(parameters, list):
                parameters = json.dumps(parameters)
            elif isinstance(parameters, str):
                json.loads(parameters)  # validate it's valid JSON
        except (json.JSONDecodeError, TypeError):
            parameters = "[]"

        cursor = self._db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) VALUES (?, ?, ?, ?)",
            (name, description[:500], parameters, code),
        )
        return cursor.lastrowid

    def get_tool(self, name: str) -> CustomToolRecord | None:
        """Get a tool by name."""
        row = self._db.fetchone(
            "SELECT * FROM custom_tools WHERE name = ? AND enabled = 1", (name,)
        )
        if not row:
            return None
        return CustomToolRecord(
            id=row["id"], name=row["name"], description=row["description"],
            parameters=row["parameters"], code=row["code"],
            times_used=row["times_used"], success_rate=row["success_rate"],
            enabled=bool(row["enabled"]),
        )

    def get_all_tools(self) -> list[CustomToolRecord]:
        """Get all enabled tools."""
        rows = self._db.fetchall(
            "SELECT * FROM custom_tools WHERE enabled = 1 ORDER BY name"
        )
        return [
            CustomToolRecord(
                id=r["id"], name=r["name"], description=r["description"],
                parameters=r["parameters"], code=r["code"],
                times_used=r["times_used"], success_rate=r["success_rate"],
                enabled=bool(r["enabled"]),
            )
            for r in rows
        ]

    def record_use(self, name: str, success: bool) -> str | None:
        """Record a tool usage. Auto-disables if success rate drops below 0.3 after 5+ uses.

        Returns a warning message if the tool was auto-disabled, None otherwise.
        """
        row = self._db.fetchone(
            "SELECT times_used, success_rate FROM custom_tools WHERE name = ?", (name,)
        )
        if not row:
            return None

        old_uses = row["times_used"]
        old_rate = row["success_rate"]
        new_uses = old_uses + 1

        # Running average
        new_rate = ((old_rate * old_uses) + (1.0 if success else 0.0)) / new_uses

        self._db.execute(
            "UPDATE custom_tools SET times_used = ?, success_rate = ? WHERE name = ?",
            (new_uses, round(new_rate, 3), name),
        )

        # Auto-disable if consistently failing
        if new_uses >= 5 and new_rate < 0.3:
            self._db.execute(
                "UPDATE custom_tools SET enabled = 0 WHERE name = ?", (name,)
            )
            msg = f"Tool '{name}' auto-disabled (success rate {new_rate*100:.0f}% after {new_uses} uses)"
            logger.info(msg)
            return msg
        return None

    def delete_tool(self, name: str) -> bool:
        """Delete a tool by name."""
        cursor = self._db.execute("DELETE FROM custom_tools WHERE name = ?", (name,))
        return cursor.rowcount > 0

    def toggle_tool(self, name: str, enabled: bool) -> bool:
        """Enable or disable a tool."""
        cursor = self._db.execute(
            "UPDATE custom_tools SET enabled = ? WHERE name = ?", (int(enabled), name)
        )
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# DynamicTool — wraps a CustomToolRecord as a BaseTool
# ---------------------------------------------------------------------------

class DynamicTool(BaseTool):
    """Executes a user-defined Python tool in a subprocess sandbox."""

    def __init__(self, record: CustomToolRecord, store: CustomToolStore):
        self.name = record.name
        self.description = record.description
        self.parameters = record.parameters
        self._code = record.code
        self._store = store

    async def execute(self, **kwargs) -> ToolResult:
        """Build and execute the tool's Python script."""
        # Re-check safety (in case code was tampered with in DB)
        safety_error = _check_code_safety(self._code)
        if safety_error:
            return ToolResult(output="", success=False, error=safety_error)

        # Validate args against declared schema
        try:
            declared = json.loads(self.parameters) if self.parameters else []
            declared_names = {p["name"] for p in declared if isinstance(p, dict)}
            if declared_names:
                unexpected = set(kwargs.keys()) - declared_names
                if unexpected:
                    return ToolResult(output="", success=False,
                        error=f"Unexpected parameters: {unexpected}. Expected: {declared_names}")
        except (json.JSONDecodeError, KeyError):
            pass  # Skip validation if schema is malformed

        # Build script: define the function, call it with provided args, print result
        args_json = json.dumps(kwargs)
        script = (
            f"{self._code}\n\n"
            f"import json as _json\n"
            f"_args = _json.loads({repr(args_json)})\n"
            f"_result = run(**_args)\n"
            f"print(_result)\n"
        )

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                script_path = f.name

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=config.CODE_EXEC_TIMEOUT,
                cwd=tempfile.gettempdir(),
            )

            try:
                Path(script_path).unlink()
            except OSError:
                pass

            success = result.returncode == 0
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            if not output.strip():
                output = "[Tool executed successfully with no output]"

            if len(output) > 5000:
                output = output[:5000] + "\n[... output truncated]"

            # Record usage
            self._store.record_use(self.name, success=success)

            if not success:
                return ToolResult(
                    output=output or result.stderr,
                    success=False,
                    error=f"Tool exited with code {result.returncode}",
                )

            return ToolResult(output=output, success=True)

        except subprocess.TimeoutExpired:
            self._store.record_use(self.name, success=False)
            return ToolResult(
                output="", success=False,
                error=f"Tool timed out after {config.CODE_EXEC_TIMEOUT}s",
            )
        except Exception as e:
            self._store.record_use(self.name, success=False)
            return ToolResult(output="", success=False, error=f"Tool failed: {e}")


# Tool-create prompt addition for the system prompt
TOOL_CREATE_DESCRIPTION = (
    "tool_create(name: str, description: str, parameters: str, code: str) "
    "— Create a new reusable tool. Write the tool as a Python function named 'run' "
    "that takes the declared parameters and returns a string result. "
    "Only create tools for capabilities you'll need repeatedly."
)
