"""Autonomous tool-creation trigger — detects repeated multi-step patterns.

When Nova uses 3+ tool rounds for similar queries repeatedly, it records
each occurrence as a "candidate". Once the same canonical tool sequence is
seen AUTO_TOOL_CREATION_THRESHOLD times (default 3) without an existing
custom tool or skill covering the pattern, a background task is spawned
that generates Python code and calls the existing tool_create pipeline.

Guards:
- tool_create is never in the triggered sequence (no infinite loops)
- If the query is already covered by a skill (regex or semantic), skip
- If a custom tool for this sequence already exists, skip
- Generated code passes the same _check_code_safety guard used by
  CustomToolStore.create_tool()
- Config flags gate both recording and triggering
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.config import config
from app.core import llm
from app.database import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB Schema
# ---------------------------------------------------------------------------

_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS auto_tool_candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    tool_sequence TEXT NOT NULL,
    sequence_key TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    triggered    INTEGER DEFAULT 0
)
"""

_SCHEMA_INDEX = """
CREATE INDEX IF NOT EXISTS idx_auto_tool_seq
    ON auto_tool_candidates(sequence_key, triggered)
"""

# Tools that must never appear in an auto-generated tool's triggering sequence.
# Prevents recursive / dangerous patterns.
_BLOCKED_TOOLS = frozenset({"tool_create", "delegate", "shell_exec"})

# Max candidates to retain per sequence_key (rolling window)
_MAX_CANDIDATES_PER_KEY = 50


# ---------------------------------------------------------------------------
# ToolCandidateStore — SQLite-backed candidate tracker
# ---------------------------------------------------------------------------

class ToolCandidateStore:
    """Tracks multi-step tool interactions as potential tool-creation candidates."""

    def __init__(self, db=None):
        self._db = db or get_db()
        self._db.execute(_SCHEMA_TABLE.strip())
        self._db.execute(_SCHEMA_INDEX.strip())

    @staticmethod
    def _sequence_key(tools: list[str]) -> str:
        """Canonical key for a tool set — sorted unique names joined by |."""
        return "|".join(sorted(set(tools)))

    def record(self, query: str, tools: list[str]) -> int:
        """Record a candidate interaction. Returns its row id."""
        seq_key = self._sequence_key(tools)
        cursor = self._db.execute(
            "INSERT INTO auto_tool_candidates (query, tool_sequence, sequence_key) VALUES (?, ?, ?)",
            (query[:500], json.dumps(tools), seq_key),
        )
        # Prune old untriggered entries beyond the rolling window
        self._db.execute(
            """DELETE FROM auto_tool_candidates
               WHERE sequence_key = ? AND triggered = 0 AND id NOT IN (
                   SELECT id FROM auto_tool_candidates
                   WHERE sequence_key = ? AND triggered = 0
                   ORDER BY id DESC LIMIT ?
               )""",
            (seq_key, seq_key, _MAX_CANDIDATES_PER_KEY),
        )
        return cursor.lastrowid

    def count_untriggered(self, tools: list[str], lookback_days: int = 30) -> int:
        """Count untriggered candidates for this tool sequence in the lookback window."""
        seq_key = self._sequence_key(tools)
        row = self._db.fetchone(
            """SELECT COUNT(*) AS c FROM auto_tool_candidates
               WHERE sequence_key = ?
                 AND triggered = 0
                 AND created_at >= datetime('now', ? || ' days')""",
            (seq_key, f"-{lookback_days}"),
        )
        return row["c"] if row else 0

    def get_example_queries(self, tools: list[str], limit: int = 5) -> list[str]:
        """Return recent example queries for this tool sequence."""
        seq_key = self._sequence_key(tools)
        rows = self._db.fetchall(
            """SELECT query FROM auto_tool_candidates
               WHERE sequence_key = ? AND triggered = 0
               ORDER BY id DESC LIMIT ?""",
            (seq_key, limit),
        )
        return [r["query"] for r in rows]

    def mark_triggered(self, tools: list[str]) -> None:
        """Mark all candidates for this sequence as triggered (suppress future fires)."""
        seq_key = self._sequence_key(tools)
        self._db.execute(
            "UPDATE auto_tool_candidates SET triggered = 1 WHERE sequence_key = ?",
            (seq_key,),
        )

    def is_already_triggered(self, tools: list[str]) -> bool:
        """Return True if tool creation was already attempted for this sequence."""
        seq_key = self._sequence_key(tools)
        row = self._db.fetchone(
            "SELECT 1 FROM auto_tool_candidates WHERE sequence_key = ? AND triggered = 1 LIMIT 1",
            (seq_key,),
        )
        return row is not None


# ---------------------------------------------------------------------------
# LLM-based tool code generator
# ---------------------------------------------------------------------------

async def _generate_tool_spec(
    example_queries: list[str],
    tool_sequence: list[str],
) -> dict | None:
    """Ask the LLM to synthesise a custom Python tool for recurring pattern.

    Returns a dict with keys: name, description, parameters (list), code.
    Returns None if the LLM decides it's not worth creating a tool.
    """
    queries_block = "\n".join(f"- {q}" for q in example_queries)
    tools_block = ", ".join(tool_sequence)

    prompt = (
        "You are designing a reusable Python tool to automate a multi-step pattern "
        "that Nova has performed repeatedly.\n\n"
        f"The pattern used these tools in sequence: {tools_block}\n\n"
        f"Example queries that triggered this pattern:\n{queries_block}\n\n"
        "Design a single Python function that encapsulates this pattern.\n"
        "The function MUST:\n"
        "- Be self-contained (no external state, no imports beyond stdlib)\n"
        "- Accept clear typed arguments\n"
        "- Return a plain string result\n"
        "- NOT call tool_create, shell_exec, or delegate\n\n"
        "Respond with JSON:\n"
        '{"name": "snake_case_name", '
        '"description": "One sentence description", '
        '"parameters": [{"name": "arg1", "type": "str", "description": "..."}], '
        '"code": "def run(arg1: str) -> str:\\n    ...\\n    return result"}\n\n'
        'If the pattern is NOT worth automating, respond: {"skip": true, "reason": "..."}'
    )

    try:
        raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=800,
                temperature=0.2,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
        obj = llm.extract_json_object(raw)
        if not obj or obj.get("skip"):
            logger.debug("Auto-tool LLM decided to skip: %s", obj.get("reason", "no reason"))
            return None

        name = obj.get("name", "").strip()
        code = obj.get("code", "").strip()
        description = obj.get("description", "").strip()
        parameters = obj.get("parameters", [])

        if not name or not code:
            logger.debug("Auto-tool LLM returned incomplete spec")
            return None

        # Extra guard: generated code must not reference tool_create
        if "tool_create" in code or "shell_exec" in code or "delegate" in code:
            logger.warning("Auto-tool LLM generated code with blocked tool references — rejected")
            return None

        return {"name": name, "description": description, "parameters": parameters, "code": code}

    except Exception as e:
        logger.debug("Auto-tool LLM call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main trigger — called from brain.py post-tool-loop
# ---------------------------------------------------------------------------

async def maybe_trigger_tool_creation(
    query: str,
    tool_results: list[dict],
    svc: "Services",  # type: ignore[name-defined]
    *,
    _candidate_store: ToolCandidateStore | None = None,
) -> None:
    """Check if this interaction should trigger autonomous tool creation.

    Called as a fire-and-forget background task from brain.py after the
    tool loop ends with 3+ rounds.

    Safety exits (in order):
    1. Feature disabled
    2. Custom tools not available
    3. Blocked tool in sequence
    4. Existing skill already covers the query (regex or semantic)
    5. Tool already created for this sequence
    6. Below recurrence threshold
    """
    if not config.ENABLE_AUTONOMOUS_TOOL_CREATION:
        return
    if not svc or not svc.custom_tools:
        return

    tools_used = [tr["tool"] for tr in tool_results]

    # Guard: skip if any blocked tool is in the sequence
    if _BLOCKED_TOOLS & set(tools_used):
        return

    # Guard: skip if an existing skill already covers this query
    if svc.skills:
        try:
            existing_skill = await asyncio.to_thread(svc.skills.get_matching_skill, query)
            if existing_skill:
                logger.debug(
                    "Auto-tool skipped: skill '%s' already covers query", existing_skill.name
                )
                return
        except Exception:
            pass  # If skill check fails, continue

    # Use shared DB — ToolCandidateStore initialises its own schema
    from app.database import get_db
    candidate_store = _candidate_store or ToolCandidateStore(get_db())

    # Guard: skip if already triggered for this sequence
    if candidate_store.is_already_triggered(tools_used):
        return

    # Record this occurrence
    candidate_store.record(query, tools_used)

    # Count recurrences
    count = candidate_store.count_untriggered(tools_used)
    threshold = config.AUTO_TOOL_CREATION_THRESHOLD
    if count < threshold:
        logger.debug(
            "Auto-tool candidate recorded (%d/%d): sequence=%s",
            count, threshold, ToolCandidateStore._sequence_key(tools_used),
        )
        return

    # Threshold reached — mark and fire generation
    candidate_store.mark_triggered(tools_used)
    example_queries = candidate_store.get_example_queries(tools_used)

    logger.info(
        "Auto-tool threshold reached (%d occurrences) — generating tool for sequence: %s",
        count, tools_used,
    )

    async def _safe_generate():
        try:
            spec = await _generate_tool_spec(example_queries, tools_used)
            if not spec:
                return

            name = spec["name"].strip().lower().replace(" ", "_")
            description = spec.get("description", "").strip()
            parameters = spec.get("parameters", [])
            code = spec.get("code", "").strip()

            if not name or not code:
                logger.debug("Auto-tool: spec missing name or code")
                return

            # Create via CustomToolStore (same path as _handle_tool_create)
            params_json = json.dumps(parameters) if isinstance(parameters, list) else parameters
            tool_id = svc.custom_tools.create_tool(name, description, params_json, code)
            if tool_id == -1:
                logger.debug("Auto-tool '%s' rejected by CustomToolStore", name)
                return

            # Register in live registry if available
            if svc.tool_registry:
                record = svc.custom_tools.get_tool(name)
                if record:
                    from app.core.custom_tools import DynamicTool
                    svc.tool_registry.register(DynamicTool(record, svc.custom_tools))

            logger.info("Auto-tool created: '%s' (id=%d)", name, tool_id)
        except Exception as e:
            logger.warning("Auto-tool generation failed: %s", e)

    asyncio.create_task(_safe_generate())
