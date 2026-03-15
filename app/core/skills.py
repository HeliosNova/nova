"""Skills — learned multi-step procedures from corrections.

When Nova gets corrected on a multi-step task, the correction handler can
extract a skill: trigger pattern + tool sequence + answer template.
Next time a matching query arrives, Nova follows the learned procedure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.config import config
from app.core import llm
from app.database import get_db

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    id: int
    name: str
    trigger_pattern: str
    steps: list[dict]           # [{"tool": "...", "args_template": {...}, "output_key": "..."}]
    answer_template: str | None  # Template with {output_key} placeholders
    learned_from: int | None
    times_used: int
    success_rate: float
    enabled: bool
    created_at: str | None


class SkillStore:
    """Store, match, and execute learned skills."""

    def __init__(self, db=None):
        self._db = db or get_db()

    def get_matching_skill(self, query: str) -> Skill | None:
        """Find the best matching skill for a query.

        Checks trigger_pattern (regex) against the query.
        When multiple skills match, returns the one with the longest regex
        pattern (more specific = higher priority).
        """
        rows = self._db.fetchall(
            "SELECT * FROM skills WHERE enabled = 1 ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT ?",
            (config.MAX_SKILLS_CHECK,),
        )

        matches = []
        for row in rows:
            try:
                pattern = row["trigger_pattern"]
                # Check for ReDoS-prone patterns before running
                if _is_redos_risk(pattern):
                    logger.warning("Skill trigger pattern rejected (ReDoS risk): %s", pattern)
                    self._db.execute("UPDATE skills SET enabled = 0 WHERE id = ?", (row["id"],))
                    continue
                if re.search(pattern, query, re.IGNORECASE):
                    matches.append(row)
            except re.error:
                # Invalid regex — skip
                logger.warning("Invalid skill trigger pattern: %s", row["trigger_pattern"])
                continue

        if not matches:
            return None

        # Sort by regex pattern length descending (longer = more specific = higher priority)
        matches.sort(key=lambda r: len(r["trigger_pattern"]), reverse=True)
        return self._row_to_skill(matches[0])

    def create_skill(
        self,
        name: str,
        trigger_pattern: str,
        steps: list[dict],
        answer_template: str | None = None,
        learned_from: int | None = None,
        initial_success_rate: float = 0.7,
    ) -> int | None:
        """Create a new skill. Returns skill ID, or None if rejected by guards."""
        # Guard: reject ReDoS-prone patterns
        if _is_redos_risk(trigger_pattern):
            logger.warning("Skill '%s' rejected: ReDoS risk (%s)", name, trigger_pattern)
            return None

        # Guard: reject overly broad trigger patterns
        if _is_too_broad(trigger_pattern):
            logger.warning("Skill '%s' rejected: trigger too broad (%s)", name, trigger_pattern)
            return None

        # Deduplication — if same trigger pattern exists, boost confidence
        existing = self._db.fetchone(
            "SELECT id, success_rate FROM skills WHERE trigger_pattern = ?",
            (trigger_pattern,),
        )
        if existing:
            new_rate = min(1.0, existing["success_rate"] + 0.1)
            self._db.execute(
                "UPDATE skills SET success_rate = ?, enabled = 1 WHERE id = ?",
                (new_rate, existing["id"]),
            )
            logger.info(
                "Skill dedup: boosted #%d confidence to %.2f (trigger: %s)",
                existing["id"], new_rate, trigger_pattern,
            )
            return existing["id"]

        # Name-based dedup — if same name (case-insensitive) exists, update it
        existing_by_name = self._db.fetchone(
            "SELECT id FROM skills WHERE LOWER(name) = LOWER(?)",
            (name,),
        )
        if existing_by_name:
            self._db.execute(
                "UPDATE skills SET trigger_pattern = ?, steps = ?, answer_template = ?, enabled = 1 WHERE id = ?",
                (trigger_pattern, json.dumps(steps), answer_template, existing_by_name["id"]),
            )
            logger.info(
                "Skill name-dedup: updated #%d '%s' with new trigger/steps",
                existing_by_name["id"], name,
            )
            return existing_by_name["id"]

        cursor = self._db.execute(
            """INSERT INTO skills (name, trigger_pattern, steps, answer_template, learned_from, success_rate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                name,
                trigger_pattern,
                json.dumps(steps),
                answer_template,
                learned_from,
                initial_success_rate,
            ),
        )
        skill_id = cursor.lastrowid
        logger.info("Created skill #%d: '%s' (trigger: %s)", skill_id, name, trigger_pattern)
        return skill_id

    def record_use(self, skill_id: int, success: bool) -> None:
        """Record a skill execution. Updates times_used and success_rate.

        Auto-disables skills with success_rate < 0.3 after 5+ uses.
        Uses atomic UPDATE to avoid read-then-write race.
        """
        success_val = 1.0 if success else 0.0
        alpha = 0.15
        self._db.execute(
            "UPDATE skills SET times_used = times_used + 1, "
            "success_rate = ? * ? + (1 - ?) * success_rate "
            "WHERE id = ?",
            (alpha, success_val, alpha, skill_id),
        )

        # Check for auto-disable (separate read after atomic update)
        row = self._db.fetchone(
            "SELECT name, times_used, success_rate FROM skills WHERE id = ?",
            (skill_id,),
        )
        if row and row["times_used"] >= 5 and row["success_rate"] < 0.3:
            self._db.execute(
                "UPDATE skills SET enabled = 0 WHERE id = ?", (skill_id,)
            )
            logger.warning(
                "Auto-disabled skill #%d '%s': success_rate=%.2f after %d uses",
                skill_id, row["name"], row["success_rate"], row["times_used"],
            )

    def get_skill(self, skill_id: int) -> Skill | None:
        """Get a skill by ID."""
        row = self._db.fetchone("SELECT * FROM skills WHERE id = ?", (skill_id,))
        return self._row_to_skill(row) if row else None

    def get_all_skills(self, limit: int = 50) -> list[Skill]:
        """Get all skills."""
        rows = self._db.fetchall(
            "SELECT * FROM skills ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT ?",
            (limit,),
        )
        return [self._row_to_skill(r) for r in rows]

    def get_active_skills(self) -> list[Skill]:
        """Get all enabled skills for prompt injection."""
        rows = self._db.fetchall(
            "SELECT * FROM skills WHERE enabled = 1 ORDER BY times_used DESC, success_rate DESC, id ASC LIMIT 50"
        )
        return [self._row_to_skill(r) for r in rows]

    def toggle_skill(self, skill_id: int, enabled: bool) -> bool:
        """Enable or disable a skill. Re-enabling resets stats for a fresh start."""
        if enabled:
            cursor = self._db.execute(
                "UPDATE skills SET enabled = 1, times_used = 0, success_rate = 0.7 WHERE id = ?",
                (skill_id,),
            )
        else:
            cursor = self._db.execute(
                "UPDATE skills SET enabled = 0 WHERE id = ?",
                (skill_id,),
            )
        return cursor.rowcount > 0

    def delete_skill(self, skill_id: int) -> bool:
        """Delete a skill."""
        cursor = self._db.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        return cursor.rowcount > 0

    async def refine_skill(self, skill_id: int, failure_context: str) -> bool:
        """Attempt to refine a failing skill instead of just degrading it.

        LLM analyzes the failure and suggests: narrow the trigger regex,
        adjust the steps, or skip (not refinable). Returns True if refined.
        """
        skill = self.get_skill(skill_id)
        if not skill or not skill.enabled:
            return False

        import json as _json
        prompt = (
            f"A learned skill is failing. Analyze and suggest a fix.\n\n"
            f"Skill name: {skill.name}\n"
            f"Trigger pattern: {skill.trigger_pattern}\n"
            f"Steps: {_json.dumps(skill.steps)}\n"
            f"Failure context: {failure_context[:500]}\n\n"
            "Options:\n"
            "1. Narrow the trigger regex to be more specific\n"
            "2. Adjust the tool steps\n"
            "3. Skip — not refinable\n\n"
            "Return JSON: {\"action\": \"narrow\"|\"adjust\"|\"skip\", "
            "\"new_trigger\": \"...\", \"new_steps\": [...], \"reason\": \"...\"}\n"
            "For 'skip', only include action and reason."
        )

        try:
            raw = await llm.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=400,
                temperature=0.2,
            )
            obj = llm.extract_json_object(raw)
            if not obj or obj.get("action") == "skip":
                return False

            action = obj.get("action", "skip")

            if action == "narrow" and obj.get("new_trigger"):
                new_trigger = obj["new_trigger"]
                # Validate: must be valid regex, not ReDoS-prone, and not too broad
                try:
                    re.compile(new_trigger)
                except re.error:
                    return False
                if _is_redos_risk(new_trigger):
                    logger.warning("Skill #%d refinement rejected: ReDoS risk (%s)", skill_id, new_trigger)
                    return False
                if _is_too_broad(new_trigger):
                    return False
                self._db.execute(
                    "UPDATE skills SET trigger_pattern = ? WHERE id = ?",
                    (new_trigger, skill_id),
                )
                logger.info("Skill #%d refined: narrowed trigger to '%s'", skill_id, new_trigger)
                return True

            elif action == "adjust" and obj.get("new_steps"):
                new_steps = obj["new_steps"]
                if not isinstance(new_steps, list):
                    return False
                # Validate tool names
                valid_tools = _get_tool_names()
                for step in new_steps:
                    if not isinstance(step, dict) or step.get("tool") not in valid_tools:
                        return False
                self._db.execute(
                    "UPDATE skills SET steps = ? WHERE id = ?",
                    (_json.dumps(new_steps), skill_id),
                )
                logger.info("Skill #%d refined: adjusted steps", skill_id)
                return True

        except Exception as e:
            logger.debug("Skill refinement failed: %s", e)

        return False

    def _row_to_skill(self, row) -> Skill:
        """Convert a DB row to a Skill dataclass."""
        steps = json.loads(row["steps"]) if isinstance(row["steps"], str) else row["steps"]
        return Skill(
            id=row["id"],
            name=row["name"],
            trigger_pattern=row["trigger_pattern"],
            steps=steps,
            answer_template=row["answer_template"],
            learned_from=row["learned_from"],
            times_used=row["times_used"],
            success_rate=row["success_rate"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# Guards — ported from old nova's hard-won lessons
# ---------------------------------------------------------------------------

def _is_redos_risk(pattern: str) -> bool:
    """Heuristic check for regex patterns likely to cause catastrophic backtracking.

    Detects nested quantifiers like (a+)+, (a*)+, (a+)*, overlapping
    alternations in quantified groups, and similar ReDoS-prone constructs.
    """
    # Nested quantifiers: (X+)+, (X*)+, (X+)*, (X*)*
    if re.search(r'\([^)]*[+*][^)]*\)[+*{]', pattern):
        return True
    # Overlapping quantifiers without anchors: \w+\w+, .+.+
    if re.search(r'(?:\\w|\.)[+*].*(?:\\w|\.)[+*].*[+*$]', pattern):
        return True
    return False


_BROADNESS_TEST_QUERIES = [
    "What's the weather like today?",
    "Tell me a joke",
    "How do I cook pasta?",
    "What is quantum computing?",
    "Recommend a good book",
    "How tall is Mount Everest?",
    "Translate hello to Spanish",
    "What time is it in Tokyo?",
    "How much does a Tesla cost?",
    "What's the price of gold?",
    "Compare Python and JavaScript",
    "How do I fix a flat tire?",
    "Who won the World Cup?",
    "What should I eat for dinner?",
    "How much is a flight to Paris?",
    # Non-English queries to prevent non-English patterns from always passing
    "¿Cuál es el clima hoy?",          # Spanish
    "今天天气怎么样？",                    # Chinese
    "ما هو الطقس اليوم؟",              # Arabic
    "आज मौसम कैसा है?",                # Hindi
    "今日の天気は何ですか？",               # Japanese
]


def _is_too_broad(pattern: str) -> bool:
    """Test a trigger regex against 20 unrelated queries. Reject if ≥2 match."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return True
    matches = sum(1 for q in _BROADNESS_TEST_QUERIES if regex.search(q))
    return matches >= 2


def _has_capture_group_mismatch(pattern: str, steps: list[dict], answer_template: str | None) -> bool:
    """Check if templates reference capture groups that don't exist in the regex."""
    try:
        num_groups = re.compile(pattern).groups
    except re.error:
        return True

    # Collect all template strings to check
    templates = []
    if answer_template:
        templates.append(answer_template)
    for step in steps:
        args = step.get("args_template", {})
        if isinstance(args, dict):
            templates.extend(str(v) for v in args.values())
        elif isinstance(args, str):
            templates.append(args)

    # Look for $N or {capture_N} references
    for tmpl in templates:
        for match in re.finditer(r"\$(\d+)", tmpl):
            if int(match.group(1)) > num_groups:
                return True
        for match in re.finditer(r"\{capture_(\d+)\}", tmpl):
            if int(match.group(1)) > num_groups:
                return True

    return False


# Fallback tool names for validation when ToolRegistry is unavailable
_FALLBACK_TOOL_NAMES = frozenset({
    "web_search", "calculator", "http_fetch", "knowledge_search",
    "code_exec", "memory_search", "file_ops", "shell_exec", "browser",
    "integration", "screenshot", "monitor", "email_send", "calendar",
    "reminder", "webhook", "delegate",
})


def _get_tool_names() -> set[str]:
    """Get valid tool names from ToolRegistry (dynamic), falling back to hardcoded set."""
    try:
        from app.tools.base import ToolRegistry
        from app.core.brain import get_services
        svc = get_services()
        if svc.tool_registry:
            names = set(svc.tool_registry.tool_names)
            if names:
                return names
    except Exception:
        pass
    return _FALLBACK_TOOL_NAMES


def _mentions_tool_procedure(text: str) -> bool:
    """Check if a correction message describes a tool-based procedure."""
    lower = text.lower()
    # Check for explicit tool names
    if any(tool in lower for tool in _get_tool_names()):
        return True
    # Check for procedural language about tools/searches/actions
    procedural = re.search(
        r"(?i)\b(?:search|look\s+up|fetch|calculate|check|use|try)\b.*\b(?:first|then|instead|always|next)\b",
        text,
    )
    return bool(procedural)


async def extract_skill_from_correction(
    correction_context: str,
    tool_history: list[dict],
    lesson_id: int | None = None,
) -> dict | None:
    """Try to extract a reusable skill from a correction.

    Creates a skill if:
    - The correction involved tool use, OR
    - The correction describes a tool-based procedure
    Returns skill dict or None.
    """
    # Skip if there's no tool usage AND the correction doesn't describe a procedure
    if not tool_history and not _mentions_tool_procedure(correction_context):
        return None

    tool_info = ""
    if tool_history:
        tool_info = f"\n\nTool calls that happened: {json.dumps(tool_history)}"

    try:
        result = await llm.invoke_nothink(
            [
                {
                    "role": "system",
                    "content": (
                        "You extract reusable skills from corrections. A skill is a "
                        "trigger pattern (regex) and a sequence of tool calls.\n\n"
                        "Given a correction, create a skill if the user describes a "
                        "reusable procedure involving tool calls.\n\n"
                        "Respond with JSON:\n"
                        '{"name": "short_name", "trigger_pattern": "regex_to_match_queries", '
                        '"steps": [{"tool": "tool_name", "args_template": {"key": "{query}"}}], '
                        '"answer_template": "Use the result to answer: {result}"}\n\n'
                        "IMPORTANT:\n"
                        "- trigger_pattern must be a valid regex that matches similar future queries\n"
                        "- steps must reference actual tools: web_search, calculator, http_fetch, "
                        "knowledge_search, code_exec, memory_search, file_ops\n"
                        "- Use {query} as placeholder for the user's query in args_template\n\n"
                        'If this is NOT a reusable tool procedure, respond: {"skip": true}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Correction: {correction_context}{tool_info}",
                },
            ],
            json_mode=True,
            json_prefix="{",
        )

        obj = llm.extract_json_object(result)
        if obj.get("skip") or not obj.get("name"):
            return None

        # Validate trigger pattern is a valid regex
        pattern = obj.get("trigger_pattern", "")
        if pattern:
            try:
                re.compile(pattern)
            except re.error:
                logger.warning("Skill extraction produced invalid regex: %s", pattern)
                return None

        # Validate steps reference real tools
        steps = obj.get("steps", [])
        valid_tools = _get_tool_names()
        if steps:
            for step in steps:
                if step.get("tool") not in valid_tools:
                    return None

        answer_template = obj.get("answer_template")

        # Guard: capture group references must match actual groups in regex
        if _has_capture_group_mismatch(pattern, steps, answer_template):
            logger.warning("Skill extraction has capture group mismatch: %s", pattern)
            return None

        return {
            "name": obj["name"],
            "trigger_pattern": pattern,
            "steps": steps,
            "answer_template": answer_template,
            "learned_from": lesson_id,
        }

    except Exception as e:
        logger.warning("Skill extraction failed: %s", e)
        return None
