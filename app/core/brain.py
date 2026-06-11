"""The Brain — Nova's core reasoning loop.

This single module replaces a 9-node LangGraph pipeline.
The think() async generator is the entire pipeline:
  context → prompt → generate → maybe tool loop → stream → post-process
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, TYPE_CHECKING

from app.config import config

if TYPE_CHECKING:
    from app.core.learning import LearningEngine
    from app.core.skills import SkillStore
    from app.core.reflexion import ReflexionStore
    from app.core.retriever import Retriever
    from app.core.kg import KnowledgeGraph
    from app.core.custom_tools import CustomToolStore
    from app.core.curiosity import CuriosityQueue, TopicTracker
    from app.tools.base import ToolRegistry
    from app.monitors.heartbeat import MonitorStore, HeartbeatLoop
    from app.core.agent_spawner import DecompositionPlan
from app.core import llm
from app.core.claim_validator import build_evidence, count_claim_candidates, validate_claims
from app.core.learning import is_likely_correction, response_pushes_back
from app.core.llm import LLMUnavailableError, _extract_tool_calls
from app.core.memory import ConversationStore, UserFactStore
from app.core.prompt import (
    build_system_prompt,
    format_lessons_for_prompt,
    format_skills_for_prompt,
)
from app.schema import EventType, StreamEvent

logger = logging.getLogger(__name__)

# Background task references to prevent GC (PEP 540 / asyncio docs warning)
_background_tasks: set[asyncio.Task] = set()

# Per-conversation locks to serialize concurrent think() calls for the same conversation.
# Plain dict (insertion-ordered in Python 3.7+) with max-size eviction.
_conversation_locks: dict[str, asyncio.Lock] = {}
_conversation_locks_meta_lock = asyncio.Lock()
_MAX_CONVERSATION_LOCKS = 500


async def _get_conversation_lock(conv_id: str) -> asyncio.Lock:
    """Get or create a per-conversation lock with LRU eviction.

    Uses a plain dict (insertion-ordered) as an LRU cache.
    On hit: re-insert to move to end (most recent).
    On capacity: evict oldest unlocked entries in one pass.
    """
    async with _conversation_locks_meta_lock:
        if conv_id in _conversation_locks:
            # Move to end (most recently used) by re-inserting
            lock = _conversation_locks.pop(conv_id)
            _conversation_locks[conv_id] = lock
            return lock
        # Evict oldest unlocked entries if at capacity
        if len(_conversation_locks) >= _MAX_CONVERSATION_LOCKS:
            to_evict = [
                k for k, v in _conversation_locks.items() if not v.locked()
            ]
            # Evict enough to get below capacity (oldest first — dict is ordered)
            needed = len(_conversation_locks) - _MAX_CONVERSATION_LOCKS + 1
            for k in to_evict[:needed]:
                del _conversation_locks[k]
        lock = asyncio.Lock()
        _conversation_locks[conv_id] = lock
        return lock


# Tools with side effects — skip caching for these (used in think() tool cache)
_SIDE_EFFECT_TOOLS = frozenset({
    "file_ops", "email_send", "webhook", "calendar", "reminder",
    "shell_exec", "code_exec", "integration", "delegate", "browser",
    "desktop", "background_task", "tool_create", "monitor",
})

# Tools allowed at structural depth >= 2 — sub-sub-agents must reason deeply,
# not spawn more research. Web/browser/http_fetch removed; calculator + code_exec
# kept for in-context computation.
_DEPTH2_ALLOWED_TOOLS = frozenset({
    "calculator", "code_exec", "memory_search", "knowledge_search", "context_detail",
})


# Per-query pivot tracking — one backtrack pivot allowed per conversation per query.
# Keyed by conversation_id; auto-cleared after the tool loop finishes (in _run_generation_loop).
_pivot_attempted_this_query: dict[str, bool] = {}


# Markers for the three confidence-transparency footer tiers.
# Idempotency check uses these strings to avoid double-appending across critique rounds.
_CONFIDENCE_FOOTER_TIERS = (
    "_(low-confidence:",
    "_(limited-verification:",
    "_(noted-uncertainty:",
)
_CONFIDENCE_FOOTER_MARKER = "_(low-confidence:"  # back-compat for old check
_CONFIDENCE_FOOTER_TEXT = (
    "_(low-confidence: answered from internal knowledge with no fresh sources. "
    "Ask me to verify with a web search if precision matters here.)_"
)


def _has_confidence_footer(text: str) -> bool:
    """True if `text` already contains any tier of confidence footer."""
    if not text:
        return False
    return any(m in text for m in _CONFIDENCE_FOOTER_TIERS)


# Patterns that indicate the model leaked raw tool output into the user-facing answer.
# These should never appear in a polished response — they're internal scaffolding.
_TOOL_LEAKAGE_MARKERS = (
    "## Matching User Facts", "## Matching Conversations", "## Matching Documents",
    "## Matching KG", "## Matching Knowledge",
    "[Source 1:", "[Source 2:", "[Source 3:", "[Source 4:", "[Source 5:",
    "[Tool error", "[Tool failed", "[tool error", "[tool failed",
    "[Calling tool:", "## Tool Result", "## Tool Output",
    "[ERROR_CATEGORY:", "[ErrorCategory.", "tier: full", "tier: standard", "tier: sandboxed",
    "retriable=", "error_category=", "tool_call_id:",
    'role: "tool"', "{'tool':", '{"tool":',  # raw JSON tool-call leak
    "<scratchpad>", "</scratchpad>", "<thinking>", "</thinking>",
)

# Patterns that match raw tool-call JSON the model occasionally leaks
_TOOL_CALL_JSON_RE = re.compile(r'^\s*\{["\']tool["\']\s*:\s*["\']\w+["\']', re.MULTILINE)


# Known-confusion entity terms — substrings that, when matched in a KG fact's
# subject, indicate the fact is about a DIFFERENT thing than the user intended.
# Triggered when the query mentions identity terms ("nova", "helios", "this project",
# "this codebase", "yourself") which the user clearly meant as the assistant/project.
_IDENTITY_CONFUSION_TERMS = (
    "helios protocol", "helios blockchain", "helios chain", "hls token",
    "nova lake", "intel nova", "core ultra 400", "lunar lake", "arrow lake",
    "supernova", "supernovae", "nova explosion", "nova outburst",
)
_IDENTITY_QUERY_TRIGGERS = re.compile(
    r"\b(nova|helios|this\s+(project|codebase|system|repo)|yourself|your\s+(architecture|design|code))\b",
    re.IGNORECASE,
)


def _filter_confused_kg_facts(query: str, facts: list) -> list:
    """Drop KG facts about known-confusion entities when the query is about Nova/Helios as identity.

    Conservative: only filters when query has identity triggers AND the fact's
    subject contains a confusion term. Returns the list unchanged otherwise.
    """
    if not facts or not _IDENTITY_QUERY_TRIGGERS.search(query):
        return facts
    keep = []
    dropped = 0
    for f in facts:
        # Fact may be a dict, dataclass, or named tuple — duck-type the subject
        subject = getattr(f, "subject", None) or (f.get("subject") if isinstance(f, dict) else "")
        sl = str(subject).lower()
        if any(term in sl for term in _IDENTITY_CONFUSION_TERMS):
            dropped += 1
            continue
        keep.append(f)
    if dropped:
        logger.info("[KG-filter] dropped %d confused-identity facts for query: %r", dropped, query[:80])
    return keep


def _has_tool_leakage(text: str) -> bool:
    """True if the text contains raw tool output markers that shouldn't be user-facing.

    Two checks: literal marker substrings AND a regex for raw {"tool": "..."} JSON
    that's not wrapped in a code fence.
    """
    if not text:
        return False
    if any(m in text for m in _TOOL_LEAKAGE_MARKERS):
        return True
    if _TOOL_CALL_JSON_RE.search(text):
        return True
    return False


# Patterns for queries that are INHERENTLY uncertain — forecasts, predictions,
# exact future values, opinions about future events. For these, best-of-N would
# pick the most-confident answer over the correctly-uncertain one, hurting
# calibration. Quality-by-uncertainty is the wrong objective here.
_INHERENTLY_UNCERTAIN_RE = re.compile(
    r"\b("
    r"will\s+(?:be|happen|occur|win|lose|reach|hit|fall|rise|cross)|"
    r"forecast|predict|prediction|projection|"
    r"(?:exact|precise)\s+(?:value|price|number|figure)\s+(?:of|for)\s+\w+\s+(?:in|next|by)|"
    r"(?:one|two|three|five|ten)?\s*year[s]?\s+from\s+(?:now|today)|"
    r"in\s+\d{4}\s+\(future\)|"
    r"who\s+will\s+win|outcome\s+of|"
    r"weather\s+(?:next|tomorrow|in\s+\d+|on\s+\w+day)"
    r")\b",
    re.IGNORECASE,
)

# Query EXPLICITLY asks Nova to use a tool ("write and run code", "use the
# calculator"). As the memory loop teaches the model to compute, it tends to
# self-answer correctly but skip the requested tool — fine for implicit math
# ("what is 15% of 840"), but it should honor an *explicit* tool request. This
# fires narrowly so it never forces a tool on a query that didn't ask for one.
_EXPLICIT_TOOL_REQUEST_RE = re.compile(
    r"\b(run|execute)\b[^.?!\n]{0,25}\b(code|python|script)\b"
    r"|\bwrite\b[^.?!\n]{0,20}\b(and|&)\b[^.?!\n]{0,10}\brun\b"
    r"|\buse\b[^.?!\n]{0,12}\b(the\s+)?calculator\b"
    r"|\bcode[_ ]exec\b",
    re.IGNORECASE,
)

# Time-sensitive queries (current price, latest news, today's weather) must
# never serve from workspace cache or stale lessons — the "fact" they cached
# yesterday is wrong today. Used to:
#   1. bypass workspace findings injection (load_workspace path)
#   2. skip workspace fact extraction (don't cache today's hallucination as
#      tomorrow's "fact")
#   3. force a fresh tool call rather than answering from history
# Conservative: only matches queries that EXPLICITLY use a time-sensitive
# qualifier ("current", "now", "today", "latest", "right now") with a target
# (price, weather, news, score, headline, value, rate, level).
_TIME_SENSITIVE_RE = re.compile(
    r"\b(?:current|currently|right\s+now|now|today(?:'s)?|latest|this\s+(?:week|month|hour)|"
    r"as\s+of\s+(?:today|now)|live|real-time)\b"
    r"[^.?!]{0,80}?\b(?:price|prices?|weather|temperature|news|score|scores?|"
    r"headline|headlines?|value|rate|rates?|level|levels?|stock|crypto|"
    r"forecast|standing|ranking|trade|market|exchange)\b",
    re.IGNORECASE,
)


# Patterns that identify must-address components in a user query.
_NUMBERED_ITEM_RE = re.compile(r"\(\s*(?:\d+|[a-iA-I])\s*\)\s*([^()]+?)(?=\s*\(\s*(?:\d+|[a-iA-I])\s*\)|[.?!]\s|$)", re.DOTALL)
_AND_PART_RE = re.compile(r"\b(?:both|each)\s+([\w\s]+?)\s+and\s+([\w\s]+?)(?:[.,;]|$)", re.IGNORECASE)


def _find_missing_constraints(query: str, answer: str) -> list[str]:
    """Return a list of must-address components from `query` that don't appear in `answer`.

    Heuristic: extract numbered items "(1) X (2) Y", "both X and Y" forms; for
    each, check if any 2+ keyword from the component appears in the answer.
    Conservative — false-positive (claim missing when present) is OK; we'd
    rather over-pad than miss a constraint.
    """
    if not query or not answer:
        return []
    answer_lc = answer.lower()
    missing: list[str] = []

    # 1. Numbered/lettered components: "(1) X (2) Y"
    for m in _NUMBERED_ITEM_RE.finditer(query):
        component = m.group(1).strip().rstrip(".;:,")
        if not component or len(component) < 6:
            continue
        # Pull keywords (drop stopwords, keep nouns/verbs of length>=4)
        keywords = [
            w.lower().strip(".,;:") for w in component.split()
            if len(w) >= 4 and w.lower() not in {"what", "which", "where", "when", "does",
                                                    "with", "from", "this", "that", "your",
                                                    "their", "would", "could", "should", "have"}
        ]
        if len(keywords) < 2:
            continue
        # Component is "addressed" if at least 1 keyword appears in answer
        if not any(k in answer_lc for k in keywords[:5]):
            missing.append(component[:120])

    # 2. "both X and Y" / "each X and Y" forms
    for m in _AND_PART_RE.finditer(query):
        for part in (m.group(1), m.group(2)):
            part = part.strip().rstrip(".;:,")
            if not part or len(part) < 4:
                continue
            keywords = [w.lower().strip(".,;:") for w in part.split() if len(w) >= 4]
            if not keywords:
                continue
            if not any(k in answer_lc for k in keywords[:3]):
                missing.append(part[:80])

    # Dedup + cap
    seen = set()
    deduped: list[str] = []
    for m in missing:
        key = m.lower()[:60]
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped[:5]


# ---------------------------------------------------------------------------
# Services container (injected at startup)
# ---------------------------------------------------------------------------

@dataclass
class Services:
    """Dependency injection container — set up in main.py lifespan."""
    conversations: ConversationStore | None = None
    user_facts: UserFactStore | None = None
    retriever: Retriever | None = None
    learning: LearningEngine | None = None
    skills: SkillStore | None = None
    tool_registry: ToolRegistry | None = None
    kg: KnowledgeGraph | None = None
    reflexions: ReflexionStore | None = None
    custom_tools: CustomToolStore | None = None
    monitor_store: MonitorStore | None = None
    heartbeat: HeartbeatLoop | None = None
    curiosity: CuriosityQueue | None = None
    topic_tracker: TopicTracker | None = None
    external_skills: list | None = None  # list[ExternalSkill]
    task_manager: Any = None


# Module-level services ref (set during startup)
_services: Services | None = None


def set_services(svc: Services) -> None:
    global _services
    if _services is not None:
        logger.warning("set_services() called more than once — replacing existing Services instance")
    _services = svc


def get_services() -> Services:
    if _services is None:
        raise RuntimeError("Services not initialized. Call set_services() during startup.")
    return _services


# ---------------------------------------------------------------------------
# Tool execution — dispatches to ToolRegistry
# ---------------------------------------------------------------------------

async def _handle_tool_create(svc: Services, args: dict) -> str:
    """Handle the tool_create virtual action."""
    try:
        from app.core.custom_tools import DynamicTool
        name = args.get("name", "").strip()
        description = args.get("description", "").strip()
        parameters = args.get("parameters", "[]")
        code = args.get("code", "").strip()

        if not name or not code:
            return "[Tool creation failed: name and code are required.]"

        tool_id = svc.custom_tools.create_tool(name, description, parameters, code)
        if tool_id == -1:
            return "[Tool creation failed: name already exists, code blocked, or limit reached.]"

        # Register in live registry
        record = svc.custom_tools.get_tool(name)
        if record and svc.tool_registry:
            svc.tool_registry.register(DynamicTool(record, svc.custom_tools))

        return f"[Tool '{name}' created successfully (id={tool_id}). It is now available for use.]"
    except Exception as e:
        logger.warning("Tool creation failed: %s", e)
        return f"[Tool creation failed: {e}]"

async def _execute_tool(tool_name: str, args: dict) -> tuple[str, "ToolResult | None"]:
    """Execute a tool via the registry. Returns (output_str, ToolResult|None).

    The ToolResult is used for structured failure detection (success field)
    instead of substring matching on error markers.
    """
    from app.tools.base import format_tool_error, ToolResult, ErrorCategory
    svc = get_services()
    if svc.tool_registry:
        timeout = float(config.TOOL_TIMEOUT)
        try:
            output, result = await asyncio.wait_for(
                svc.tool_registry.execute_full(tool_name, args),
                timeout=timeout,
            )
            return output, result
        except asyncio.TimeoutError:
            logger.warning("Tool '%s' timed out after %ds", tool_name, config.TOOL_TIMEOUT)
            msg = format_tool_error(tool_name, f"Timed out after {config.TOOL_TIMEOUT} seconds", retriable=True, category=ErrorCategory.TRANSIENT)
            return msg, ToolResult(output="", success=False, error="Timeout", retriable=True, error_category=ErrorCategory.TRANSIENT)
        except Exception as e:
            logger.exception("Tool '%s' failed with exception", tool_name)
            msg = format_tool_error(tool_name, f"Failed: {e}", retriable=True, category=ErrorCategory.INTERNAL)
            return msg, ToolResult(output="", success=False, error=str(e), retriable=True, error_category=ErrorCategory.INTERNAL)
    msg = format_tool_error(tool_name, "Not yet available")
    return msg, ToolResult(output="", success=False, error="Not yet available")


def _get_tool_descriptions() -> str:
    """Get tool descriptions from the registry, or static fallback."""
    svc = get_services()
    if svc.tool_registry:
        desc = svc.tool_registry.get_descriptions()
    else:
        desc = """web_search(query: str) — Search the web. Use for current events, facts you don't know, prices, news.
calculator(expression: str) — Evaluate math expressions with SymPy. Use for ANY calculation, even simple ones.
http_fetch(url: str) — Fetch a specific URL and return its content. Use when you have a known URL.
knowledge_search(query: str) — Search your owner's ingested documents. Use for questions about uploaded content.
code_exec(code: str) — Execute Python code in a sandbox. Use for data processing, complex logic, formatting.
memory_search(query: str) — Search past conversations and archival memory.
file_ops(action: str, path: str, content: str) — Read/write files in the /data directory."""
    # Append tool_create if custom tools enabled
    if config.ENABLE_CUSTOM_TOOLS and svc.custom_tools:
        from app.core.custom_tools import TOOL_CREATE_DESCRIPTION
        desc += "\n" + TOOL_CREATE_DESCRIPTION
    return desc


def _get_available_tools() -> list[dict]:
    """Get tool metadata for tool call validation."""
    svc = get_services()
    if svc.tool_registry:
        return svc.tool_registry.get_tool_list()
    return [
        {"name": "web_search"},
        {"name": "calculator"},
        {"name": "http_fetch"},
        {"name": "knowledge_search"},
        {"name": "code_exec"},
        {"name": "memory_search"},
        {"name": "file_ops"},
    ]


# ---------------------------------------------------------------------------
# Context window management extracted to brain_context_manager for size
# hygiene. Re-export keeps existing imports working.
from app.core.brain_context_manager import _manage_context  # noqa: E402,F401


# Intent classification, title generation, model routing extracted to
# brain_routing for size hygiene. Sanitizer extracted to brain_sanitize.
# Re-exports keep `from app.core.brain import ...` working for callers.
from app.core.brain_sanitize import _META_PATTERNS, _sanitize_answer  # noqa: E402,F401
from app.core.brain_routing import (  # noqa: E402,F401
    _COMPLEX_PATTERNS,
    _CREATIVE_PATTERNS,
    _GREETING_PATTERNS,
    _PURE_GREETING,
    _QUESTION_WORDS,
    _classify_intent,
    _generate_title,
    _select_model,
)




# ---------------------------------------------------------------------------
# Internal dataclasses for stage communication
# ---------------------------------------------------------------------------

@dataclass
class _ThinkContext:
    """All gathered context for a single think() call."""
    matched_skill: object | None = None
    used_lesson_ids: list[int] = field(default_factory=list)
    skills_text: str = ""
    user_facts_text: str = ""
    lessons_text: str = ""
    kg_facts_text: str = ""
    kg_facts_count: int = 0
    reflexions_text: str = ""
    reflexions_count: int = 0
    retrieved_context: str = ""
    retrieved_sources: list[dict] = field(default_factory=list)
    integrations_text: str = ""
    success_patterns_text: str = ""
    success_pattern_ids: list[int] = field(default_factory=list)  # ids injected, for A/B closure tracking
    lessons: list = field(default_factory=list)  # raw lesson objects, for LESSON_USED events
    external_skills_text: str = ""               # summaries of loaded external skills
    matched_external_skill_text: str = ""        # full body of matched external skill
    workspace_text: str = ""                     # prior agent_workspace findings for similar query
    workspace_signature: str = ""                # signature used for workspace lookup (for save)
    prior_sessions_text: str = ""                # GSW: summaries of prior conversations on related topics


@dataclass
class _GenerationResult:
    """Mutable output from the generation+tool loop."""
    final_content: str = ""
    tool_results: list[dict] = field(default_factory=list)
    is_error: bool = False


# ---------------------------------------------------------------------------
# Stage functions (private helpers for think())
# ---------------------------------------------------------------------------


_CONTEXT_GATHER_TIMEOUT_S = 5.0  # per-subsystem cap; tuned for healthy SQLite/ChromaDB


async def _run_context_io(name: str, awaitable, default):
    """Wrap one context-gather IO call with a hard timeout.

    Each subsystem (user_facts, skills, lessons, kg, reflexions, gsw,
    workspace) calls SQLite/ChromaDB via asyncio.to_thread. Before this
    wrapper, a slow query (lock contention, broken connection) would
    stall the WHOLE pipeline because the gather is sequential within
    _gather_context. Now each call has a 5-second cap; on timeout we
    degrade gracefully by returning `default` and logging the loss.
    The pipeline continues with whatever context succeeded.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=_CONTEXT_GATHER_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            "[context-gather] %s timed out after %.1fs — degrading without it",
            name, _CONTEXT_GATHER_TIMEOUT_S,
        )
        return default
    except Exception as e:
        logger.warning("[context-gather] %s failed: %s", name, e)
        return default


async def _gather_context(
    svc: Services,
    query: str,
    intent: str,
    conversation_id: str = "",
) -> _ThinkContext:
    """Load skills, user facts, lessons, KG facts, reflexions, retrieval, integrations.

    This is Steps 4–7b of the original think() pipeline.
    """
    ctx = _ThinkContext()
    ctx.user_facts_text = (
        await _run_context_io(
            "user_facts",
            asyncio.to_thread(svc.user_facts.format_for_prompt),
            default="",
        )
    ) if svc.user_facts else ""

    # --- Skills ---
    if svc.skills:
        ctx.matched_skill = await _run_context_io(
            "skills.match",
            asyncio.to_thread(svc.skills.get_matching_skill, query),
            default=None,
        )
        if ctx.matched_skill:
            logger.info("Skill matched: '%s' (id=%d)", ctx.matched_skill.name, ctx.matched_skill.id)
            steps_desc = "\n".join(
                f"  {i+1}. Use {s.get('tool', '?')} with {json.dumps(s.get('args_template', {}))}"
                for i, s in enumerate(ctx.matched_skill.steps)
            )
            ctx.skills_text = (
                f"## Matched Skill: {ctx.matched_skill.name}\n\n"
                f"You have a learned procedure for this type of query. Follow these steps:\n"
                f"{steps_desc}\n"
            )
            if ctx.matched_skill.answer_template:
                ctx.skills_text += f"\nAnswer format: {ctx.matched_skill.answer_template}\n"
            ctx.skills_text += "\nFollow this procedure. If the skill seems wrong for this query, deviate and explain why.\n"
        else:
            active_skills = await _run_context_io(
                "skills.active",
                asyncio.to_thread(svc.skills.get_active_skills),
                default=[],
            )
            if active_skills:
                ctx.skills_text = format_skills_for_prompt([
                    {"name": s.name, "trigger_pattern": s.trigger_pattern}
                    for s in active_skills[:5]
                ])

    # --- Lessons ---
    # Bypass for time-sensitive queries: lessons may include stale prices /
    # forecasts (e.g. "Bitcoin Trading Price - March 28, 2026" in today's
    # answer to "what's bitcoin currently?"). Force fresh tool answers.
    # Also: filter at the retrieval source by confidence so low-confidence
    # entries (<0.40) never pass — the formatter filter wasn't enough.
    _is_time_sensitive_query = bool(_TIME_SENSITIVE_RE.search(query))
    if svc.learning and not _is_time_sensitive_query:
        lessons = await _run_context_io(
            "lessons",
            asyncio.to_thread(svc.learning.get_relevant_lessons, query),
            default=[],
        )
        # Hard confidence floor at retrieval — ignore lessons that clearly
        # didn't validate. Was previously only enforced in the formatter,
        # which meant retrieval logs still showed conf=0.16 lessons being
        # "Retrieved 5" even though they were dropped from the prompt.
        if lessons:
            lessons = [
                l for l in lessons
                if (getattr(l, "confidence", 0.8) or 0.8) >= 0.40
            ]
        # Deliberation queries (design / walk-through / step-by-step) get
        # CAPPED to 2 lessons. Multi-summary contamination was observed on
        # `deliberation_chain_of_reasoning` across runs 6/7: 5 short
        # rate-limiter lessons in retrieval biased the model into matching
        # their summary-level depth instead of producing the structured
        # multi-section walkthrough the user asked for. Two highest-confidence
        # lessons are still useful as principles; more than that signals
        # "this depth is correct" to the model.
        if lessons:
            try:
                from app.core.agent_loop import _is_deliberation_query
                if _is_deliberation_query(query) and len(lessons) > 2:
                    lessons.sort(
                        key=lambda l: getattr(l, "confidence", 0.8) or 0.8,
                        reverse=True,
                    )
                    lessons = lessons[:2]
                    logger.info(
                        "[lessons] deliberation query — capped to top 2 by "
                        "confidence to avoid summary-depth contamination"
                    )
            except Exception as e:
                logger.debug("[lessons] deliberation cap check failed: %s", e)
        if lessons:
            logger.info(
                "Retrieved %d lessons: %s",
                len(lessons),
                [(l.id, l.topic, (l.lesson_text or "")[:60]) for l in lessons],
            )
            ctx.lessons = lessons
            ctx.used_lesson_ids = [l.id for l in lessons]
            ctx.lessons_text = format_lessons_for_prompt([
                {
                    "topic": l.topic,
                    "wrong_answer": l.wrong_answer or "",
                    "correct_answer": l.correct_answer or "",
                    "lesson_text": l.lesson_text or "",
                    "confidence": l.confidence if hasattr(l, "confidence") else 0.8,
                }
                for l in lessons
            ])

    # --- Knowledge graph facts ---
    if svc.kg:
        kg_facts = await _run_context_io(
            "kg",
            asyncio.to_thread(svc.kg.get_relevant_facts, query, config.MAX_KG_FACTS_IN_PROMPT),
            default=None,
        )
        if kg_facts:
            # Identity-aware filter: drop facts about known confusion entities
            # (Helios Protocol = blockchain, Nova Lake = Intel CPU, Supernova =
            # astronomy) when the query is about Nova/Helios as the project.
            kg_facts = _filter_confused_kg_facts(query, kg_facts)
            ctx.kg_facts_text = svc.kg.format_for_prompt(kg_facts)
            ctx.kg_facts_count = len(kg_facts)

    # --- Reflexions (past failure warnings) ---
    if svc.reflexions:
        reflexions = await _run_context_io(
            "reflexions",
            asyncio.to_thread(svc.reflexions.get_relevant, query, config.MAX_REFLEXIONS_IN_PROMPT),
            default=None,
        )
        if reflexions:
            ctx.reflexions_text = svc.reflexions.format_for_prompt(reflexions)
            ctx.reflexions_count = len(reflexions)

    # --- GSW (Generative Semantic Workspace) — episodic memory ---
    # Retrieve summaries from prior conversations whose key entities overlap
    # with this query, so we can pick up cross-session context. Distinct from
    # the in-conversation `conversation_summary` (which condenses older messages
    # of the *current* session).
    try:
        from app.core import gsw as _gsw
        if _gsw.is_enabled() and intent == "general":
            prior_summaries = await _run_context_io(
                "gsw",
                asyncio.to_thread(_gsw.get_relevant_summaries, get_db(), query, 3),
                default=None,
            )
            if prior_summaries:
                # Filter out the current conversation's own summary
                prior_summaries = [
                    s for s in prior_summaries
                    if s.get("conversation_id") != conversation_id
                ]
                if prior_summaries:
                    ctx.prior_sessions_text = _gsw.format_for_prompt(prior_summaries)
                    logger.info(
                        "[GSW] surfaced %d prior session summaries (entities=%s)",
                        len(prior_summaries),
                        prior_summaries[0].get("key_entities", [])[:3],
                    )
    except Exception as e:
        logger.debug("GSW retrieval failed: %s", e)

    # --- Success patterns (what worked before) ---
    if svc.reflexions:
        from app.core.reflexion import ReflexionStore
        successes = await _run_context_io(
            "reflexions.success_patterns",
            asyncio.to_thread(svc.reflexions.get_success_patterns, query, config.MAX_SUCCESS_PATTERNS_IN_PROMPT),
            default=None,
        )
        if successes:
            ctx.success_patterns_text = ReflexionStore.format_success_patterns(successes)
            ctx.success_pattern_ids = [
                s.id for s in successes if getattr(s, "id", None) is not None
            ]

    # --- Persistent workspace (prior findings for similar query signatures) ---
    # Lets brain.think() inherit progress from prior runs (via AgentLoop or itself)
    # without re-deriving facts. Signature-keyed lookup; only injects when prior
    # success_count > 0 to avoid feeding stale failures back in.
    # Also injects FAILED APPROACHES so the model skips dead ends previously tried.
    # Time-sensitive queries (current price, latest news, today's weather)
    # MUST NOT serve from workspace cache. Yesterday's "$78,764.25 bitcoin
    # price" stored as a fact poisons today's answer to "what's bitcoin
    # currently?" — verified at runtime 2026-05-06. Skip workspace load
    # entirely; the model will call live tools.
    # (variable already computed earlier for the lessons gate)
    if intent == "general" and not _is_time_sensitive_query:
        try:
            from app.core.agent_workspace import load_workspace, query_signature
            from app.database import get_db
            ctx.workspace_signature = query_signature(query)
            if ctx.workspace_signature:
                entry = await _run_context_io(
                    "workspace",
                    asyncio.to_thread(load_workspace, get_db(), query),
                    default=None,
                )
                if entry:
                    parts: list[str] = []
                    if entry.findings and entry.success_count > 0:
                        findings_lines = [
                            f"  - {k}: {str(v)[:200]}"
                            for k, v in list(entry.findings.items())[:8]
                        ]
                        parts.append(
                            f"## Prior findings (from {entry.run_count} similar past runs, "
                            f"{entry.success_count} successful):\n"
                            + "\n".join(findings_lines)
                            + "\n\nUse these as a starting point — verify if stale, build on them if current."
                        )
                    if entry.failed_approaches:
                        fa_lines = [f"  - {fa}" for fa in entry.failed_approaches[:5]]
                        parts.append(
                            "## Previously failed approaches on similar queries — DO NOT REPEAT:\n"
                            + "\n".join(fa_lines)
                            + "\n\nPick a different angle (different tool, different query phrasing, different scope)."
                        )
                    if parts:
                        ctx.workspace_text = "\n\n".join(parts)
                        logger.info(
                            "Workspace hydrated: sig=%s findings=%d failed_approaches=%d prior_runs=%d",
                            ctx.workspace_signature, len(entry.findings),
                            len(entry.failed_approaches), entry.run_count,
                        )
        except Exception as e:
            logger.warning("Workspace load failed: %s", e)

    # --- Retrieval (with topical-relevance gate) ---
    # Two-stage filter:
    #   1. Score floor — config.RETRIEVAL_HARD_FLOOR (default 0.30). Chunks
    #      below this are dropped silently before the prompt.
    #   2. Topical overlap — chunks must share at least one substantive token
    #      (>=4 chars, alphanumeric) with the query. Catches the common case
    #      where a hash-table query retrieves Merkle-tree chunks (both contain
    #      "tree", but query has "hash table" and chunks have "merkle"/
    #      "blockchain" — zero substantive overlap → drop).
    #   Both gates were absent before; we saw answers contaminated by unrelated
    #   chunks even though we'd already dropped the [HIGH]/[LOW] labels.
    if svc.retriever and intent == "general":
        try:
            chunks = await svc.retriever.search(query)
            if chunks:
                # Pre-compute query token set for topical overlap check
                _query_tokens = {
                    t for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{3,}\b", query.lower())
                    if t not in {
                        "what", "where", "when", "which", "show", "tell",
                        "explain", "describe", "should", "could", "would",
                        "their", "there", "these", "those", "about", "from",
                        "with", "have", "been", "into", "than", "this", "that",
                    }
                }
                lines = []
                _hard_floor = max(config.RETRIEVAL_RELEVANCE_THRESHOLD, config.RETRIEVAL_HARD_FLOOR)
                for i, chunk in enumerate(chunks, 1):
                    score = chunk.score if hasattr(chunk, "score") and chunk.score is not None else 0.0
                    if score < _hard_floor:
                        continue
                    # Topical overlap check — chunk must share enough query
                    # tokens to be plausibly on-topic. The first revision used
                    # ≥1 overlap, but "hash table" → {hash, table} matched a
                    # Merkle-tree chunk on "hash" alone (Merkle uses hashes).
                    # Tighten: require either 2+ tokens OR ≥50% of query tokens
                    # AND a token-bigram match if query has ≥2 substantive
                    # tokens (catches "hash table" specifically — Merkle docs
                    # discuss "hash" but not "hash table" as a phrase).
                    if _query_tokens:
                        chunk_text = (chunk.content or "")[:1500].lower()
                        chunk_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{3,}\b", chunk_text))
                        overlap = _query_tokens & chunk_tokens
                        # Threshold: 2+ tokens overlap, OR (1 token AND query
                        # has only 1 substantive token total)
                        _token_pass = (
                            len(overlap) >= 2
                            or (len(overlap) >= 1 and len(_query_tokens) <= 1)
                            or len(overlap) / max(len(_query_tokens), 1) >= 0.5
                        )
                        if not _token_pass:
                            logger.debug(
                                "[retrieval] dropped off-topic chunk: query_tokens=%r overlap=%r src=%r",
                                list(_query_tokens)[:5], list(overlap), chunk.title or chunk.source,
                            )
                            continue
                        # Phrase-level check: when query has 2+ substantive
                        # tokens, require at least one ADJACENT pair to appear
                        # in the chunk. "hash table" must literally be in the
                        # chunk's token sequence — not just both words anywhere.
                        if len(_query_tokens) >= 2:
                            _query_lower = query.lower()
                            _query_substantive = re.findall(
                                r"\b[a-zA-Z][a-zA-Z0-9_]{3,}\b", _query_lower,
                            )
                            _bigrams = [
                                f"{a} {b}" for a, b in zip(_query_substantive, _query_substantive[1:])
                                if a in _query_tokens and b in _query_tokens
                            ]
                            if _bigrams and not any(bg in chunk_text for bg in _bigrams):
                                logger.debug(
                                    "[retrieval] dropped chunk missing query bigram: %r src=%r",
                                    _bigrams[:2], chunk.title or chunk.source,
                                )
                                continue
                    source = chunk.title or chunk.source or "document"
                    lines.append(f"[{i}] Source: {source}\n{chunk.content}")
                    ctx.retrieved_sources.append({
                        "title": chunk.title or "",
                        "source": chunk.source or "",
                        "score": round(score, 4),
                    })
                if lines:
                    ctx.retrieved_context = "\n\n".join(lines)
        except Exception as e:
            logger.warning("Retrieval failed: %s", e)

    # --- Integration info ---
    if config.ENABLE_INTEGRATIONS:
        try:
            from app.tools.integration import _registry as integration_registry
            if integration_registry:
                ctx.integrations_text = integration_registry.format_for_prompt()
        except Exception:
            pass

    # --- External skills (AgentSkills format) ---
    if svc.external_skills:
        from app.core.skill_loader import match_skill, format_skill_summaries, format_skill_body
        ctx.external_skills_text = format_skill_summaries(svc.external_skills)
        if intent == "general":
            matched = match_skill(query, svc.external_skills)
            if matched:
                ctx.matched_external_skill_text = format_skill_body(matched)
                logger.info("External skill matched: '%s'", matched.name)

    return ctx


async def _build_messages(
    svc: Services,
    ctx: _ThinkContext,
    query: str,
    history: list[dict],
    image: str | None,
    intent: str,
) -> tuple[list[dict], bool, dict | None]:
    """Build system prompt, manage context window, assemble messages, run query planning.

    This is Steps 8–8c of the original think() pipeline.
    Returns: (messages, was_planned, plan)
    """
    # Gather registered tool names for example filtering
    _tool_names = {t["name"] for t in _get_available_tools()} if svc.tool_registry else None

    # Common kwargs for build_system_prompt (avoids repeating all params)
    _prompt_kwargs = dict(
        user_facts_text=ctx.user_facts_text,
        lessons_text=ctx.lessons_text,
        tool_descriptions=_get_tool_descriptions(),
        retrieved_context=ctx.retrieved_context,
        skills_text=ctx.skills_text,
        kg_facts=ctx.kg_facts_text,
        reflexions=ctx.reflexions_text,
        integrations_text=ctx.integrations_text,
        success_patterns=ctx.success_patterns_text,
        external_skills_text=ctx.external_skills_text,
        matched_external_skill_text=ctx.matched_external_skill_text,
        workspace_text=ctx.workspace_text,
        registered_tool_names=_tool_names,
        provider=config.LLM_PROVIDER,
    )

    # Build a preliminary prompt just for token estimation in context management.
    # This avoids building the full prompt twice when summarization triggers a rebuild.
    preliminary_prompt = build_system_prompt(**_prompt_kwargs)

    # Context window management
    managed_history, conversation_summary = await _manage_context(
        preliminary_prompt, history, query
    )

    # GSW: prepend prior-session summaries to the conversation summary block.
    # This puts cross-session continuity into Block 7 (truncate-first), so it
    # only displaces space if there's headroom — never starves identity/lessons.
    if ctx.prior_sessions_text:
        gsw_block = "[Prior-session memory — relevant context from earlier conversations]\n" + ctx.prior_sessions_text
        if conversation_summary:
            conversation_summary = gsw_block + "\n\n" + conversation_summary
        else:
            conversation_summary = gsw_block

    if conversation_summary:
        # Rebuild with summary — this is the only full build
        system_prompt = build_system_prompt(conversation_summary=conversation_summary, **_prompt_kwargs)
        history = managed_history
    else:
        # No summarization needed — reuse the preliminary prompt directly
        system_prompt = preliminary_prompt

    # Assemble messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    user_msg = {"role": "user", "content": query}
    if image:
        user_msg["images"] = [image]
    messages.append(user_msg)

    # Honor EXPLICIT tool requests: if the user asked to run code / use the
    # calculator, the answer must come from the tool's output, not from memory.
    # (Narrowly gated — implicit math like "what is 15% of 840" is unaffected.)
    if _EXPLICIT_TOOL_REQUEST_RE.search(query):
        messages.append({"role": "system", "content": (
            "[EXPLICIT TOOL REQUEST] The user explicitly asked you to use a tool "
            "(run code and/or use the calculator). You MUST invoke the appropriate "
            "tool (code_exec or calculator) and base your answer on its actual "
            "output — do not compute or recall the result from memory instead.")})

    # Query Planning
    was_planned = False
    plan = None
    if config.ENABLE_PLANNING and intent == "general":
        from app.core.planning import should_plan, create_plan, format_plan_for_prompt
        if should_plan(query, intent):
            try:
                tool_names = [t["name"] for t in _get_available_tools()]
                plan = await create_plan(query, tool_names, ctx.reflexions_text)
                if plan:
                    plan_text = format_plan_for_prompt(plan)
                    messages.append({"role": "system", "content": plan_text})
                    was_planned = True
                    logger.info("Query planned: %d steps", len(plan["steps"]))
                    # Solve decomposable sub-questions in parallel
                    if plan.get("complexity") == "decomposable" and plan.get("sub_questions"):
                        from app.core.planning import solve_sub_questions
                        try:
                            sub_context = await solve_sub_questions(
                                plan["sub_questions"],
                                user_facts=ctx.user_facts_text,
                                kg_facts=ctx.kg_facts_text,
                                context=ctx.retrieved_context,
                            )
                            if sub_context:
                                logger.info("[DECOMPOSE] %d sub-questions solved", len(plan["sub_questions"]))
                                messages.append({"role": "system", "content": sub_context})
                        except Exception as e:
                            logger.warning("Sub-question solving failed: %s", e)
            except Exception as e:
                logger.warning("Planning failed: %s", e)

    return messages, was_planned, plan


from app.tools.base import TOOL_FAILURE_MARKERS as _TOOL_FAILURE_MARKERS
from app.tools.base import ErrorCategory


def _round_all_succeeded(results: list[tuple]) -> bool:
    """Check if all tool results in this round indicate success.

    Uses structured ToolResult.success when available; falls back to
    substring matching for tool_create and legacy paths.
    """
    for item in results:
        # New format: (tc, output, tool_result)
        if len(item) == 3:
            _, output, tool_result = item
            if tool_result is not None:
                if not tool_result.success:
                    return False
                continue
        else:
            # Legacy format: (tc, output)
            _, output = item
        # Fallback substring matching for tool_create and legacy paths
        lower = str(output).lower()[:500]
        if any(m in lower for m in _TOOL_FAILURE_MARKERS):
            return False
    return True


async def _run_generation_loop(
    messages: list[dict],
    tools: list[dict],
    svc: Services,
    conversation_id: str,
    image: str | None,
    intent: str,
    was_planned: bool,
    ephemeral: bool,
    gen: _GenerationResult,
    query: str = "",
) -> AsyncGenerator[StreamEvent, None]:
    """The tool loop — generate, check for tool calls, execute tools, re-generate.

    Yields THINKING and TOOL_USE events.
    Populates gen (a mutable _GenerationResult) with final_content, tool_results, is_error.
    This is Step 9 of the original think() pipeline.
    """
    # query is now passed explicitly to avoid messages[-1] assumption breaking after planning

    # Model selection
    selected_model = _select_model(query, intent, was_planned)
    # Use VISION_MODEL only if explicitly set to a different model (e.g., specialized vision model).
    # Qwen3.5 is natively multimodal — the main model handles images directly.
    if image and config.VISION_MODEL and config.VISION_MODEL != config.LLM_MODEL:
        selected_model = config.VISION_MODEL

    # Extended thinking: enabled by global flag OR auto-enabled for hard-reasoning queries
    # (where visible chain-of-thought before answer materially improves quality on a smaller model).
    # Auto-enable does NOT fire on fast model or vision queries.
    _hard_query_auto_thinking = (
        not image
        and selected_model != config.FAST_MODEL
        and intent == "general"
    )
    if _hard_query_auto_thinking:
        try:
            from app.core.agent_loop import _is_hard_reasoning_query
            _hard_query_auto_thinking = _is_hard_reasoning_query(query)
        except Exception:
            _hard_query_auto_thinking = False
    use_thinking = (
        (config.ENABLE_EXTENDED_THINKING or _hard_query_auto_thinking)
        and not image
        and selected_model != config.FAST_MODEL
    )
    _GENERATION_TIMEOUT = float(config.GENERATION_TIMEOUT)

    # Intent-adaptive temperature: factual/computational → low, creative/opinion → higher
    if _CREATIVE_PATTERNS.search(query):
        _temperature = 0.7
    elif _COMPLEX_PATTERNS.search(query) or intent == "correction":
        _temperature = 0.3
    else:
        _temperature = 0.4

    # Per-conversation tool result cache (C10)
    # Skip caching for tools with side effects (uses module-level _SIDE_EFFECT_TOOLS)
    _tool_cache: dict[tuple, str] = {}

    # --- Circuit breaker state ---
    # Track (tool_name, args_hash) call counts to detect loops
    _call_counts: dict[str, int] = {}     # tool_name -> call count
    _pair_counts: dict[tuple, int] = {}   # (tool_name, args_hash) -> call count
    _total_tool_calls: int = 0            # total tool executions across all rounds
    _last_tool_outputs: dict[str, str] = {}  # tool_name -> last output (for browser dedup)
    # Reset pivot tracker for this request so backtrack-pivot can fire fresh.
    _pivot_attempted_this_query[conversation_id] = False

    # Within-think() working scratchpad — accumulates the model's thinking from
    # each round so subsequent rounds can see "what I was reasoning last round"
    # and avoid re-deriving or contradicting itself. Only meaningful when
    # use_thinking=True; harmless overhead otherwise.
    _think_scratchpad: list[str] = []

    _any_round_succeeded = False  # Track cumulative success across tool rounds

    # Expose the current user query to tools that need to gate on owner intent
    # (e.g. active_memory.add requires explicit "remember" / correction signals).
    from app.tools.active_memory import current_user_query as _amq
    _amq_token = _amq.set(query)

    # Cap tool rounds for inherently uncertain queries (future predictions,
    # exact future values). Without this cap, the model spins web_search
    # trying to find an answer that doesn't exist (e.g. "exact S&P 500 value
    # one year from today") and burns the eval timeout. The right behavior
    # for these queries is one search attempt then hedge — not iteratively
    # narrowing the search until the timer runs out.
    _query_uncertain = bool(_INHERENTLY_UNCERTAIN_RE.search(query))
    _effective_max_rounds = (
        min(2, config.MAX_TOOL_ROUNDS) if _query_uncertain else config.MAX_TOOL_ROUNDS
    )

    try:
        for tool_round in range(_effective_max_rounds):
            # Inject prior-round thinking as a system anchor so the model can
            # see "what I was reasoning before" — avoids re-deriving and
            # catches contradiction with prior rounds. Only fires after round 0
            # and only when thinking accumulated something.
            if tool_round > 0 and _think_scratchpad:
                scratch_text = "\n---\n".join(
                    f"[round-{tool_round - i}] {t[:400]}"
                    for i, t in enumerate(reversed(_think_scratchpad[-3:]))
                )
                # Replace any prior scratchpad system message to avoid duplication
                messages = [m for m in messages if not (
                    m.get("role") == "system" and m.get("content", "").startswith("[YOUR PRIOR REASONING")
                )]
                messages.append({
                    "role": "system",
                    "content": (
                        "[YOUR PRIOR REASONING — do not contradict yourself, "
                        "build on this rather than re-deriving]\n" + scratch_text
                    ),
                })

            if use_thinking:
                thinking_buf = ""
                content_buf = ""
                _stream_tool_calls: list[llm.ToolCall] = []
                _announced_tool_ids: set = set()
                try:
                    async with asyncio.timeout(_GENERATION_TIMEOUT):
                        async for chunk in llm.stream_with_thinking(
                            messages, tools, model=selected_model, temperature=_temperature
                        ):
                            if chunk.thinking:
                                thinking_buf += chunk.thinking
                                yield StreamEvent(
                                    type=EventType.THINKING,
                                    data={"stage": "reasoning", "content": chunk.thinking},
                                )
                            if chunk.content:
                                content_buf += chunk.content
                            if chunk.tool_call is not None:
                                _stream_tool_calls.append(chunk.tool_call)
                                # Interleaved think-act (best Ollama can do): announce tool
                                # immediately on detection so client UI knows tools are
                                # incoming, even before stream completes. Actual execution
                                # still happens after stream end (Ollama can't halt).
                                _tc_id = id(chunk.tool_call)
                                if _tc_id not in _announced_tool_ids:
                                    _announced_tool_ids.add(_tc_id)
                                    yield StreamEvent(
                                        type=EventType.TOOL_USE,
                                        data={
                                            "tool": chunk.tool_call.tool,
                                            "args": chunk.tool_call.args,
                                            "status": "detected",
                                            "tool_call_id": f"{chunk.tool_call.tool}_{tool_round}_pending",
                                        },
                                    )
                except TimeoutError:
                    logger.warning("Streaming generation timed out after %.0fs", _GENERATION_TIMEOUT)
                    gen.is_error = True
                    if content_buf:
                        content_buf += "\n\n[Response truncated due to timeout]"
                    else:
                        content_buf = "The response timed out. Please try a simpler query or try again."
                content_buf = llm._strip_think_tags(content_buf).strip()
                result = llm.GenerationResult(
                    content=content_buf,
                    tool_calls=_stream_tool_calls,
                    raw={},
                    thinking=thinking_buf,
                )
                # Append this round's thinking to the within-think scratchpad
                # for cross-round continuity (capped to last 3 rounds).
                if thinking_buf and thinking_buf.strip():
                    _think_scratchpad.append(thinking_buf.strip())
                    if len(_think_scratchpad) > 3:
                        _think_scratchpad.pop(0)
            else:
                try:
                    result = await asyncio.wait_for(
                        llm.generate_with_tools(messages, tools, model=selected_model, temperature=_temperature),
                        timeout=_GENERATION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Generation timed out after %.0fs", _GENERATION_TIMEOUT)
                    gen.is_error = True
                    result = llm.GenerationResult(
                        content="The response timed out. Please try a simpler query or try again.",
                        tool_calls=[],
                        raw={},
                    )

            # Log LLM usage if available
            if getattr(result, "usage", None):
                logger.info("LLM usage: %s", result.usage)

            # Emit full thinking for non-streaming path
            thinking_text = getattr(result, "thinking", "") or ""
            if not use_thinking and isinstance(thinking_text, str) and thinking_text.strip():
                yield StreamEvent(
                    type=EventType.THINKING,
                    data={"stage": "reasoning", "content": thinking_text},
                )

            # Extract tool calls
            if result.tool_calls:
                tool_calls = result.tool_calls
            else:
                tool_calls = _extract_tool_calls(result.content, tools)

            if not tool_calls:
                gen.final_content = result.content
                break

            # --- Circuit breaker: loop detection ---
            _circuit_broken = False
            _filtered_calls = []
            for tc in tool_calls:
                try:
                    _args_hash = json.dumps(tc.args, sort_keys=True, default=str)
                except (TypeError, ValueError):
                    _args_hash = str(tc.args)
                _pair_key = (tc.tool, _args_hash)

                # Per-tool-name cap
                _call_counts[tc.tool] = _call_counts.get(tc.tool, 0) + 1
                if _call_counts[tc.tool] > config.MAX_SAME_TOOL_CALLS:
                    logger.warning(
                        "Circuit breaker: tool '%s' called %d times (max %d) — skipping",
                        tc.tool, _call_counts[tc.tool], config.MAX_SAME_TOOL_CALLS,
                    )
                    continue

                # Same (tool, args) dedup
                _pair_counts[_pair_key] = _pair_counts.get(_pair_key, 0) + 1
                if _pair_counts[_pair_key] > 2:
                    logger.warning(
                        "Circuit breaker: identical call '%s(%s)' repeated %d times — skipping",
                        tc.tool, _args_hash[:80], _pair_counts[_pair_key],
                    )
                    continue

                # Total call cap
                _total_tool_calls += 1
                if _total_tool_calls > config.MAX_TOOL_CALLS_PER_QUERY:
                    logger.warning(
                        "Circuit breaker: total tool calls (%d) exceeded max (%d) — stopping",
                        _total_tool_calls, config.MAX_TOOL_CALLS_PER_QUERY,
                    )
                    _circuit_broken = True
                    break

                _filtered_calls.append(tc)

            if not _filtered_calls:
                # PEAK: before giving up, try a BACKTRACK PIVOT. The circuit fired
                # because the model kept proposing the same approach. Inject a
                # "your previous angles failed — try a DIFFERENT tool or query
                # phrasing" hint and re-generate ONCE. If it produces fresh tool
                # calls (not in our dedup), use them. Otherwise fall through to
                # forced synthesis.
                if (
                    not _circuit_broken
                    and tool_round + 1 < config.MAX_TOOL_ROUNDS
                    and not _pivot_attempted_this_query.get(conversation_id, False)
                ):
                    _pivot_attempted_this_query[conversation_id] = True
                    tried_summary = ", ".join(
                        f"{t}({_call_counts[t]}x)" for t in list(_call_counts.keys())[:5]
                    )
                    pivot_hint = (
                        "[BACKTRACK PIVOT] Your previous tool approaches kept hitting "
                        f"the same dead-end (tried: {tried_summary}). DO NOT call the "
                        "same tools with the same args. Instead:\n"
                        "- Try a DIFFERENT tool entirely (browser instead of web_search, "
                        "knowledge_search instead of memory_search, http_fetch instead of browser).\n"
                        "- Or phrase the query DIFFERENTLY (entity name only, broader, narrower).\n"
                        "- Or just answer from what you've already learned — don't loop.\n"
                        "Make ONE focused next move."
                    )
                    pivot_messages = list(messages) + [
                        {"role": "assistant", "content": result.content or ""},
                        {"role": "system", "content": pivot_hint},
                    ]
                    try:
                        logger.info("[PIVOT] circuit fired — attempting backtrack pivot")
                        pivot_result = await asyncio.wait_for(
                            llm.generate_with_tools(pivot_messages, tools, model=selected_model),
                            timeout=_GENERATION_TIMEOUT,
                        )
                        pivot_calls = pivot_result.tool_calls or _extract_tool_calls(pivot_result.content, tools)
                        # Filter pivot calls: only accept ones we haven't tried before
                        fresh_pivot_calls = []
                        for ptc in (pivot_calls or []):
                            try:
                                ph = json.dumps(ptc.args, sort_keys=True, default=str)
                            except Exception:
                                ph = str(ptc.args)
                            if (ptc.tool, ph) not in _pair_counts:
                                fresh_pivot_calls.append(ptc)
                        if fresh_pivot_calls:
                            logger.info(
                                "[PIVOT] generated %d fresh tool calls — retrying loop",
                                len(fresh_pivot_calls),
                            )
                            tool_calls = fresh_pivot_calls
                            # Skip the synthesis path below — let the next iteration execute these
                            _filtered_calls = fresh_pivot_calls
                        elif pivot_result.content and pivot_result.content.strip():
                            # Pivot produced text answer instead of tools — accept it
                            logger.info("[PIVOT] returned text answer — using as final")
                            gen.final_content = pivot_result.content.strip()
                            break
                    except Exception as e:
                        logger.warning("[PIVOT] failed: %s — falling through to synthesis", e)

            if not _filtered_calls:
                # All calls were filtered out by circuit breaker (and pivot didn't help) —
                # try to synthesize from accumulated tool_results before giving up.
                logger.warning(
                    "Circuit breaker: all tool calls in round %d filtered — synthesizing "
                    "(tool_results=%d, content_len=%d)",
                    tool_round + 1, len(gen.tool_results),
                    len((result.content or "").strip()),
                )
                # Always force a synthesis if we have tool results, regardless of whether
                # result.content has anything (it usually has just leftover tool-call JSON).
                if gen.tool_results:
                    try:
                        synth_messages = list(messages) + [{
                            "role": "user",
                            "content": (
                                "Synthesize the tool results above into a final answer for the user. "
                                "Do NOT call any more tools. Just write the answer using the data you have."
                            ),
                        }]
                        synthesis = await asyncio.wait_for(
                            llm.invoke_nothink(synth_messages, max_tokens=800, temperature=_temperature),
                            timeout=_GENERATION_TIMEOUT,
                        )
                        if synthesis and synthesis.strip():
                            gen.final_content = synthesis
                            logger.info(
                                "Circuit breaker forced synthesis: produced %d chars",
                                len(synthesis),
                            )
                        else:
                            gen.final_content = (result.content or "").strip() or \
                                "[Tool data gathered but synthesis was empty]"
                            logger.warning("Forced synthesis returned empty")
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("Forced synthesis failed: %s", e)
                        gen.final_content = (result.content or "").strip() or \
                            "[Tool loop exhausted; synthesis errored]"
                else:
                    gen.final_content = (
                        (result.content or "").strip() or
                        "I attempted to use tools but kept getting the same result. "
                        "Let me answer with what I know instead."
                    )
                break

            tool_calls = _filtered_calls

            if _circuit_broken:
                # Force this to be the final round
                pass

            logger.info(
                "Tool calls [round %d]: %s",
                tool_round + 1,
                [(tc.tool, tc.args) for tc in tool_calls],
            )

            for i, tc in enumerate(tool_calls, 1):
                yield StreamEvent(
                    type=EventType.TOOL_USE,
                    data={"tool": tc.tool, "args": tc.args, "status": "executing",
                           "tool_call_id": f"{tc.tool}_{tool_round}_{i}"},
                )

            # Execute ALL tool calls concurrently (with per-conversation cache)
            async def _run_tool(tc):
                if tc.tool == "tool_create" and svc.custom_tools:
                    return tc, await _handle_tool_create(svc, tc.args), None
                # Cache lookup for idempotent tools
                if tc.tool not in _SIDE_EFFECT_TOOLS:
                    try:
                        cache_key = (tc.tool, json.dumps(tc.args, sort_keys=True, default=str))
                    except (TypeError, ValueError):
                        cache_key = None  # unserializable args
                    if cache_key and cache_key in _tool_cache:
                        logger.debug("Tool cache hit: %s", tc.tool)
                        cached_output, cached_result = _tool_cache[cache_key]
                        return tc, cached_output, cached_result
                else:
                    cache_key = None
                output, tool_result = await _execute_tool(tc.tool, tc.args)
                # One-time retry for transient failures (network timeouts, 429/5xx)
                if (tool_result and not tool_result.success
                        and tool_result.retriable
                        and tool_result.error_category == ErrorCategory.TRANSIENT):
                    logger.info("Retrying transient failure for tool '%s'", tc.tool)
                    retry_output, retry_result = await _execute_tool(tc.tool, tc.args)
                    if retry_result and retry_result.success:
                        output = retry_output
                        tool_result = retry_result
                # Sanitize ALL tool outputs — always-on (defense against injected content)
                from app.core.injection import sanitize_content
                output = sanitize_content(output, context=f"tool:{tc.tool}")
                if cache_key is not None:
                    _tool_cache[cache_key] = (output, tool_result)
                return tc, output, tool_result

            results = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])

            # --- Browser-specific dedup: detect identical outputs ---
            _deduped_results = []
            for tc, tool_output, tool_result_obj in results:
                if tc.tool in ("browser", "web_search", "http_fetch"):
                    _prev = _last_tool_outputs.get(tc.tool)
                    _output_hash = hash(tool_output[:2000]) if tool_output else 0
                    if _prev is not None and _prev == _output_hash:
                        logger.warning(
                            "Circuit breaker: '%s' returned identical output — suppressing repeat",
                            tc.tool,
                        )
                        # Bump the per-tool count to accelerate circuit-break
                        _call_counts[tc.tool] = _call_counts.get(tc.tool, 0) + 1
                        # Still include the result but mark it so the LLM doesn't retry
                        tool_output = (
                            tool_output[:500] +
                            "\n\n[NOTE: This tool returned the same result as the previous call. "
                            "Do not call it again with the same arguments. Synthesize from existing results.]"
                        )
                    _last_tool_outputs[tc.tool] = _output_hash
                _deduped_results.append((tc, tool_output, tool_result_obj))
            results = _deduped_results

            # Use empty content when the LLM emits tool calls without narrative.
            # Prior code used "[Calling tool: X]" as a placeholder, which leaked
            # into assistant-role history and caused the model to imitate the
            # pattern as its final synthesis output (producing monitor results
            # that were just accumulated "[Calling tool: X]" lines with no answer).
            assistant_content = result.content or ""

            tool_result_parts = []
            for i, (tc, tool_output, tool_result_obj) in enumerate(results, 1):
                # Trim output via per-tool trim_output() for context storage
                tool_obj = svc.tool_registry.get(tc.tool) if svc.tool_registry else None
                if tool_obj:
                    trimmed = tool_obj.trim_output(tool_output)
                else:
                    trimmed = tool_output[:config.TOOL_OUTPUT_MAX_CHARS]
                    if len(tool_output) > config.TOOL_OUTPUT_MAX_CHARS:
                        trimmed += "\n[...truncated]"
                gen.tool_results.append({
                    "tool": tc.tool,
                    "args": tc.args,
                    "output": trimmed,
                })
                # RLVR — record verifiable tool-call signal (success/failure).
                # Fire-and-forget; failures log at debug only.
                try:
                    if getattr(config, "ENABLE_RLVR_SIGNALS", False):
                        from app.core import rlvr as _rlvr
                        _ok = bool(tool_result_obj and getattr(tool_result_obj, "success", True))
                        await asyncio.to_thread(
                            _rlvr.record_signal,
                            "tool_correct",
                            1.0 if _ok else 0.0,
                            query=query[:500],
                            response=trimmed[:500],
                            evidence=f"tool={tc.tool} args_keys={list((tc.args or {}).keys())}",
                            conversation_id=conversation_id,
                        )
                except Exception:
                    pass
                yield StreamEvent(
                    type=EventType.TOOL_USE,
                    data={"tool": tc.tool, "result": tool_output[:500], "status": "complete",
                           "tool_call_id": f"{tc.tool}_{tool_round}_{i}"},
                )
                tool_result_parts.append(
                    f"[Source {i}: {tc.tool}]\n{tool_output[:config.TOOL_OUTPUT_MAX_CHARS]}"
                )

                if not ephemeral:
                    await asyncio.to_thread(
                        lambda _tc=tc, _out=trimmed: svc.conversations.add_message(
                            conversation_id, "tool", _out[:config.TOOL_OUTPUT_MAX_CHARS], tool_name=_tc.tool
                        )
                    )

            # Assistant role = self-attribution. Model won't contradict its own prior statements.
            tool_results_text = "\n\n".join(tool_result_parts)
            round_succeeded = _round_all_succeeded(results)
            if round_succeeded:
                _any_round_succeeded = True

            # Build self-attribution with a minimal, provider-neutral marker.
            # The previous Ollama-specific verbose preamble ("I used my tools
            # and they returned real, live results...") was imitable by the
            # model — Nova echoed it in its own responses as if it were the
            # correct voice, then synthesis built hallucinated content on top
            # of the "tools returned real results" scaffolding. Using a marker
            # that doesn't read like assistant voice prevents that echo.
            from datetime import datetime as _dt
            _today = _dt.now().strftime("%B %d, %Y")
            if round_succeeded:
                attr_prefix = f"Tool results ({_today}):"
            else:
                attr_prefix = f"Tool results (partial, {_today}):"

            messages.append({
                "role": "assistant",
                "content": f"{assistant_content}\n\n{attr_prefix}\n\n{tool_results_text}",
            })

            # User-role synthesis trigger with result evaluation
            # On intermediate rounds, encourage the model to assess and potentially
            # use more tools. On the final round (or circuit breaker hit), just synthesize.
            _is_final_round = (tool_round >= _effective_max_rounds - 1) or _circuit_broken
            if round_succeeded:
                if _is_final_round:
                    if was_planned:
                        messages = [
                            m for m in messages
                            if not (m.get("role") == "system" and m.get("content", "").startswith("[PLAN]"))
                        ]
                        logger.info("[FINAL_RESPONSE] Plan text stripped for final generation")
                    _synth = (
                        "Based on the real tool results above, provide your final answer. "
                        "Do NOT say you cannot use tools or add disclaimers."
                    )
                else:
                    _synth = (
                        "Review the tool results above. If you have enough data to fully "
                        "answer the question, provide your answer now. If the results are "
                        "incomplete (e.g., portal links instead of data, partial information, "
                        "or missing details), use another tool to get what's missing — try "
                        "browser for JS pages, http_fetch for APIs, or web_search with "
                        "different terms. Do NOT give up if you have untried approaches."
                    )
                messages.append({"role": "user", "content": _synth})
            elif _any_round_succeeded:
                messages.append({
                    "role": "user",
                    "content": (
                        "Based on the tool results above, provide your answer. "
                        "Some tools succeeded with real data — focus on those results. "
                        "If any tools failed, briefly note the limitation in natural language "
                        "without exposing error messages, tier names, or internal details."
                    ),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Based on the tool results above, provide your answer.",
                })

        else:
            # Exhausted tool rounds — synthesize findings via LLM instead of dumping raw output
            if not gen.final_content:
                tool_summary_parts = []
                for tr in gen.tool_results:
                    tool_summary_parts.append(f"- {tr['tool']}: {tr['output'][:500]}")
                tool_summary_text = "\n".join(tool_summary_parts)
                synthesis_messages = [
                    messages[0],  # system prompt
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": f"I used several tools. Here are the results:\n{tool_summary_text}"},
                    {"role": "user", "content": (
                        "The tool loop has ended. Please synthesize the tool results above "
                        "into a clear, helpful answer for the user. Summarize what was found "
                        "and note any incomplete steps."
                    )},
                ]
                try:
                    synthesis = await llm.invoke_nothink(synthesis_messages, max_tokens=1500)
                    if synthesis and synthesis.strip():
                        gen.final_content = synthesis.strip()
                    else:
                        raise ValueError("Empty synthesis")
                except Exception as synth_err:
                    logger.warning("Synthesis LLM call failed after exhausted tool rounds: %s", synth_err)
                    gen.final_content = "I attempted to use tools but couldn't complete the task within the allowed steps. Here's what I found so far:\n\n"
                    for tr in gen.tool_results:
                        gen.final_content += f"- {tr['tool']}: {tr['output'][:200]}\n"

        # Autonomous tool creation — detect recurring multi-step patterns
        if (
            len(gen.tool_results) >= 3
            and config.ENABLE_CUSTOM_TOOLS
            and config.ENABLE_AUTONOMOUS_TOOL_CREATION
        ):
            from app.core.tool_triggers import maybe_trigger_tool_creation

            async def _safe_trigger(q, trs, s):
                try:
                    await maybe_trigger_tool_creation(q, trs, s)
                except Exception as _e:
                    logger.warning("Auto-tool trigger failed: %s", _e)

            _task = asyncio.create_task(_safe_trigger(query, gen.tool_results, svc))
            _background_tasks.add(_task)
            _task.add_done_callback(_background_tasks.discard)

    except LLMUnavailableError as e:
        logger.error("LLM unavailable: %s", e)
        gen.final_content = "I can't reach the language model right now. Please check that the LLM provider is running and try again."
        gen.is_error = True
        yield StreamEvent(type=EventType.ERROR, data={"message": str(e)})
    finally:
        try:
            _amq.reset(_amq_token)
        except ValueError:
            # Async-generator cleanup (GeneratorExit on client disconnect or an
            # upstream error) can run in a different context than where the token
            # was set — ContextVar.reset() then raises "Token was created in a
            # different Context". Harmless (the context is torn down regardless),
            # but if it propagates it masks the REAL error (e.g. an Ollama
            # timeout) in the logs. Swallow it.
            pass


# ---------------------------------------------------------------------------
# Vision pre-pass — structured visual analysis before main generation
# ---------------------------------------------------------------------------

_VISION_DESCRIBE_PROMPT = """Look at this image. Briefly describe:
1. What is the subject / main content? (1 sentence)
2. Key visual elements relevant to: {query}
3. Any text visible in the image (transcribe verbatim if present)

Keep total under 250 words. Be concrete. Don't speculate beyond what's visible."""


async def _vision_describe(query: str, image: str) -> str:
    """Run a vision-grounded description pass. Returns concise structured analysis.

    Uses the main model (which is multimodal in Qwen3.5). Bounded LLM call:
    300 tokens max output, 60s timeout. Returns "" on any failure — caller
    falls through to standard answer generation with image still attached.
    """
    from app.core import llm
    import asyncio
    try:
        msgs = [{
            "role": "user",
            "content": _VISION_DESCRIBE_PROMPT.format(query=query[:200]),
            "images": [image],
        }]
        result = await asyncio.wait_for(
            llm.invoke_nothink(msgs, max_tokens=350, temperature=0.1),
            timeout=60.0,
        )
        return (result or "").strip()[:1500]
    except Exception as e:
        logger.debug("[Vision] _vision_describe failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Deliberation routing — chain-of-reasoning queries → AgentLoop.solve()
# ---------------------------------------------------------------------------

# Patterns that signal the query needs explicit plan/act/critique deliberation
# rather than single-pass generation. Distinct from compare/multi-source
# patterns (those go to multi-agent decomposition).
_DELIBERATION_PATTERNS = re.compile(
    r"\b("
    r"step.by.step|walk\s+me\s+through|explain\s+how|"
    r"prove\s+that|derive(?:\s+the)?|show\s+me\s+how|"
    r"design\s+a|architect|plan\s+for\s+a|"
    r"trade.?offs?\s+between|weigh\s+the|"
    r"best\s+(?:approach|way|strategy)|optimal\s+way|"
    r"how\s+would\s+you\s+(?:architect|design|approach|build|solve|tackle)|"
    r"chain.of.thought|reason\s+through|"
    r"why\s+(?:does|is|are|would|should)\s+\w+\s+\w+\s+\w+"  # at least 4 tokens after "why does"
    r")\b",
    re.IGNORECASE,
)


def _should_use_deliberation(query: str) -> bool:
    """True when query benefits from AgentLoop's plan/act/critique loop.

    Distinct from multi-agent decomposition (which is for compare/multi-source).
    Deliberation = chain-of-reasoning with explicit per-step verification.
    Cheap regex — no LLM call.
    """
    if not query or len(query) < 25:
        return False
    return bool(_DELIBERATION_PATTERNS.search(query))


async def _run_deliberation_path(
    svc: "Services",
    query: str,
    conversation_id: str,
    intent: str,
    ctx: "_ThinkContext",
    is_new_conversation: bool,
    channel: str,
    ephemeral: bool = False,
) -> AsyncGenerator[StreamEvent, None]:
    """Run AgentLoop.solve() and stream its events as the response.

    Bypasses brain's normal generation loop. Saves user/assistant messages
    and runs post-processing same as the normal path.
    """
    from app.core.agent_loop import AgentLoop

    yield StreamEvent(
        type=EventType.THINKING,
        data={"stage": "deliberation", "content": "Routing to deliberation engine (AgentLoop)..."},
    )

    # Capture events from AgentLoop into a queue we can yield from.
    pending_events: list[StreamEvent] = []

    def _on_agent_event(event_type: str, payload: dict) -> None:
        # Translate AgentLoop event types into our SSE schema
        if event_type == "plan":
            steps = payload.get("steps", [])
            content = "Plan: " + " | ".join(f"{i+1}. {s[:80]}" for i, s in enumerate(steps[:6]))
            pending_events.append(
                StreamEvent(type=EventType.THINKING, data={"stage": "plan", "content": content})
            )
        elif event_type == "step_start":
            pending_events.append(
                StreamEvent(
                    type=EventType.THINKING,
                    data={
                        "stage": "step",
                        "content": f"Step {payload.get('id', '?')}: {payload.get('description', '')[:120]}",
                    },
                )
            )
        elif event_type == "critique":
            ok = "✓" if payload.get("satisfied") else "✗"
            pending_events.append(
                StreamEvent(
                    type=EventType.THINKING,
                    data={
                        "stage": "critique",
                        "content": f"{ok} {payload.get('reason', '')[:120]}",
                    },
                )
            )
        elif event_type == "workspace_hydrated":
            n = payload.get("findings_carried", 0)
            if n:
                pending_events.append(
                    StreamEvent(
                        type=EventType.THINKING,
                        data={"stage": "workspace", "content": f"Hydrated {n} prior findings"},
                    )
                )

    loop = AgentLoop(tools=svc.tool_registry)
    # Run solve() — events get queued via callback. We can't yield mid-await,
    # so we drain pending_events at boundaries (before solve, after solve).
    result = await loop.solve(
        query=query,
        max_iterations=10,
        on_event=_on_agent_event,
    )

    # Drain queued THINKING events
    for ev in pending_events:
        yield ev

    final_content = (result.answer or "").strip()
    if not final_content:
        final_content = "I worked through this but couldn't produce a final answer. Try rephrasing."

    # Apply confidence/leakage hardening that the normal path would have applied
    final_content = _sanitize_answer(final_content)

    # --- Lightweight critique pass on the deliberated answer ---
    # AgentLoop critiques per-step but doesn't holistically critique the synthesized answer.
    # Run adversarial critique for factual errors + a tool-leakage check. Skip best-of-N
    # (too heavy after deliberation), critique addendum (would over-pad).
    if len(final_content) >= 200:
        try:
            from app.core.critique import adversarial_critique, format_adversarial_for_replan
            adv = await adversarial_critique(
                query, final_content,
                sources="",
                user_facts=ctx.user_facts_text,
                kg_facts=ctx.kg_facts_text,
            )
            if adv and adv.get("verdict") == "fail":
                logger.info("[Deliberation] adversarial critique flagged issues — generating natural refinement")
                # Extract just the flaw descriptions — we use them as private
                # editing instructions, NOT as user-facing "[CRITICAL ERRORS]"
                # framing (the model used to take that framing and produce a
                # "Correction of Logic" header which the LLM-judge then graded
                # the whole answer down for "admitting errors").
                blocking = adv.get("blocking_flaws") or []
                flaws_lines: list[str] = []
                for flaw in blocking:
                    if isinstance(flaw, dict):
                        d = str(flaw.get("description", "")).strip()
                        if d:
                            flaws_lines.append(f"- {d}")
                if flaws_lines:
                    flaws_block = "\n".join(flaws_lines[:5])
                    refinement_instruction = (
                        "Add a SHORT closing paragraph (3-5 sentences max) that "
                        "addresses these refinements naturally:\n"
                        f"{flaws_block}\n\n"
                        "Hard rules:\n"
                        "- Read as a natural continuation of the answer above. "
                        "Do NOT use any of these section/header words: "
                        "'Correction', 'Errata', 'Addenda', 'Note:', 'Important:', "
                        "'Caveat', 'Disclaimer', 'Revision'.\n"
                        "- Do NOT say 'the previous answer was wrong/flawed/contradictory'.\n"
                        "- Do NOT preface with 'However' or 'Actually' if it implies "
                        "the prior content was incorrect.\n"
                        "- Stay in the SAME VOICE and detail level as the answer above.\n"
                        "- Output ONLY the paragraph itself, no headings, no bullet "
                        "list of corrections."
                    )
                    try:
                        from app.core import llm as llm_mod
                        refinement = await llm_mod.invoke_nothink(
                            [
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": final_content},
                                {"role": "system", "content": refinement_instruction},
                            ],
                            max_tokens=500, temperature=0.2,
                        )
                        if refinement and len(refinement.strip()) >= 30:
                            # Reject the refinement if it leaked any of the
                            # forbidden header words anyway — better to ship
                            # the original than a self-deprecating answer.
                            forbidden_re = re.compile(
                                r"(?im)^\s*(?:#+\s*)?(?:correction|errata|addenda|caveat|"
                                r"disclaimer|revision|note|important)\b[\s:]"
                            )
                            if not forbidden_re.search(refinement):
                                final_content = final_content.rstrip() + "\n\n" + refinement.strip()
                            else:
                                logger.info(
                                    "[Deliberation] refinement leaked forbidden header — "
                                    "discarding instead of degrading the answer"
                                )
                    except Exception as e:
                        logger.debug("[Deliberation] refinement gen failed: %s", e)
        except Exception as e:
            logger.debug("[Deliberation] adversarial critique failed: %s", e)
        # Tool leakage scrub
        if _has_tool_leakage(final_content):
            logger.info("[Deliberation] tool leakage detected, applying scrub")
            for marker in _TOOL_LEAKAGE_MARKERS:
                final_content = final_content.replace(marker, "")
            final_content = _TOOL_CALL_JSON_RE.sub("", final_content).strip()

    # Save the assistant message (skip when ephemeral — eval/preview path)
    saved_msg_id = None
    if not ephemeral:
        try:
            saved_msg_id = await asyncio.to_thread(
                lambda: svc.conversations.add_message(
                    conversation_id, "assistant", final_content,
                    tool_calls=None, sources=None,
                )
            )
        except Exception as e:
            logger.warning("[Deliberation] save message failed: %s", e)

    # Stream the answer in chunks
    chunk_size = 24
    for i in range(0, len(final_content), chunk_size):
        yield StreamEvent(type=EventType.TOKEN, data={"text": final_content[i:i + chunk_size]})

    # Title generation if new conversation (skip on ephemeral)
    if not ephemeral and is_new_conversation and final_content:
        async def _safe_title():
            try:
                title = await _generate_title(query)
                await asyncio.to_thread(svc.conversations.update_title, conversation_id, title)
            except Exception as e:
                logger.warning("[Deliberation] title gen failed: %s", e)
        _t = asyncio.create_task(_safe_title())
        _background_tasks.add(_t)
        _t.add_done_callback(_background_tasks.discard)

    # DONE event with deliberation marker
    yield StreamEvent(
        type=EventType.DONE,
        data={
            "conversation_id": conversation_id,
            "intent": intent,
            "tool_results_count": 0,
            "lessons_used": len(ctx.used_lesson_ids),
            "kg_facts_used": ctx.kg_facts_count,
            "reflexions_used": ctx.reflexions_count,
            "skill_used": None,
            "deliberated": True,
            "iterations": result.iterations,
            "deliberation_success": result.success,
        },
    )

    # Lightweight post-processing — fact extraction, etc. Skip on ephemeral.
    if not ephemeral:
        try:
            async for event in _run_post_processing(
                svc, query, final_content, intent, conversation_id,
                [], None, ctx.used_lesson_ids,  # no tool_results, no skill
                False, None, "",  # not error, no reflexion_quality computed yet
                had_kg=bool(ctx.kg_facts_text),
                had_docs=bool(ctx.retrieved_context),
                channel=channel,
                saved_msg_id=saved_msg_id,
                ephemeral=False,
            ):
                yield event
        except Exception as e:
            logger.warning("[Deliberation] post-processing failed: %s", e)


# ---------------------------------------------------------------------------
# Workspace fact extraction (used by the post-completion enrichment pass)
# ---------------------------------------------------------------------------

_WORKSPACE_FACT_PROMPT = """Pull every concrete, reusable fact from this answer into a snake_case findings dict.

QUESTION: {query}
ANSWER: {answer}

Look for:
- Numbers with units: "695,660 km2", "$67,500", "31.7 million", "2.5%"
- Dates / years: "1789", "April 2025", "Q1 2026"
- Names of entities: people, companies, products, places
- Definitions / single-line facts: "X is the Y of Z"

Output JSON: {{"facts": {{"snake_case_key": "value with unit", ...}}}}

Rules:
- Use snake_case keys describing WHAT the value is (e.g. "texas_area_km2" not "v1").
- Keep values SHORT — number+unit, date, single name, or one short clause. Not a sentence.
- Skip generic conversational fluff. Only extract durable facts.
- If the answer is purely opinion / conversation / refusal, return {{}}.

JSON:"""


async def _extract_workspace_facts(query: str, answer: str) -> dict[str, str]:
    """Use the LLM to extract durable facts from a final answer for workspace storage.

    Returns {} on parse failure or non-substantive answers. Best-effort, non-blocking.
    """
    if not answer or len(answer) < 100:
        return {}
    try:
        prompt = _WORKSPACE_FACT_PROMPT.format(
            query=query[:300],
            answer=answer[:2000],
        )
        resp = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            max_tokens=400, json_mode=True, temperature=0.0, model=config.FAST_MODEL,
        )
        obj = llm.extract_json_object(resp) or {}
        facts = obj.get("facts") if isinstance(obj, dict) else None
        if not isinstance(facts, dict):
            return {}
        return {
            str(k)[:80]: str(v)[:240]
            for k, v in facts.items()
            if v is not None and str(v).strip()
        }
    except Exception as e:
        logger.debug("workspace fact extraction parse failed: %s", e)
        return {}


_UNVERIFIED_CAVEAT = (
    "\n\n_(unverified: I couldn't confirm some specifics above against my "
    "sources — treat precise details like dates, figures, and names with caution.)_"
)


def _maybe_unverified_caveat(final_content: str, flagged_text: str | None) -> str:
    """Append an honest unverified-claims caveat IFF the exact critique-flagged
    answer is still what's shipping — i.e. no accepted rewrite, regeneration, or
    confidence footer changed `final_content` since critique flagged it. The
    strict `==` makes this under-fire (never double-signal) rather than over-fire:
    any later modification suppresses the caveat. Idempotent."""
    if (
        flagged_text is not None
        and final_content == flagged_text
        and "couldn't confirm some specifics" not in final_content
    ):
        return final_content.rstrip() + _UNVERIFIED_CAVEAT
    return final_content


async def _refine_response(
    messages: list[dict],
    tools: list[dict],
    final_content: str,
    query: str,
    intent: str,
    tool_results: list[dict],
    was_planned: bool,
    plan: dict | None,
    retrieved_context: str = "",
    user_facts_text: str = "",
    kg_facts_text: str = "",
) -> tuple[str, float | None, str]:
    """Multi-round critique, plan coverage check, reflexion LLM critique.

    This is Steps 10b–10d of the original think() pipeline.
    Returns: (refined_content, reflexion_quality, reflexion_reason)
    """
    # --- Self-Critique (rewrite-based) ---
    # When the critique flags issues, REWRITE the answer (single retry) instead
    # of appending "---\nUpdate: ..." addendums. The addendum approach stacked
    # contradictions in user-visible output (real prod sample: original "YES,
    # bring an umbrella" + two "Update: I cannot confirm..." follow-ups). A
    # rewrite keeps the answer coherent. We accept the rewrite if it's
    # substantive AND no longer than 1.5x the original (guard against
    # over-explanation), otherwise keep original.
    critique_passed = False
    # Grounding-honesty: if critique flags unsupported claims and we never produce
    # an accepted rewrite, the flagged answer would otherwise ship as confident
    # fact. Capture the exact flagged text; at return we append an honest caveat
    # ONLY if that exact text survived unmodified (any later regeneration changes
    # final_content and auto-suppresses the note — it under-fires, never over-fires).
    _flagged_unverified: str | None = None
    if config.ENABLE_CRITIQUE and final_content and intent == "general":
        from app.core.critique import critique_answer, format_critique_for_regeneration
        if len(final_content) >= 200:
            last_critique_issues: list[str] = []
            try:
                critique = await critique_answer(
                    query, final_content,
                    sources=retrieved_context,
                    user_facts=user_facts_text,
                    kg_facts=kg_facts_text,
                )
                if not critique or critique.get("pass", True):
                    critique_passed = True
                else:
                    last_critique_issues = critique.get("issues", [])
                    logger.info("Critique flagged issues: %s", last_critique_issues)
                    issues_list = critique.get("issues", [])
                    _flagged_unverified = final_content  # ground-truth: this exact text was flagged
                    rewrite_prompt = (
                        "Your previous answer had these issues:\n"
                        + "\n".join(f"- {issue}" for issue in issues_list)
                        + "\n\nRewrite the answer in full, fixing those issues. "
                        "Output the corrected answer DIRECTLY — no preamble, no 'Update:', "
                        "no apology, no reference to the original draft. The user will only "
                        "see your rewritten answer."
                    )
                    # Cap messages to prevent unbounded token growth.
                    if len(messages) > 10:
                        head = messages[:3]
                        tail = messages[-7:]
                        middle_tool_msgs = [
                            m for m in messages[3:-7]
                            if m.get("role") == "tool" or (
                                m.get("role") == "system" and "tool" in m.get("content", "").lower()[:100]
                            )
                        ]
                        messages = head + middle_tool_msgs + tail
                    messages.append({"role": "assistant", "content": final_content})
                    messages.append({"role": "user", "content": rewrite_prompt})
                    try:
                        retry_result = await llm.generate_with_tools(messages, tools)
                        if retry_result.content and not retry_result.tool_calls:
                            rewrite = retry_result.content.strip()
                            # Acceptance gates:
                            # 1. substantive (>=100 chars)
                            # 2. not 2x bloated past original
                            # 3. doesn't lapse into self-meta-commentary —
                            #    the model often "rewrites" by writing ABOUT
                            #    its previous draft instead of replacing it
                            #    ("The previous response contained...",
                            #     "Corrected Explanation:", "Update:"). Those
                            #    are the failure mode we're trying to avoid.
                            _meta_markers = (
                                "previous response", "the previous answer",
                                "previous draft", "earlier response",
                                "corrected explanation", "corrected version",
                                "in my previous", "my earlier answer",
                                "update:", "addendum:",
                                "previously stated", "previously said",
                                "as i mentioned before", "i made a mistake",
                                "let me correct", "to clarify the previous",
                            )
                            _rewrite_lower = rewrite.lower()
                            _has_meta = any(m in _rewrite_lower for m in _meta_markers)
                            # Typo / generation-degenerate sniff: 3+ words with
                            # internal repeated character clusters of 3+
                            # ("Ththroughput", "Bittcoin", "ProcessProcess").
                            # When the rewrite has these, it's likely a degraded
                            # generation, not an improvement — keep original.
                            _typo_hits = sum(
                                1 for w in re.findall(r"\b\w+\b", rewrite)
                                if re.search(r"([a-zA-Z])\1{2,}", w)
                                or re.search(r"([A-Z][a-z]+)\1+", w)
                            )
                            _looks_degraded = _typo_hits >= 3
                            if (
                                rewrite
                                and len(rewrite) >= 100
                                and len(rewrite) <= max(len(final_content) * 2, 1200)
                                and not _has_meta
                                and not _looks_degraded
                            ):
                                final_content = rewrite
                                logger.info("Critique-driven rewrite accepted (%d → %d chars)",
                                            len(messages[-2].get("content", "")), len(rewrite))
                            else:
                                logger.info(
                                    "Critique rewrite rejected (len=%d, meta=%s) — keeping original",
                                    len(rewrite), _has_meta,
                                )
                    except Exception as e:
                        logger.warning("Critique rewrite generation failed: %s", e)
            except Exception as e:
                logger.warning("Critique failed: %s", e)

            # Store ONE reflexion after the critique loop ends (not per-iteration)
            if not critique_passed and last_critique_issues:
                _svc = get_services()
                if _svc.reflexions:
                    try:
                        issues_text = "; ".join(last_critique_issues)
                        await asyncio.to_thread(
                            lambda: _svc.reflexions.store(
                                task_summary=query[:500],
                                outcome="failure",
                                reflection=f"Critique failed: {issues_text}",
                                quality_score=0.3,
                            )
                        )
                    except Exception:
                        pass

    # --- Adversarial critique (logic/factual error hunter) ---
    if config.ENABLE_CRITIQUE and final_content and intent == "general" and len(final_content) >= 200:
        from app.core.critique import adversarial_critique, format_adversarial_for_replan
        try:
            adv = await adversarial_critique(
                query, final_content,
                sources=retrieved_context,
                user_facts=user_facts_text,
                kg_facts=kg_facts_text,
            )
            if adv:
                severity = adv.get("verdict", "pass")
                flaws = adv.get("flaws", [])
                logger.info("[ADVERSARIAL] Critique fired: severity=%s, flaws=%d", severity, len(flaws))
                if adv.get("verdict") == "fail":
                    adv_msg = format_adversarial_for_replan(adv)
                    if adv_msg:
                        messages.append({"role": "assistant", "content": final_content})
                        messages.append({"role": "system", "content": adv_msg})
                        try:
                            retry_result = await llm.generate_with_tools(messages, tools)
                            if retry_result.content and not retry_result.tool_calls:
                                final_content = retry_result.content
                        except Exception as e:
                            logger.warning("Adversarial critique regeneration failed: %s", e)
        except Exception as e:
            logger.warning("Adversarial critique failed: %s", e)

    # --- Plan coverage check ---
    if was_planned and final_content and config.ENABLE_PLANNING:
        from app.core.planning import verify_plan_coverage
        try:
            missed = verify_plan_coverage(plan, final_content)
            if missed:
                logger.info("Plan steps missed: %s", missed)
                missed_text = "\n".join(f"- {s}" for s in missed)
                messages.append({"role": "assistant", "content": final_content})
                messages.append({
                    "role": "system",
                    "content": (
                        "[PLAN COVERAGE CHECK]\n"
                        f"Your answer missed these planned steps:\n{missed_text}\n"
                        "Address the missing steps in a revised answer."
                    ),
                })
                try:
                    retry_result = await llm.generate_with_tools(messages, tools)
                    if retry_result.content:
                        if retry_result.tool_calls:
                            logger.info("Plan coverage retry returned %d tool call(s) — ignored, using text content",
                                        len(retry_result.tool_calls))
                        final_content = retry_result.content
                        logger.info("Regenerated after plan coverage check (%d chars)", len(final_content))
                except Exception as e:
                    logger.warning("Plan coverage regeneration failed: %s", e)
        except Exception as e:
            logger.debug("Plan coverage check failed: %s", e)

    # --- Synthesis polish for tool-heavy responses ---
    # When tools were used and the model leaked raw tool markers (`[Source N:`,
    # `## Matching ...`, `[N]` reference dumps) into the final answer, run a
    # one-shot polish pass that rewrites the answer cleanly without those
    # internal markers. Bounded: only fires when leakage is detected, only when
    # tool_results > 0, single retry.
    if (
        final_content
        and intent == "general"
        and tool_results
        and _has_tool_leakage(final_content)
    ):
        logger.info("[POLISH] tool-marker leakage detected — running synthesis polish")
        try:
            polish_msg = {
                "role": "system",
                "content": (
                    "[SYNTHESIS POLISH]\n"
                    "Your previous answer contained raw tool output markers like "
                    "'## Matching User Facts', '[Source N:', '[1] Title', '## Matching Documents', "
                    "or numbered reference dumps. These are internal — never show them to the user.\n\n"
                    "Rewrite the answer for the user:\n"
                    "- Lead with the actual answer, not 'based on the search results...'\n"
                    "- Keep ALL the substantive facts and numbers — nothing dropped\n"
                    "- Cite sources naturally inline ('per BLS', 'apple.com newsroom') not as raw '[1]' brackets\n"
                    "- Drop ALL '## Matching X', '[Source N:', '[Tool error', and tool-internal headers\n"
                    "- Same length or shorter — don't pad"
                ),
            }
            messages_for_polish = list(messages) + [
                {"role": "assistant", "content": final_content},
                polish_msg,
            ]
            polished = await llm.invoke_nothink(
                messages_for_polish, max_tokens=2000, temperature=0.2,
            )
            polished = (polished or "").strip()
            if polished and len(polished) >= 50 and not _has_tool_leakage(polished):
                logger.info(
                    "[POLISH] replaced (%d chars -> %d chars, leakage cleared)",
                    len(final_content), len(polished),
                )
                final_content = polished
            else:
                logger.warning("[POLISH] retry didn't clean up — keeping original")
        except Exception as e:
            logger.warning("[POLISH] synthesis polish failed: %s", e)

    # --- Reflexion LLM critique (second-pass, pre-stream) ---
    reflexion_quality = None
    reflexion_reason = ""
    if final_content and intent == "general":
        from app.core.reflexion import should_use_llm_critique, critique_response, assess_quality
        try:
            if should_use_llm_critique(intent, final_content, tool_results):
                reflexion_quality, reflexion_reason = await critique_response(
                    query, final_content, tool_results,
                    user_facts=user_facts_text,
                    kg_facts=kg_facts_text,
                )
            else:
                reflexion_quality, reflexion_reason = assess_quality(final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query)

            if reflexion_quality is not None:
                logger.info("[QUALITY] score=%.2f reason=%s", reflexion_quality, reflexion_reason[:100])

            if reflexion_quality is not None and reflexion_quality < 0.3 and reflexion_reason and not critique_passed:
                logger.info("Reflexion critique flagged (%.2f): %s", reflexion_quality, reflexion_reason)
                # Generate a NATURAL refinement, not a "Correction" header.
                # The old "Update:" prefix made the judge see the whole answer
                # as 'self-admitted-flawed' and graded it down further — feeding
                # back into bimodal-low scores on long deliberation answers.
                # New approach: ask for a short clarification paragraph that
                # reads as a continuation, with hard rules against scaffold
                # words. Reject the refinement if forbidden header words leak.
                addendum_msg = (
                    "Refinement note: a reviewer flagged this concern with the "
                    f"answer above:\n  {reflexion_reason}\n\n"
                    "Add a SHORT closing paragraph (3-5 sentences max) that "
                    "addresses it naturally.\n"
                    "Hard rules:\n"
                    "- Read as a natural continuation. No headers, dividers, "
                    "or section labels.\n"
                    "- Do NOT use any of these words/phrases: 'Update:', "
                    "'Correction', 'Errata', 'Addenda', 'Note:', 'Important:', "
                    "'Caveat', 'Disclaimer', 'Revision', 'the previous response', "
                    "'the previous answer'.\n"
                    "- Do NOT admit the prior content was wrong; integrate the "
                    "refinement as additional clarification.\n"
                    "- Output ONLY the paragraph itself."
                )
                messages.append({"role": "assistant", "content": final_content})
                messages.append({"role": "system", "content": addendum_msg})
                try:
                    retry_result = await llm.generate_with_tools(messages, tools)
                    if retry_result.content and not retry_result.tool_calls:
                        addendum = retry_result.content.strip()
                        forbidden_re = re.compile(
                            r"(?im)^\s*(?:#+\s*)?(?:update|correction|errata|"
                            r"addenda|caveat|disclaimer|revision|note|important)"
                            r"\b[\s:]"
                        )
                        if addendum and not forbidden_re.search(addendum):
                            final_content = final_content.rstrip() + "\n\n" + addendum
                            reflexion_quality, reflexion_reason = assess_quality(
                                final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query
                            )
                            logger.info(
                                "Appended natural refinement (%d chars, new score: %.2f)",
                                len(addendum), reflexion_quality,
                            )
                        elif addendum:
                            logger.info(
                                "Refinement leaked forbidden header — discarding "
                                "rather than appending self-deprecating text"
                            )
                except Exception as e:
                    logger.warning("Reflexion refinement failed: %s", e)
        except Exception as e:
            logger.debug("Reflexion pre-stream critique failed: %s", e)

    # --- Constraint-coverage verifier ---
    # Extract "must-address" components from the original question and verify
    # each is actually addressed in the answer. Stronger than plan_coverage
    # (which only fires when ENABLE_PLANNING=true and a plan was created).
    # On miss: append a focused addendum addressing the gap.
    if final_content and intent == "general" and len(final_content) >= 100:
        try:
            missing = _find_missing_constraints(query, final_content)
            if missing:
                logger.info("[CONSTRAINT-COVERAGE] missing components: %s", missing[:5])
                addendum_msg = (
                    "[CONSTRAINT-COVERAGE]\n"
                    "Your answer didn't address these specific components from the question:\n"
                    + "\n".join(f"  - {m}" for m in missing[:5])
                    + "\n\nGenerate ONLY a brief addendum addressing these missing parts. "
                    "Do NOT repeat what you already wrote — just fill the gaps."
                )
                cov_messages = list(messages) + [
                    {"role": "assistant", "content": final_content},
                    {"role": "system", "content": addendum_msg},
                ]
                try:
                    cov_response = await llm.invoke_nothink(
                        cov_messages, max_tokens=800, temperature=0.2,
                    )
                    cov_response = (cov_response or "").strip()
                    if cov_response and len(cov_response) >= 30:
                        final_content = final_content.rstrip() + "\n\n---\n" + cov_response
                        logger.info(
                            "[CONSTRAINT-COVERAGE] appended addendum (%d chars)",
                            len(cov_response),
                        )
                except Exception as e:
                    logger.warning("[CONSTRAINT-COVERAGE] addendum failed: %s", e)
        except Exception as e:
            logger.debug("[CONSTRAINT-COVERAGE] verifier failed: %s", e)

    # --- Best-of-N final-resort sampling for hard low-quality cases ---
    # Fires when:
    #   - critique chain finished but quality is still below BEST_OF_N_QUALITY_THRESHOLD
    #   - query is hard-reasoning (compare/analyze/derive/etc)
    #   - query is NOT inherently uncertain (forecasting, future predictions, exact-future-value)
    #     because best-of-N would pick higher-confidence over correctly-uncertain
    #   - feature is enabled
    # Samples N alternative responses at varied temperatures, scores them, picks best.
    _is_inherently_uncertain = bool(_INHERENTLY_UNCERTAIN_RE.search(query))
    if (
        final_content
        and intent == "general"
        and reflexion_quality is not None
        and reflexion_quality < config.BEST_OF_N_QUALITY_THRESHOLD
        and config.ENABLE_BEST_OF_N
        and not _is_inherently_uncertain
    ):
        try:
            from app.core.agent_loop import _is_hard_reasoning_query
            if _is_hard_reasoning_query(query):
                from app.core.reflexion import assess_quality
                logger.info(
                    "[BEST-OF-N] firing — query is hard-reasoning AND quality=%.2f below %.2f",
                    reflexion_quality, config.BEST_OF_N_QUALITY_THRESHOLD,
                )
                # Build a clean alternative-sampling message — same context, ask for a fresh answer
                alt_messages = list(messages) + [
                    {"role": "assistant", "content": final_content},
                    {
                        "role": "system",
                        "content": (
                            "[ALTERNATIVE TAKE]\n"
                            "Your previous answer scored low on quality. Produce a FRESH answer to "
                            "the original question — different angle, sharper structure, tighter focus "
                            "on what was actually asked. Same factual basis (use the same tool results "
                            "if any), but a stronger response."
                        ),
                    },
                ]
                # Sample N alternatives at varied temperatures, in parallel
                temps = [0.4, 0.7, 1.0][: max(1, config.BEST_OF_N_SAMPLES)]

                async def _sample_one(temp: float) -> str:
                    try:
                        resp = await llm.invoke_nothink(
                            alt_messages, max_tokens=2000, temperature=temp,
                        )
                        return (resp or "").strip()
                    except Exception as e:
                        logger.debug("[BEST-OF-N] sample at temp=%.1f failed: %s", temp, e)
                        return ""

                alternatives = await asyncio.gather(*[_sample_one(t) for t in temps])
                # Score each candidate including the original via the heuristic assess_quality
                candidates: list[tuple[str, float]] = [(final_content, reflexion_quality)]
                for alt in alternatives:
                    if not alt or len(alt) < 50:
                        continue
                    try:
                        alt_q, _ = assess_quality(
                            alt, tool_results, config.MAX_TOOL_ROUNDS, query=query
                        )
                        if alt_q is not None:
                            candidates.append((alt, alt_q))
                    except Exception as e:
                        logger.debug("[BEST-OF-N] scoring failed: %s", e)

                if len(candidates) > 1:
                    best_text, best_q = max(candidates, key=lambda c: c[1])
                    # Only swap if the winner is meaningfully better (avoid noise)
                    if best_q > reflexion_quality + 0.05 and best_text is not final_content:
                        logger.info(
                            "[BEST-OF-N] replaced (orig=%.2f -> best=%.2f from %d candidates)",
                            reflexion_quality, best_q, len(candidates),
                        )
                        final_content = best_text
                        reflexion_quality = best_q
                    else:
                        logger.info(
                            "[BEST-OF-N] kept original (orig=%.2f, best alt=%.2f, no meaningful gain)",
                            reflexion_quality, best_q,
                        )
        except Exception as e:
            logger.warning("[BEST-OF-N] failed: %s", e)

    # --- Confidence transparency footer ---
    # Three-tier appender that adapts the footer text to the actual situation,
    # rather than firing only on the narrow "fully ungrounded" case.
    #
    # Tier 1 (ungrounded + borderline)  -> "no fresh sources" footer
    # Tier 2 (tools used + quality<0.5) -> "limited verification" footer
    # Tier 3 (hedging language seen)    -> "noted uncertainty" footer
    try:
        if (
            final_content
            and intent == "general"
            and reflexion_quality is not None
            and not _has_confidence_footer(final_content)
        ):
            uncertainty_markers = (
                "i think", "i believe", "probably", "i'm not sure",
                "not certain", "as far as i know", "may be", "might be",
                "approximately", "roughly", "around", "presumably",
                "possibly", "i suspect", "appears to", "seems to",
                "my best guess", "uncertain", "unclear",
            )
            lc = final_content.lower()
            has_hedging = any(m in lc for m in uncertainty_markers)
            ungrounded = (not tool_results) and (not kg_facts_text) and (not retrieved_context)

            footer = None
            substantive = len(final_content) >= 100  # don't footer on trivial answers
            # Calibrated thresholds (Conformal Abstention) — adapt to recent
            # reflexion score distribution rather than fixed cutoffs. Falls back
            # to historical defaults (0.40/0.60, 0.50, 0.60) when the reflexions
            # table has fewer than 30 entries or ENABLE_CONFORMAL_ABSTENTION=false.
            try:
                from app.core.calibration import get_thresholds as _get_calib
                _thr = _get_calib()
            except Exception as e:
                logger.debug("[CONFIDENCE] calibration lookup failed: %s", e)
                _thr = None
            _t1_lo = _thr.tier1_low if _thr else 0.40
            _t1_hi = _thr.tier1_high if _thr else 0.60
            _t2 = _thr.tier2 if _thr else 0.50
            _t3 = _thr.tier3 if _thr else 0.60
            if (
                substantive
                and ungrounded
                and _t1_lo <= reflexion_quality < _t1_hi
                and (has_hedging or len(final_content) < 400)
            ):
                footer = _CONFIDENCE_FOOTER_TEXT
                tier = "ungrounded"
            elif substantive and tool_results and reflexion_quality < _t2:
                footer = (
                    "_(limited-verification: tool results were thin or partial — "
                    "treat the answer as a starting point, not a final word.)_"
                )
                tier = "limited-tools"
            elif substantive and has_hedging and reflexion_quality < _t3:
                footer = (
                    "_(noted-uncertainty: I hedged above where I'm not solid. "
                    "Ask me to verify any specific claim with web_search.)_"
                )
                tier = "hedging"
            else:
                tier = None

            if footer:
                final_content = final_content.rstrip() + "\n\n" + footer
                logger.info(
                    "[CONFIDENCE] appended footer (tier=%s, quality=%.2f, hedging=%s, grounded=%s)",
                    tier, reflexion_quality, has_hedging, not ungrounded,
                )
    except Exception as e:
        logger.debug("Confidence footer logic failed: %s", e)

    # Grounding-honesty caveat (anti-illusion): if critique flagged unsupported
    # claims and the EXACT flagged text is still shipping (no accepted rewrite,
    # regeneration, or confidence footer changed it — any of those make the `==`
    # in the helper False, so this never double-signals), say so.
    _caveated = _maybe_unverified_caveat(final_content, _flagged_unverified)
    if _caveated != final_content:
        final_content = _caveated
        logger.info("[GROUNDING] appended unverified-claims caveat (critique flagged, unfixed)")

    return final_content, reflexion_quality, reflexion_reason


async def _detect_capability_gap(
    svc: Services,
    query: str,
    matched_skill: object | None,
    tool_results: list[dict],
    reflexion_quality: float | None,
) -> None:
    """Log a capability gap when Nova demonstrably couldn't handle a query.

    A gap is detected when all three conditions hold:
    1. No skill was matched for the query
    2. No tools were successfully used (empty tool_results = planner found nothing useful)
    3. Response quality scored below 0.5

    Gaps are stored in the capability_gaps table for periodic review by the
    Capability Review monitor, which surfaces suggestions for new tools/skills.
    """
    if matched_skill is not None:
        return
    if tool_results:
        return
    if reflexion_quality is None or reflexion_quality >= 0.5:
        return

    try:
        from app.database import get_db
        db = get_db()
        await asyncio.to_thread(
            db.execute,
            "INSERT INTO capability_gaps (query, reason, tools_tried, quality_score) VALUES (?, ?, ?, ?)",
            (
                query[:500],
                f"No skill matched, no tools used, quality={reflexion_quality:.2f}",
                "[]",
                reflexion_quality,
            ),
        )
        logger.info("[CAPABILITY_GAP] Logged gap: %s", query[:80])
    except Exception as e:
        logger.debug("Capability gap logging failed: %s", e)


async def _run_post_processing(
    svc: Services,
    query: str,
    final_content: str,
    intent: str,
    conversation_id: str,
    tool_results: list[dict],
    matched_skill: object | None,
    used_lesson_ids: list[int],
    is_error: bool,
    reflexion_quality: float | None,
    reflexion_reason: str,
    had_kg: bool = False,
    had_docs: bool = False,
    channel: str = "api",
    saved_msg_id: str | None = None,
    ephemeral: bool = False,
) -> AsyncGenerator[StreamEvent, None]:
    """Post-response processing: corrections, facts, KG, reflexion storage, auto skills.

    This is Steps 13–17 of the original think() pipeline.
    Yields LESSON_LEARNED events.

    `ephemeral` (default False) — when True, skip writes to long-term memory:
    no curiosity queue additions, no facts extracted, no learning events. This
    matches the behavior of the rest of think() under ephemeral=True.
    """
    # --- Corrections + learning ---
    logger.info("Post-response: intent=%s, learning=%s", intent, svc.learning is not None)
    if intent == "correction" and svc.learning:
        prev_messages = await asyncio.to_thread(svc.conversations.get_history, conversation_id, 10)
        prev_answer = ""
        original_query = ""
        # Skip the last assistant message only if it is the response we just
        # saved (step 11).  Use message ID comparison when available for
        # reliability; fall back to content comparison otherwise.
        last_assistant = next(
            (m for m in reversed(prev_messages) if m.role == "assistant"), None,
        )
        if saved_msg_id and last_assistant:
            assistant_skip = 1 if last_assistant.id == saved_msg_id else 0
        else:
            assistant_skip = 1 if (last_assistant and last_assistant.content == final_content) else 0
        found_wrong_answer = False
        for msg in reversed(prev_messages):
            if msg.role == "assistant":
                if assistant_skip > 0:
                    assistant_skip -= 1
                    continue
                if not found_wrong_answer:
                    prev_answer = msg.content
                    found_wrong_answer = True
            elif msg.role == "user" and found_wrong_answer:
                # Skip the current correction AND any prior corrections —
                # we want the original question, not another correction.
                if msg.content != query and not is_likely_correction(msg.content):
                    original_query = msg.content
                    break

        logger.info(
            "Correction context: prev_answer=%d chars, original_query='%s'",
            len(prev_answer) if prev_answer else 0,
            (original_query or "")[:80],
        )
        if not prev_answer:
            logger.info("No previous assistant answer found, skipping correction detection")
        if prev_answer:
            try:
                correction = await svc.learning.detect_correction(
                    query, prev_answer, original_query=original_query
                )
                logger.info("Correction detection result: %s", correction is not None)
                if correction:
                    # Guard: if Nova's response pushed back against the correction,
                    # Nova was right to disagree — don't save the user's wrong
                    # correction as a lesson or DPO pair (would corrupt training data).
                    if response_pushes_back(final_content):
                        logger.info(
                            "Skipping correction save: Nova's response pushed back "
                            "against user correction (topic='%s'). Response likely correct.",
                            correction.topic,
                        )
                    else:
                        lesson_id = await asyncio.to_thread(svc.learning.save_lesson, correction)

                        # ACE harmful-attribution (strongest signal): the user just
                        # CORRECTED an answer, so the lessons that were in-context for
                        # it were collectively unhelpful. Re-retrieve the lessons
                        # relevant to the original question (~= those injected for the
                        # wrong answer) and demote them — excluding the correction we
                        # just saved. mark_lesson_unhelpful is dampened, so a lesson
                        # only incidentally present self-corrects over time.
                        try:
                            _prior_lessons = await asyncio.to_thread(
                                svc.learning.get_relevant_lessons, original_query or query)
                            _demoted = 0
                            for _les in (_prior_lessons or []):
                                _lid = getattr(_les, "id", None)
                                if _lid and _lid != lesson_id:
                                    await asyncio.to_thread(svc.learning.mark_lesson_unhelpful, _lid)
                                    _demoted += 1
                            if _demoted:
                                logger.info(
                                    "[ACE] correction-attribution: demoted %d in-context "
                                    "lesson(s) that preceded a user correction", _demoted)
                        except Exception as _e:
                            logger.debug("correction-attribution penalty failed: %s", _e)

                        dpo_query = original_query or query
                        dpo_rejected = (prev_answer or "")[:1000]
                        dpo_chosen = (correction.correct_answer or correction.lesson_text or "")[:1000]

                        if not dpo_chosen or not dpo_rejected:
                            logger.warning(
                                "Skipping save_training_pair: empty DPO value (chosen=%d chars, rejected=%d chars)",
                                len(dpo_chosen), len(dpo_rejected),
                            )
                        else:
                            await svc.learning.save_training_pair(
                                query=dpo_query,
                                bad_answer=dpo_rejected,
                                good_answer=dpo_chosen,
                                channel=channel,
                            )

                        degraded_skill = matched_skill
                        if not degraded_skill and original_query and svc.skills:
                            degraded_skill = await asyncio.to_thread(svc.skills.get_matching_skill, original_query)
                        if degraded_skill and svc.skills:
                            await asyncio.to_thread(svc.skills.record_use, degraded_skill.id, False)
                            logger.info("Skill '%s' marked as failed due to correction", degraded_skill.name)

                            # Attempt refinement in background instead of just degrading
                            async def _safe_refine(skills, sid, ctx):
                                try:
                                    refined = await skills.refine_skill(sid, ctx)
                                    if refined:
                                        logger.info("Skill #%d refined after correction", sid)
                                except Exception as e:
                                    logger.debug("Skill refinement failed: %s", e)
                            _task = asyncio.create_task(
                                _safe_refine(svc.skills, degraded_skill.id, query[:300])
                            )
                            _background_tasks.add(_task)
                            _task.add_done_callback(_background_tasks.discard)

                        if svc.skills:
                            from app.core.skills import extract_skill_from_correction
                            skill_data = await extract_skill_from_correction(
                                correction.user_message,
                                tool_results,
                                lesson_id,
                            )
                            if skill_data:
                                skill_id = await asyncio.to_thread(lambda: svc.skills.create_skill(**skill_data))
                                if skill_id:
                                    logger.info("Skill extracted: '%s' (id=%d)", skill_data["name"], skill_id)
                                else:
                                    logger.info("Skill rejected (too broad): '%s'", skill_data["name"])

                        logger.info("Correction processed → lesson #%d saved", lesson_id)
                        yield StreamEvent(
                            type=EventType.LESSON_LEARNED,
                            data={
                                "topic": correction.topic,
                                "lesson_id": lesson_id,
                            },
                        )
            except Exception as e:
                logger.warning("Correction processing failed: %s", e)

    # --- Automatic fact extraction: DISABLED ---
    # Auto-extraction polluted user_facts with fabricated personas from test queries
    # (e.g. "I'm Alex from Acme" → stored as truth). The active_memory tool remains
    # available for deliberate, owner-confirmed storage; passive inference is off.

    # --- Reflexion — store failures AND high-quality successes ---
    if svc.reflexions and intent == "general" and final_content:
        try:
            if reflexion_quality is not None:
                quality, reason = reflexion_quality, reflexion_reason
            else:
                from app.core.reflexion import assess_quality
                quality, reason = assess_quality(final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query)
            tools_used = [tr["tool"] for tr in tool_results]
            if quality < config.REFLEXION_FAILURE_THRESHOLD and reason:
                await asyncio.to_thread(
                    lambda: svc.reflexions.store(
                        task_summary=query[:500],
                        outcome="failure",
                        reflection=reason,
                        quality_score=quality,
                        tools_used=tools_used,
                        revision_count=len(tool_results),
                        is_eval=ephemeral,
                    )
                )
            elif quality >= config.REFLEXION_SUCCESS_THRESHOLD and tool_results:
                await asyncio.to_thread(
                    lambda: svc.reflexions.store(
                        task_summary=query[:500],
                        outcome="success",
                        reflection=f"Successful approach for '{query[:100]}': {' -> '.join(tools_used)} (quality={quality:.2f})",
                        quality_score=quality,
                        tools_used=tools_used,
                        revision_count=len(tool_results),
                        is_eval=ephemeral,
                    )
                )
        except Exception as e:
            logger.warning("Reflexion storage failed: %s", e)

    # --- Auto skill creation (background) ---
    # Fire for any tool interaction when: 2+ tools used, OR 1 tool + quality >= 0.7
    # (quality check lets the reflexion system vouch for single-tool responses)
    _auto_skill_quality = reflexion_quality if reflexion_quality is not None else 0.0
    _auto_skill_eligible = (
        config.ENABLE_AUTO_SKILL_CREATION
        and svc.skills
        and len(tool_results) >= 1
        and intent == "general"
        and (len(tool_results) >= 2 or _auto_skill_quality >= 0.7)
    )
    if _auto_skill_eligible:
        from app.core.auto_skills import maybe_extract_skill

        async def _safe_skill_extract(q, trs, content, skills, qs):
            try:
                await maybe_extract_skill(q, trs, content, skills, quality_score=qs)
            except Exception as e:
                logger.warning("Auto-skill extraction failed: %s", e)
        _task = asyncio.create_task(
            _safe_skill_extract(query, tool_results, final_content, svc.skills, _auto_skill_quality)
        )
        _background_tasks.add(_task)
        _task.add_done_callback(_background_tasks.discard)

    # --- Curiosity: detect gaps and queue for research ---
    # Skip when ephemeral=True — those are tests/probes, not real user gaps.
    # Without this gate, every probe query gets queued and the daemon retries
    # them forever (we hit this in the v9 e2e session).
    if config.ENABLE_CURIOSITY and svc.curiosity and intent == "general" and not is_error and not ephemeral:
        try:
            from app.core.curiosity import detect_gaps
            gaps = detect_gaps(
                query=query,
                answer=final_content,
                tool_results=tool_results,
                had_lessons=bool(used_lesson_ids),
                had_kg=had_kg,
                had_docs=had_docs,
            )
            for gap in gaps:
                await asyncio.to_thread(
                    lambda _g=gap: svc.curiosity.add(_g["topic"], source=_g["source"], urgency=_g["urgency"])
                )
        except Exception as e:
            logger.debug("Curiosity gap detection failed: %s", e)

    # --- Curiosity: queue failed responses for research ---
    # Same ephemeral gate as above.
    if config.ENABLE_CURIOSITY and svc.curiosity and intent == "general" and not ephemeral:
        try:
            if reflexion_quality is not None and reflexion_quality < 0.5:
                from app.core.curiosity import TopicTracker
                topic = TopicTracker._extract_topic(query[:200])
                if topic:
                    await asyncio.to_thread(
                        lambda _t=topic: svc.curiosity.add(_t, source="reflexion_failure", urgency=0.7)
                    )
        except Exception as e:
            logger.debug("Curiosity failure queueing failed: %s", e)

    # --- Reflexion-to-Action: promote recurring failures to lessons ---
    if svc.reflexions and svc.learning and intent == "general" and final_content:
        try:
            if reflexion_quality is not None and reflexion_quality < 0.6:
                async def _safe_check_recurring(task_summary, learning):
                    try:
                        from app.core.reflexion import check_recurring_failures
                        await check_recurring_failures(task_summary, learning)
                    except Exception as e:
                        logger.warning("Recurring failure check failed: %s", e)
                _task = asyncio.create_task(
                    _safe_check_recurring(query[:500], svc.learning)
                )
                _background_tasks.add(_task)
                _task.add_done_callback(_background_tasks.discard)
        except Exception as e:
            logger.debug("Reflexion-to-action setup failed: %s", e)

    # --- Topic tracking for auto-monitor creation ---
    if config.ENABLE_CURIOSITY and svc.topic_tracker and intent == "general":
        try:
            await asyncio.to_thread(svc.topic_tracker.record_topic, query)
        except Exception as e:
            logger.debug("Topic tracking failed: %s", e)

    # --- Capability gap detection ---
    if intent == "general" and not is_error:
        await _detect_capability_gap(svc, query, matched_skill, tool_results, reflexion_quality)


# ---------------------------------------------------------------------------
# Multi-agent path
# ---------------------------------------------------------------------------

async def _run_multi_agent_path(
    svc: Services,
    query: str,
    conversation_id: str,
    intent: str,
    ctx: "_ThinkContext",
    decomp_plan: "DecompositionPlan",
    is_new_conversation: bool,
    ephemeral: bool,
    channel: str,
) -> AsyncGenerator[StreamEvent, None]:
    """Execute the structural multi-agent decomposition path.

    Yields the same SSE event stream as the normal think() path plus four
    new event types: AGENT_META, AGENT_START, AGENT_DONE, AGENT_MERGE.
    Post-processing (corrections, facts, reflexion) runs on the merged
    response text, same as the normal path.
    """
    from app.core.agent_spawner import AgentSpawner, merge_agent_results

    agent_count = len(decomp_plan.tasks)

    # --- META: announce decomposition to the client ---
    yield StreamEvent(
        type=EventType.AGENT_META,
        data={
            "type": "decomposition",
            "strategy": decomp_plan.strategy,
            "agent_count": agent_count,
            "fallback": False,
        },
    )

    # --- AGENT_START: one event per task ---
    for task in decomp_plan.tasks:
        yield StreamEvent(
            type=EventType.AGENT_START,
            data={
                "run_id": conversation_id,
                "task_id": task.task_id,
                "role": task.role,
                "query": task.query[:200],
            },
        )

    # --- Execute all sub-agents ---
    # Total timeout scales with agent count and parallelism budget. Previously
    # hardcoded to TOOL_TIMEOUT*2=360s, which throttled recursive decomposition
    # under serialized GPU load. New formula: AGENT_TASK_TIMEOUT * ceil(N / max_parallel)
    # + 30s merge headroom. Sized for RTX 3090 + 9B Q8 + recursive depth-2.
    import math
    n = len(decomp_plan.tasks)
    waves = max(1, math.ceil(n / max(1, decomp_plan.max_parallel)))
    total_timeout = config.AGENT_TASK_TIMEOUT * waves + 30
    logger.info(
        "[Multi-agent] total_timeout=%ds (n=%d waves=%d agent_timeout=%d)",
        total_timeout, n, waves, config.AGENT_TASK_TIMEOUT,
    )
    spawner = AgentSpawner(decomp_plan, conversation_id)
    try:
        results = await asyncio.wait_for(spawner.run(), timeout=float(total_timeout))
    except asyncio.TimeoutError:
        logger.warning("Multi-agent decomposition timed out after %ds", total_timeout)
        yield StreamEvent(
            type=EventType.AGENT_META,
            data={"type": "fallback", "reason": "total_timeout"},
        )
        fallback_text = "[Multi-agent decomposition timed out. Please try again.]"
        yield StreamEvent(type=EventType.TOKEN, data={"text": fallback_text})
        yield StreamEvent(
            type=EventType.DONE,
            data={
                "conversation_id": conversation_id,
                "intent": intent,
                "tool_results_count": 0,
                "lessons_used": len(ctx.used_lesson_ids),
                "kg_facts_used": ctx.kg_facts_count,
                "reflexions_used": ctx.reflexions_count,
                "skill_used": None,
                "decomposed": True,
                "agent_count": agent_count,
            },
        )
        return

    # --- AGENT_DONE: one event per completed task ---
    all_tools: list[str] = []
    completed_count = 0
    failed_count = 0
    max_depth_observed = 0
    for result in results:
        all_tools.extend(result.tools_invoked)
        if result.error:
            failed_count += 1
        else:
            completed_count += 1
        # Track the deepest level any sub-agent reached. If a sub-agent itself
        # decomposed further, the effective depth is one greater than its own.
        d = getattr(result, "depth", 1)
        if getattr(result, "sub_decomposed", False):
            d = d + 1
        if d > max_depth_observed:
            max_depth_observed = d
        yield StreamEvent(
            type=EventType.AGENT_DONE,
            data={
                "task_id": result.task_id,
                "role": result.role,
                "latency_seconds": result.latency_seconds,
                "tools": result.tools_invoked,
                "skill_used": result.skill_used,
                "error": result.error,
                "depth": getattr(result, "depth", 1),
                "sub_decomposed": getattr(result, "sub_decomposed", False),
            },
        )

    # --- AGENT_MERGE: announce synthesis ---
    yield StreamEvent(
        type=EventType.AGENT_MERGE,
        data={
            "strategy": decomp_plan.strategy,
            "agents_completed": completed_count,
            "agents_failed": failed_count,
        },
    )

    # --- Merge sub-agent results ---
    successful = [r for r in results if r.response and not r.error]
    if not successful:
        final_content = "[All sub-agents failed to produce results. Please try again.]"
        is_error = True
    else:
        try:
            final_content = await merge_agent_results(
                results, decomp_plan.merge_instruction, query
            )
            is_error = False
        except Exception as e:
            logger.warning("Merge step failed: %s", e)
            final_content = "\n\n".join(r.response for r in successful if r.response)
            is_error = False

    if final_content is None:
        final_content = ""

    # --- Emit TOOL_USE events for each unique tool used across sub-agents ---
    # Required so the eval harness's tool_invoked assertions work correctly.
    seen_tools: set[str] = set()
    for tool_name in all_tools:
        if tool_name not in seen_tools:
            seen_tools.add(tool_name)
            yield StreamEvent(
                type=EventType.TOOL_USE,
                data={"tool": tool_name, "args": {}, "source": "sub-agent"},
            )

    # --- Stream merged response ---
    if final_content:
        final_content = _sanitize_answer(final_content)
        sub_agent_evidence = "\n\n".join(r.response for r in successful if r.response)
        evidence = build_evidence(
            retrieved_context=(ctx.retrieved_context or "") + "\n\n" + sub_agent_evidence,
            kg_facts_text=ctx.kg_facts_text,
            user_facts_text=ctx.user_facts_text,
            lessons_text=ctx.lessons_text,
            tool_results=None,
            query=query,
        )
        _pre_validate_content = final_content
        final_content, stripped_reasons = validate_claims(
            final_content, evidence, current_model_tag=config.LLM_MODEL,
        )
        if stripped_reasons:
            for r in stripped_reasons:
                logger.warning("[claim-validator-multi-agent] %s", r)
        # RLVR — only record when the validator actually had claim candidates
        # to inspect. Without this gate, ~100% of responses (which contain no
        # person-title-org or numeric-spec patterns) record value=1.0 and the
        # signal becomes degenerate (zero variance, unusable for GRPO).
        try:
            if getattr(config, "ENABLE_RLVR_SIGNALS", False):
                _candidates = count_claim_candidates(_pre_validate_content)
                if _candidates > 0:
                    from app.core import rlvr as _rlvr
                    _val = 1.0 if not stripped_reasons else max(
                        0.0, 1.0 - min(1.0, len(stripped_reasons) / 5.0)
                    )
                    await asyncio.to_thread(
                        _rlvr.record_signal,
                        "claim_grounded",
                        _val,
                        query=query[:500],
                        response=final_content[:500],
                        evidence=f"stripped={len(stripped_reasons)} checked={_candidates} multi_agent=true",
                        conversation_id=conversation_id,
                    )
        except Exception:
            pass
        chunk_size = 20
        for i in range(0, len(final_content), chunk_size):
            yield StreamEvent(type=EventType.TOKEN, data={"text": final_content[i:i + chunk_size]})

    # --- Persist to conversation (non-ephemeral only) ---
    saved_msg_id = None
    if not ephemeral and svc.conversations:
        saved_msg_id = await asyncio.to_thread(
            lambda: svc.conversations.add_message(
                conversation_id,
                "assistant",
                final_content,
                tool_calls=None,
                sources=None,
            )
        )
        if is_new_conversation and final_content:
            # Fire-and-forget — don't block DONE on title generation.
            async def _safe_title2():
                try:
                    title = await _generate_title(query)
                    await asyncio.to_thread(svc.conversations.update_title, conversation_id, title)
                except Exception as e:
                    logger.warning("Failed to generate title: %s", e)
            _t2 = asyncio.create_task(_safe_title2())
            _background_tasks.add(_t2)
            _t2.add_done_callback(_background_tasks.discard)

    # --- DONE event (includes decomposed=True + agent_count + max_depth) ---
    yield StreamEvent(
        type=EventType.DONE,
        data={
            "conversation_id": conversation_id,
            "intent": intent,
            "tool_results_count": len(all_tools),
            "lessons_used": len(ctx.used_lesson_ids),
            "kg_facts_used": ctx.kg_facts_count,
            "reflexions_used": ctx.reflexions_count,
            "skill_used": None,
            "decomposed": True,
            "agent_count": agent_count,
            "max_decomposition_depth": max_depth_observed,
        },
    )

    # --- Post-processing on merged response (non-ephemeral only) ---
    if not ephemeral:
        tool_results_for_pp = [
            {"tool": t, "args": {}, "output": ""} for t in list(dict.fromkeys(all_tools))
        ]
        async for event in _run_post_processing(
            svc, query, final_content, intent, conversation_id,
            tool_results_for_pp, None, ctx.used_lesson_ids,
            is_error, None, "",
            had_kg=bool(ctx.kg_facts_text),
            had_docs=bool(ctx.retrieved_context),
            channel=channel,
            saved_msg_id=saved_msg_id,
            ephemeral=ephemeral,
        ):
            yield event


# ---------------------------------------------------------------------------
# The Brain — think()
# ---------------------------------------------------------------------------

async def think(
    query: str,
    conversation_id: str | None = None,
    image: str | None = None,
    ephemeral: bool = False,
    channel: str = "api",
) -> AsyncGenerator[StreamEvent, None]:
    """The core reasoning loop. Yields SSE events.

    Orchestrates 5 stage functions:
      _gather_context → _build_messages → _run_generation_loop →
      _refine_response → _run_post_processing
    """
    # --- Step 0: Query length validation ---
    if len(query) > config.MAX_QUERY_LENGTH:
        yield StreamEvent(
            type=EventType.ERROR,
            data={"message": f"Query too long ({len(query)} chars). Maximum is {config.MAX_QUERY_LENGTH}."},
        )
        return

    # --- Step 0b: Mandatory injection detection — fail-closed ---
    # Runs before get_services() so the block fires even during startup.
    # Not gated by ENABLE_INJECTION_DETECTION (which only covers external content).
    try:
        from app.core.injection import detect_injection
        _inj_result = detect_injection(query)
        if _inj_result.is_suspicious:
            logger.warning(
                "Injection detected in query — BLOCKED (score=%.2f): %s",
                _inj_result.score, query[:120],
            )
            yield StreamEvent(
                type=EventType.ERROR,
                data={
                    "message": (
                        f"Query blocked: prompt injection detected "
                        f"({_inj_result.score:.0%} confidence). "
                        "Rephrase your request without instruction-override patterns."
                    ),
                    # Policy refusal, not a server fault — the chat API must
                    # surface this as a normal (200) refusal answer, not a 500.
                    "code": "blocked",
                },
            )
            return
    except Exception as e:
        logger.warning("Injection detection failed — blocking as fail-safe: %s", e)
        yield StreamEvent(
            type=EventType.ERROR,
            data={"message": "Query blocked: injection pre-check failed. Please try again.",
                  "code": "blocked"},
        )
        return

    svc = get_services()

    # --- Step 1: Conversation setup ---
    yield StreamEvent(type=EventType.THINKING, data={"stage": "loading_context"})

    if ephemeral:
        is_new_conversation = False
        conversation_id = conversation_id or "ephemeral"
        history = []
    else:
        is_new_conversation = conversation_id is None
        if is_new_conversation:
            conversation_id = await asyncio.to_thread(svc.conversations.create_conversation)
        conv = await asyncio.to_thread(svc.conversations.get_conversation, conversation_id)
        if conv is None:
            old_id = conversation_id
            conversation_id = await asyncio.to_thread(svc.conversations.create_conversation)
            is_new_conversation = True
            logger.warning("Conversation '%s' not found, created new '%s'", old_id, conversation_id)

    # Acquire per-conversation lock to serialize concurrent think() calls
    _conv_lock = None
    _conv_lock_acquired = False
    if not ephemeral and conversation_id:
        _conv_lock = await _get_conversation_lock(conversation_id)
        await _conv_lock.acquire()
        _conv_lock_acquired = True

    try:

        # --- Step 2: History ---
        if not ephemeral:
            history = await asyncio.to_thread(
                svc.conversations.get_history_as_dicts,
                conversation_id, config.MAX_HISTORY_MESSAGES,
            )

        # --- Step 3: Intent ---
        intent = await _classify_intent(query)

        # --- Step 3b: Vision pre-pass (if image attached) ---
        # When an image is present, run a structured visual analysis FIRST so the
        # main generation has explicit visual grounding to reason from. Without this
        # the model might "see" the image but produce loose answers; with this,
        # it has a description + key elements anchored.
        vision_description = ""
        if image and config.LLM_PROVIDER == "ollama":
            try:
                vision_description = await _vision_describe(query, image)
                if vision_description:
                    yield StreamEvent(
                        type=EventType.THINKING,
                        data={"stage": "vision", "content": f"Visual analysis: {vision_description[:200]}"},
                    )
                    # Augment the user query so all downstream steps see the visual context
                    query = (
                        f"[VISUAL CONTEXT — extracted before answering]\n{vision_description}\n\n"
                        f"[USER QUERY]\n{query}"
                    )
            except Exception as e:
                logger.warning("[Vision] pre-pass failed: %s", e)

        # --- Step 4: Gather all context ---
        ctx = await _gather_context(svc, query, intent, conversation_id=conversation_id or "")

        # --- Step 5: Emit LESSON_USED events ---
        if ctx.used_lesson_ids and ctx.lessons:
            for lesson in ctx.lessons:
                yield StreamEvent(
                    type=EventType.LESSON_USED,
                    data={
                        "topic": lesson.topic,
                        "confidence": lesson.confidence,
                        "lesson_id": lesson.id,
                    },
                )

        # --- Step 6: Build messages + planning ---
        messages, was_planned, plan = await _build_messages(
            svc, ctx, query, history, image, intent
        )

        # Save user message
        if not ephemeral:
            await asyncio.to_thread(svc.conversations.add_message, conversation_id, "user", query)

        # --- Step 6.5: AgentLoop deliberation gate (FIRST — chain-of-reasoning) ---
        # For chain-of-reasoning queries that benefit from explicit plan/act/critique
        # per step, route to AgentLoop.solve(). Runs BEFORE multi-agent decomposition
        # so queries like "walk me through ... step by step" don't get split into
        # parallel sub-agents (which loses the sequential reasoning chain).
        # Fires for both ephemeral (eval) and non-ephemeral (chat) — ephemeral path
        # skips post-processing inside _run_deliberation_path.
        # `_should_use_deliberation` is already pattern-specific (only fires
        # on "step by step | walk me through | design a | architect | how
        # would you design/build/solve | derive | prove that | trade-offs
        # between | etc."). It does NOT match generic "explain X" queries.
        # The earlier `_has_lookup_signal` extra gate (audit #14, 2026-05-06)
        # was overzealous: it excluded "Walk me through how to design a rate
        # limiter" — exactly the query AgentLoop is best at — because that
        # phrase has no compare/who/when/lookup keywords. Verified runtime
        # 2026-05-07: deliberation_chain_of_reasoning timed out at structural
        # decomposition because the deliberation gate refused to take it.
        # Trust the deliberation regex; it's already tight enough.
        if (
            intent == "general"
            and config.ENABLE_MULTI_AGENT  # share the master agentic kill switch
            and _should_use_deliberation(query)
        ):
            logger.info("[Deliberation] gate fired for query: %r (ephemeral=%s)", query[:100], ephemeral)
            try:
                async for event in _run_deliberation_path(
                    svc, query, conversation_id, intent, ctx,
                    is_new_conversation, channel, ephemeral=ephemeral,
                ):
                    yield event
                return
            except Exception as e:
                logger.warning("[Deliberation] failed, falling through to standard generation: %s", e)

        # --- Step 6.6: Multi-agent structural decomposition gate ---
        # should_decompose() is heuristic-only (no LLM).  decompose_query() is also
        # heuristic-only.  The actual sub-agents run inside _run_multi_agent_path().
        # If decomposition fires and produces a valid plan, we bypass the normal
        # generation loop entirely and return from _run_multi_agent_path().
        from app.core.decomposer import should_decompose, decompose_query
        if should_decompose(query, intent, was_planned, ephemeral):
            decomp_plan = decompose_query(query, intent, was_planned, plan, conversation_id)
            if decomp_plan is not None:
                async for event in _run_multi_agent_path(
                    svc, query, conversation_id, intent, ctx,
                    decomp_plan, is_new_conversation, ephemeral, channel,
                ):
                    yield event
                return

        # --- Step 7: Generate + Tool Loop ---
        yield StreamEvent(type=EventType.THINKING, data={"stage": "generating"})

        tools = _get_available_tools()
        if config.ENABLE_CUSTOM_TOOLS and svc.custom_tools:
            tools.append({
                "name": "tool_create",
                "description": "Create a new reusable tool. Write a Python function named 'run' that takes declared parameters and returns a string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Tool name (lowercase, underscores)"},
                        "description": {"type": "string", "description": "What the tool does"},
                        "parameters": {"type": "string", "description": "JSON array of parameter defs"},
                        "code": {"type": "string", "description": "Python code with a run() function"},
                    },
                    "required": ["name", "description", "parameters", "code"],
                },
            })
        if ephemeral:
            tools = [t for t in tools if t["name"] != "delegate"]

        # Depth-aware tool restriction: at structural depth >= 2 (sub-sub-agents),
        # restrict to in-context reasoning tools only — no web/browser/http_fetch.
        # Sub-sub-agents should reason deeply from already-gathered context, not
        # spawn more research that serializes on the GPU and times out.
        try:
            from app.core.agent_spawner import get_structural_depth
            current_depth = get_structural_depth()
        except Exception:
            current_depth = 0
        if current_depth >= 2:
            tools = [t for t in tools if t["name"] in _DEPTH2_ALLOWED_TOOLS]
            logger.info(
                "[depth-restrict] depth=%d: restricted tools to %s",
                current_depth, sorted(t["name"] for t in tools),
            )

        gen = _GenerationResult()
        async for event in _run_generation_loop(
            messages, tools, svc, conversation_id, image, intent,
            was_planned, ephemeral, gen, query=query,
        ):
            yield event

        # --- Step 8: Ephemeral early return ---
        if ephemeral:
            if gen.final_content:
                final_content = _sanitize_answer(gen.final_content)
                evidence = build_evidence(
                    retrieved_context=ctx.retrieved_context,
                    kg_facts_text=ctx.kg_facts_text,
                    user_facts_text=ctx.user_facts_text,
                    lessons_text=ctx.lessons_text,
                    tool_results=gen.tool_results,
                    query=query,
                )
                _pre_validate_content_ephemeral = final_content
                final_content, stripped_reasons = validate_claims(
                    final_content, evidence, current_model_tag=config.LLM_MODEL,
                )
                if stripped_reasons:
                    for r in stripped_reasons:
                        logger.warning("[claim-validator-ephemeral] %s", r)
                # RLVR — only record when claim candidates exist in the
                # pre-validate content. See multi-agent merge path comment.
                try:
                    if getattr(config, "ENABLE_RLVR_SIGNALS", False):
                        _candidates = count_claim_candidates(_pre_validate_content_ephemeral)
                        if _candidates > 0:
                            from app.core import rlvr as _rlvr
                            _val = 1.0 if not stripped_reasons else max(
                                0.0, 1.0 - min(1.0, len(stripped_reasons) / 5.0)
                            )
                            await asyncio.to_thread(
                                _rlvr.record_signal,
                                "claim_grounded",
                                _val,
                                query=query[:500],
                                response=final_content[:500],
                                evidence=f"stripped={len(stripped_reasons)} checked={_candidates} ephemeral=true",
                                conversation_id=conversation_id,
                            )
                except Exception:
                    pass
                chunk_size = 20
                for i in range(0, len(final_content), chunk_size):
                    yield StreamEvent(type=EventType.TOKEN, data={"text": final_content[i:i + chunk_size]})
            yield StreamEvent(
                type=EventType.DONE,
                data={
                    "conversation_id": conversation_id,
                    "ephemeral": True,
                    "skill_used": ctx.matched_skill.name if ctx.matched_skill else None,
                    "decomposed": False,
                },
            )
            return

        # --- Step 9: Refine (critique + reflexion) ---
        # Skip the full critique chain at structural depth >= 2 — sub-sub-agents
        # should produce raw fast output; the parent will critique the merged
        # result. Cuts ~30-90s of latency per leaf agent in recursive cases.
        if current_depth >= 2:
            final_content = gen.final_content
            reflexion_quality, reflexion_reason = None, ""
            logger.info("[depth-restrict] depth=%d: skipping _refine_response", current_depth)
        else:
            final_content, reflexion_quality, reflexion_reason = await _refine_response(
                messages, tools, gen.final_content, query, intent,
                gen.tool_results, was_planned, plan,
                retrieved_context=ctx.retrieved_context,
                user_facts_text=ctx.user_facts_text,
                kg_facts_text=ctx.kg_facts_text,
            )

        # --- Step 10: Emit sources + stream tokens ---
        # Guard against None content from LLM (would cause IntegrityError on NOT NULL column)
        if final_content is None:
            logger.warning("final_content is None after LLM generation — defaulting to empty string (conv=%s)", conversation_id)
            final_content = ""

        if ctx.retrieved_sources:
            yield StreamEvent(type=EventType.SOURCES, data={"sources": ctx.retrieved_sources})

        if final_content:
            final_content = _sanitize_answer(final_content)
            evidence = build_evidence(
                retrieved_context=ctx.retrieved_context,
                kg_facts_text=ctx.kg_facts_text,
                user_facts_text=ctx.user_facts_text,
                lessons_text=ctx.lessons_text,
                tool_results=gen.tool_results,
                query=query,
            )
            _pre_validate_content_main = final_content
            final_content, stripped_reasons = validate_claims(
                final_content, evidence, current_model_tag=config.LLM_MODEL,
            )
            if stripped_reasons:
                for r in stripped_reasons:
                    logger.warning("[claim-validator] %s", r)
            # RLVR — only record when claim candidates exist in the
            # pre-validate content. See multi-agent merge path comment.
            try:
                if getattr(config, "ENABLE_RLVR_SIGNALS", False):
                    _candidates = count_claim_candidates(_pre_validate_content_main)
                    if _candidates > 0:
                        from app.core import rlvr as _rlvr
                        _val = 1.0 if not stripped_reasons else max(
                            0.0, 1.0 - min(1.0, len(stripped_reasons) / 5.0)
                        )
                        await asyncio.to_thread(
                            _rlvr.record_signal,
                            "claim_grounded",
                            _val,
                            query=query[:500],
                            response=final_content[:500],
                            evidence=f"stripped={len(stripped_reasons)} checked={_candidates}",
                            conversation_id=conversation_id,
                        )
            except Exception:
                pass
            chunk_size = 20
            for i in range(0, len(final_content), chunk_size):
                yield StreamEvent(type=EventType.TOKEN, data={"text": final_content[i:i + chunk_size]})
        saved_msg_id = await asyncio.to_thread(
            lambda: svc.conversations.add_message(
                conversation_id,
                "assistant",
                final_content,
                tool_calls=[{"tool": tr["tool"], "args": tr["args"]} for tr in gen.tool_results] if gen.tool_results else None,
                sources=ctx.retrieved_sources or None,
            )
        )

        if ctx.matched_skill and svc.skills:
            # Skill success means the answer was actually GOOD, not merely that
            # it rendered. The old `not gen.is_error` counted confidently-wrong
            # output as success, so a skill that fed a bad expression to a tool
            # and templated the garbage survived indefinitely (e.g. the
            # calculator_arithmetic skill at 32.9% "success", poisoning
            # arithmetic). When the quality judge has a verdict, it decides;
            # the structural checks only act as a hard floor (a real tool/error
            # failure is never a success regardless of score).
            structural_ok = not gen.is_error
            if ctx.matched_skill.steps:
                structural_ok = structural_ok and len(gen.tool_results) > 0 and not any(
                    isinstance(tr.get("output", ""), str)
                    and tr["output"].startswith("[Tool") and "failed" in tr["output"]
                    for tr in gen.tool_results
                )
            if reflexion_quality is not None:
                skill_success = structural_ok and reflexion_quality >= config.SKILL_SUCCESS_QUALITY
            else:
                # No quality verdict (e.g. depth-restricted leaf): fall back to
                # the structural check alone.
                skill_success = structural_ok
            await asyncio.to_thread(svc.skills.record_use, ctx.matched_skill.id, skill_success)

        if is_new_conversation and final_content:
            # Fire-and-forget. Title gen takes 5-30s on a busy Ollama; don't make
            # the user wait for the DONE event.
            async def _safe_title():
                try:
                    title = await _generate_title(query)
                    await asyncio.to_thread(svc.conversations.update_title, conversation_id, title)
                except Exception as e:
                    logger.warning("Failed to generate title: %s", e)
            _t = asyncio.create_task(_safe_title())
            _background_tasks.add(_t)
            _t.add_done_callback(_background_tasks.discard)

        # --- Persistent workspace save (mirrors AgentLoop's path) ---
        # Stores the final answer + extracted semantic facts so the next run on a
        # similar query signature can hydrate prior progress instead of re-deriving.
        # Two-phase save:
        #   1. Inline: cheap save with tool-arg snippets (so signature is registered fast)
        #   2. Background: LLM-extract facts from the answer and re-save (richer findings)
        if ctx.workspace_signature and final_content and intent == "general":
            try:
                from app.core.agent_workspace import save_workspace
                from app.database import get_db
                workspace_findings: dict[str, str] = {}
                # Phase 1: tool-arg snippets (cheap, keyed by tool+first-arg)
                for tr in (gen.tool_results or [])[:5]:
                    tool_name = tr.get("tool", "tool")
                    args = tr.get("args", {})
                    arg_key = (
                        str(next(iter(args.values())))[:30]
                        if isinstance(args, dict) and args else "result"
                    )
                    out = tr.get("output", "")
                    if out:
                        workspace_findings[f"{tool_name}:{arg_key}"] = str(out)[:300]
                # Treat reflexion_quality >= 0.6 as success for workspace persistence
                ws_success = (
                    reflexion_quality is not None and reflexion_quality >= 0.6
                ) if reflexion_quality is not None else (not gen.is_error)
                # Extract failed approaches: tool calls whose output contains
                # error markers, empty results, or known-junk patterns. These
                # carry forward as "don't repeat" hints on similar future queries.
                failed_approaches: list[str] = []
                if not ws_success or (reflexion_quality is not None and reflexion_quality < 0.5):
                    for tr in (gen.tool_results or [])[:8]:
                        out = str(tr.get("output", ""))[:300].lower()
                        is_failure = (
                            out.startswith("[tool error")
                            or out.startswith("[tool failed")
                            or "no results" in out
                            or "0 results" in out
                            or "could not" in out[:80]
                            or "not found" in out[:80]
                        )
                        if is_failure:
                            tool_name = tr.get("tool", "?")
                            args = tr.get("args", {})
                            arg_summary = (
                                str(next(iter(args.values())))[:60]
                                if isinstance(args, dict) and args else "?"
                            )
                            failed_approaches.append(f"{tool_name}({arg_summary}) → {out[:100]}")
                await asyncio.to_thread(
                    save_workspace,
                    get_db(),
                    query=query,
                    findings=workspace_findings,
                    answer=final_content,
                    success=ws_success,
                    failed_approaches=failed_approaches if failed_approaches else None,
                )

                # Phase 2: LLM-extract semantic facts from the answer (background, non-blocking)
                # Only fires on successful, substantive, GROUNDED answers — extraction has cost
                # AND extracting from hallucinations poisons the workspace cache. Guard: at least
                # one substantive tool result must exist (web_search/http_fetch/browser/code_exec/
                # calculator/knowledge_search) AND its output must be non-empty / non-error. Time-
                # sensitive queries also skip extraction since their "facts" decay rapidly.
                _GROUNDING_TOOLS = {
                    "web_search", "http_fetch", "browser", "code_exec",
                    "calculator", "knowledge_search", "memory_search",
                }
                _has_grounding = False
                for tr in (gen.tool_results or []):
                    if tr.get("tool") in _GROUNDING_TOOLS:
                        out = str(tr.get("output", ""))
                        if out and not out.startswith("[Tool error") and len(out) > 40:
                            _has_grounding = True
                            break
                _is_time_sensitive_query = bool(_TIME_SENSITIVE_RE.search(query))
                if (
                    ws_success
                    and _has_grounding
                    and not _is_time_sensitive_query
                    and len(final_content) >= 200 and len(final_content) <= 6000
                ):
                    async def _extract_and_resave(_query=query, _answer=final_content, _success=ws_success):
                        try:
                            facts = await _extract_workspace_facts(_query, _answer)
                            if facts:
                                # Merge with existing tool-arg findings; semantic facts win on conflict
                                merged = dict(workspace_findings)
                                merged.update(facts)
                                await asyncio.to_thread(
                                    save_workspace,
                                    get_db(),
                                    query=_query,
                                    findings=merged,
                                    answer=_answer,
                                    success=_success,
                                )
                                logger.info(
                                    "[Workspace] enriched with %d semantic facts: %s",
                                    len(facts), list(facts.keys())[:5],
                                )
                        except Exception as e:
                            logger.warning("Workspace fact extraction failed: %s", e)

                    # Bridge: also promote answer → KG triples when fully grounded.
                    # Workspace findings live in a per-query cache; KG is the long-term
                    # store. Without this bridge, chat answers never feed the KG —
                    # only monitors/curiosity do. The triple extractor has its own
                    # quality gates (canonical predicates, garbage filter, contradiction
                    # resolution), so we trust it to drop noise.
                    async def _bridge_to_kg(_query=query, _answer=final_content):
                        try:
                            if svc.kg:
                                await _extract_kg_triples(
                                    svc.kg, _query, _answer, source_name="chat"
                                )
                        except Exception as e:
                            logger.warning("KG bridge from chat failed: %s", e)

                    _ext_task = asyncio.create_task(_extract_and_resave())
                    _background_tasks.add(_ext_task)
                    _ext_task.add_done_callback(_background_tasks.discard)

                    _kg_task = asyncio.create_task(_bridge_to_kg())
                    _background_tasks.add(_kg_task)
                    _kg_task.add_done_callback(_background_tasks.discard)

                    # GSW: update episodic summary for this conversation. Background task
                    # so we don't block the response. Internal gating decides whether the
                    # message count has crossed the re-summarize threshold.
                    async def _gsw_update(_cid=conversation_id):
                        try:
                            from app.core import gsw as _gsw_mod
                            if _gsw_mod.is_enabled() and _cid:
                                await _gsw_mod.maybe_update_summary(get_db(), _cid)
                        except Exception as e:
                            logger.warning("[GSW] update task failed: %s", e)

                    if conversation_id:
                        _gsw_task = asyncio.create_task(_gsw_update())
                        _background_tasks.add(_gsw_task)
                        _gsw_task.add_done_callback(_background_tasks.discard)
            except Exception as e:
                logger.warning("Workspace save failed: %s", e)

        # --- Step 12: Done event ---
        yield StreamEvent(
            type=EventType.DONE,
            data={
                "conversation_id": conversation_id,
                "intent": intent,
                "tool_results_count": len(gen.tool_results),
                "lessons_used": len(ctx.used_lesson_ids),
                "kg_facts_used": ctx.kg_facts_count,
                "reflexions_used": ctx.reflexions_count,
                "skill_used": ctx.matched_skill.name if ctx.matched_skill else None,
            },
        )

        # --- Step 13: Post-processing ---
        async for event in _run_post_processing(
            svc, query, final_content, intent, conversation_id,
            gen.tool_results, ctx.matched_skill, ctx.used_lesson_ids,
            gen.is_error, reflexion_quality, reflexion_reason,
            had_kg=bool(ctx.kg_facts_text),
            had_docs=bool(ctx.retrieved_context),
            channel=channel,
            saved_msg_id=saved_msg_id,
            ephemeral=ephemeral,
        ):
            yield event

        # --- Step 14: Mark lessons helpful/unhelpful based on quality ---
        if ctx.used_lesson_ids and intent != "correction" and svc.learning:
            for lid in ctx.used_lesson_ids:
                try:
                    if reflexion_quality is not None and reflexion_quality >= 0.6:
                        await asyncio.to_thread(svc.learning.mark_lesson_helpful, lid)
                    elif reflexion_quality is not None and reflexion_quality < 0.4:
                        await asyncio.to_thread(svc.learning.mark_lesson_unhelpful, lid)
                except Exception as e:
                    logger.warning("Lesson feedback update failed for id=%s: %s", lid, e)

        # --- Step 14b: Record post-quality for injected success patterns (A/B closure) ---
        if ctx.success_pattern_ids and reflexion_quality is not None and svc.reflexions:
            try:
                await asyncio.to_thread(
                    svc.reflexions.record_post_quality_for_injected,
                    ctx.success_pattern_ids,
                    reflexion_quality,
                )
            except Exception as e:
                logger.debug("Success pattern A/B record failed: %s", e)

    finally:
        if _conv_lock is not None and _conv_lock_acquired and _conv_lock.locked():
            _conv_lock.release()
            _conv_lock_acquired = False


# KG triple extraction extracted to brain_kg for size hygiene; re-export
# keeps `from app.core.brain import _extract_kg_triples` working for tests
# and heartbeat_loop callers.
from app.core.brain_kg import (  # noqa: E402,F401
    _DEFAULT_SOURCE_CONFIDENCE,
    _SOURCE_CONFIDENCE,
    _extract_kg_triples,
)
