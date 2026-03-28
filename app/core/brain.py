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
from app.core import llm
from app.core.learning import is_likely_correction, response_pushes_back
from app.core.llm import LLMUnavailableError, _extract_tool_calls
from app.core.memory import ConversationStore, UserFactStore, has_fact_signals, extract_facts_from_message
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
# Context window management
# ---------------------------------------------------------------------------

from app.core.text_utils import estimate_tokens as _estimate_tokens


async def _manage_context(
    system_prompt: str,
    history: list[dict],
    query: str,
) -> tuple[list[dict], str]:
    """Manage context window to stay within budget.

    If the total token count exceeds MAX_CONTEXT_TOKENS, summarize older
    messages and keep only the most recent RECENT_MESSAGES_KEEP messages
    verbatim.

    Returns: (trimmed_history, conversation_summary)
    """
    # Estimate total tokens
    system_tokens = _estimate_tokens(system_prompt)
    query_tokens = _estimate_tokens(query)
    history_tokens = sum(_estimate_tokens(m.get("content", "")) for m in history)
    response_budget = config.RESPONSE_TOKEN_BUDGET  # Reserve tokens for the response

    # 20% safety buffer on estimates (heuristic ~4 chars/token can be 30-50% off)
    total = int((system_tokens + history_tokens + query_tokens) * 1.2) + response_budget

    if total <= config.MAX_CONTEXT_TOKENS:
        return history, ""

    # Over budget — need to summarize older messages
    keep = config.RECENT_MESSAGES_KEEP
    if len(history) <= keep:
        return history, ""

    # Split: old messages to summarize, recent to keep
    old_messages = history[:-keep]
    recent_messages = history[-keep:]

    # Build a summary of old messages
    summary_input = "\n".join(
        f"[{m.get('role', '?')}]: {m.get('content', '')[:300]}"
        for m in old_messages
    )

    try:
        summary = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation in 2-3 SHORT sentences. "
                            "Only key facts and decisions. No detail, no examples. "
                            "IMPORTANT: Preserve ALL dates, numbers, proper nouns, "
                            "monetary amounts, and technical values exactly as stated. "
                            "Example: 'Discussed deployment on March 15. User prefers Y. Budget is $50,000.'"
                        ),
                    },
                    {"role": "user", "content": summary_input},
                ],
                max_tokens=150,
                temperature=0.1,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
        summary = summary.strip()
        logger.info(
            "Context managed: %d→%d messages, summary=%d chars (budget: %d/%d tokens)",
            len(history), keep, len(summary),
            system_tokens + sum(_estimate_tokens(m.get("content", "")) for m in recent_messages) + query_tokens,
            config.MAX_CONTEXT_TOKENS,
        )

        # Re-check: if total tokens still exceed budget after summarization,
        # truncate the summary further (keep only the last half)
        post_summary_tokens = (
            system_tokens
            + sum(_estimate_tokens(m.get("content", "")) for m in recent_messages)
            + query_tokens
            + _estimate_tokens(summary)
            + response_budget
        )
        if post_summary_tokens > config.MAX_CONTEXT_TOKENS and summary:
            half = len(summary) // 2
            # Find nearest sentence boundary after the half position
            boundary = -1
            for sep in (". ", "\n"):
                pos = summary.find(sep, half)
                if pos != -1 and (boundary == -1 or pos < boundary):
                    boundary = pos + len(sep)
            if boundary != -1 and boundary < len(summary):
                summary = summary[boundary:].lstrip()
            else:
                summary = summary[half:].lstrip()
            logger.info(
                "Post-summarization budget still exceeded — truncated summary to %d chars",
                len(summary),
            )

        return recent_messages, summary
    except (Exception, asyncio.TimeoutError) as e:
        logger.warning("Summarization failed: %s — truncating instead", e)
        truncation_note = f"[{len(old_messages)} older messages truncated due to context limits]"
        return recent_messages, truncation_note


# ---------------------------------------------------------------------------
# Intent classification (fast, no LLM call)
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)\b",
    re.IGNORECASE,
)

# Pure greeting = greeting words + optional filler (punctuation, "there", "nova"), nothing else
_PURE_GREETING = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)"
    r"(?:\s+(?:there|nova|buddy|mate|friend|everyone|all))?"
    r"[!?.,:;\s]*$",
    re.IGNORECASE,
)


async def _classify_intent(query: str) -> str:
    """Intent classification — regex first, LLM tiebreaker for ambiguous greetings.

    Returns: 'greeting', 'correction', or 'general'
    """
    stripped = query.strip()

    # Use the single source of truth from learning.py
    if is_likely_correction(stripped):
        return "correction"

    if _GREETING_PATTERNS.match(stripped):
        # Pure greeting (short, no real content) → fast path
        if _PURE_GREETING.match(stripped):
            return "greeting"
        # Ambiguous: starts with greeting but has substantive content
        # Any non-pure greeting with 4+ words likely has real content
        word_count = len(stripped.split())
        if word_count > 3:
            try:
                result = await asyncio.wait_for(
                    llm.invoke_nothink(
                        [{"role": "user", "content": (
                            f'Is this a simple greeting or a real question/request? '
                            f'Reply with ONE word: "greeting" or "general".\n\n"{stripped}"'
                        )}],
                        max_tokens=10,
                        temperature=0,
                    ),
                    timeout=config.INTERNAL_LLM_TIMEOUT,
                )
                if result is None:
                    return "general"
                classification = result.strip().lower().strip('"\'.')
                if classification in ("greeting", "general"):
                    return classification
            except (Exception, asyncio.TimeoutError):
                pass  # Fall through to "general" on LLM failure
            return "general"
        return "greeting"

    return "general"


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------

async def _generate_title(query: str) -> str:
    """Generate a short conversation title from the first query."""
    try:
        result = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {"role": "system", "content": (
                        "Generate a 3-5 word title summarizing this conversation topic. "
                        "Rules: NO emojis, NO quotes, NO punctuation, plain English only. "
                        "Return ONLY the title words, nothing else."
                    )},
                    {"role": "user", "content": query},
                ],
                max_tokens=15,
                temperature=0.2,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Title generation timed out")
        return query[:40].strip()
    if result is None:
        return query[:40].strip()
    # Clean up: strip quotes, emojis, limit length
    title = result.strip().strip('"\'').strip()
    # Remove any emojis or non-ASCII
    title = "".join(c for c in title if c.isascii()).strip()
    if not title or len(title) > 60:
        # Fallback: first few words of query
        words = query.split()[:5]
        return " ".join(words)
    return title


# ---------------------------------------------------------------------------
# Answer sanitizer — strip meta-commentary
# ---------------------------------------------------------------------------

_META_PATTERNS = [
    re.compile(r"\*+Note:.*?correction.*?\*+", re.IGNORECASE),
    re.compile(r"\*+Note:.*?lesson.*?\*+", re.IGNORECASE),
    re.compile(r"\*+Note:.*?(?:I'(?:ve|ll)|updated|saved|remembered|stored|recorded).*?\*+", re.IGNORECASE),
    re.compile(r"^I've (?:noted|recorded|saved|updated|stored) (?:your|that|this) correction.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Thank you for (?:the )?correction.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<think>.*</think>", re.DOTALL),
    # Date confusion disclaimers (Qwen calls 2026 a "simulated future date")
    re.compile(r"\b(?:simulated|hypothetical)\s+(?:future\s+)?date\b[^.]*\.?", re.IGNORECASE),
    re.compile(r"\b(?:since|as)\s+(?:my\s+)?training\s+(?:data\s+)?cut-?off\b[^.]*\.?", re.IGNORECASE),
    re.compile(r"\bthis\s+(?:appears?\s+to\s+be\s+)?a\s+future\s+date\b[^.]*\.?", re.IGNORECASE),
]


def _sanitize_answer(text: str) -> str:
    """Strip meta-commentary and internal markers from final answers."""
    if not isinstance(text, str):
        return str(text) if text else ""
    for pat in _META_PATTERNS:
        text = pat.sub("", text)
    # Collapse excess blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# ---------------------------------------------------------------------------
# Model routing — fast model for simple queries
# ---------------------------------------------------------------------------

_QUESTION_WORDS = re.compile(
    r"(?i)^(?:who|what|when|where|why|how|which|is|are|was|were|do|does|did|can|could|will|would|should)\b"
)

_COMPLEX_PATTERNS = re.compile(
    r"(?i)\b(?:step[- ]by[- ]step|prove|derive|implement|algorithm|refactor|debug|architect|"
    r"design pattern|trade-?offs?|compare and contrast|write (?:a |an )?(?:function|class|script|program)|"
    r"solve|equation|integral|derivative|optimize|complexity|recursion|dynamic programming|"
    r"explain (?:how|why)|multi-?step|chain of thought)\b"
)

_CREATIVE_PATTERNS = re.compile(
    r"(?i)\b(?:opinion|brainstorm|creative|imagine|suggest|ideas?|write (?:a |an )?(?:poem|story|essay)|what do you think)\b"
)

def _select_model(query: str, intent: str, needs_plan: bool) -> str | None:
    """Return model override or None for default."""
    if not config.ENABLE_MODEL_ROUTING:
        return None
    stripped = query.strip()

    # Fast model for greetings and very short non-questions
    if config.FAST_MODEL:
        if intent == "greeting":
            return config.FAST_MODEL
        if len(stripped) < 40 and not needs_plan and "?" not in stripped and not _QUESTION_WORDS.match(stripped):
            return config.FAST_MODEL

    # Heavy model for complex reasoning queries
    if config.HEAVY_MODEL:
        if needs_plan:
            return config.HEAVY_MODEL
        if _COMPLEX_PATTERNS.search(stripped):
            return config.HEAVY_MODEL

    return None


# ---------------------------------------------------------------------------
# Fact extraction gate — blacklist (skip obvious non-facts)
# ---------------------------------------------------------------------------

_SKIP_FACT_EXTRACTION_RE = re.compile(
    r"^(?:search|find|look up|calculate|show|tell me about|explain|summarize|compare|check|list|describe)\b"
    r"|^(?:hey|hi|hello|thanks|thank you|ok|sure|yes|no|yeah|nah|bye|goodbye)\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _is_pure_question_or_command(text: str) -> bool:
    """Return True if the message is unlikely to contain personal facts.

    Used as a blacklist gate: fact extraction runs by default UNLESS
    this returns True.  The LLM extraction prompt handles remaining
    edge cases by returning {} when no facts are present.
    """
    stripped = text.strip()
    if len(stripped) < 8:
        return True
    # Pure questions (starts with question word)
    if _QUESTION_WORDS.match(stripped):
        return True
    # Pure commands / greetings
    if _SKIP_FACT_EXTRACTION_RE.match(stripped):
        return True
    return False


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
    lessons: list = field(default_factory=list)  # raw lesson objects, for LESSON_USED events
    external_skills_text: str = ""               # summaries of loaded external skills
    matched_external_skill_text: str = ""        # full body of matched external skill


@dataclass
class _GenerationResult:
    """Mutable output from the generation+tool loop."""
    final_content: str = ""
    tool_results: list[dict] = field(default_factory=list)
    is_error: bool = False


# ---------------------------------------------------------------------------
# Stage functions (private helpers for think())
# ---------------------------------------------------------------------------


async def _gather_context(svc: Services, query: str, intent: str) -> _ThinkContext:
    """Load skills, user facts, lessons, KG facts, reflexions, retrieval, integrations.

    This is Steps 4–7b of the original think() pipeline.
    """
    ctx = _ThinkContext()
    ctx.user_facts_text = (await asyncio.to_thread(svc.user_facts.format_for_prompt)) if svc.user_facts else ""

    # --- Skills ---
    if svc.skills:
        ctx.matched_skill = await asyncio.to_thread(svc.skills.get_matching_skill, query)
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
            active_skills = await asyncio.to_thread(svc.skills.get_active_skills)
            if active_skills:
                ctx.skills_text = format_skills_for_prompt([
                    {"name": s.name, "trigger_pattern": s.trigger_pattern}
                    for s in active_skills[:5]
                ])

    # --- Lessons ---
    if svc.learning:
        lessons = await asyncio.to_thread(svc.learning.get_relevant_lessons, query)
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
        try:
            kg_facts = await asyncio.to_thread(svc.kg.get_relevant_facts, query, config.MAX_KG_FACTS_IN_PROMPT)
            if kg_facts:
                ctx.kg_facts_text = svc.kg.format_for_prompt(kg_facts)
                ctx.kg_facts_count = len(kg_facts)
        except Exception as e:
            logger.warning("KG retrieval failed: %s", e)

    # --- Reflexions (past failure warnings) ---
    if svc.reflexions:
        try:
            reflexions = await asyncio.to_thread(svc.reflexions.get_relevant, query, config.MAX_REFLEXIONS_IN_PROMPT)
            if reflexions:
                ctx.reflexions_text = svc.reflexions.format_for_prompt(reflexions)
                ctx.reflexions_count = len(reflexions)
        except Exception as e:
            logger.warning("Reflexion retrieval failed: %s", e)

    # --- Success patterns (what worked before) ---
    if svc.reflexions:
        try:
            from app.core.reflexion import ReflexionStore
            successes = await asyncio.to_thread(svc.reflexions.get_success_patterns, query, config.MAX_SUCCESS_PATTERNS_IN_PROMPT)
            if successes:
                ctx.success_patterns_text = ReflexionStore.format_success_patterns(successes)
        except Exception as e:
            logger.warning("Success pattern retrieval failed: %s", e)

    # --- Retrieval ---
    if svc.retriever and intent == "general":
        try:
            chunks = await svc.retriever.search(query)
            if chunks:
                lines = []
                for i, chunk in enumerate(chunks, 1):
                    score = chunk.score if hasattr(chunk, "score") and chunk.score is not None else 0.0
                    # Skip very low relevance chunks — they add noise
                    if score < config.RETRIEVAL_RELEVANCE_THRESHOLD:
                        continue
                    source = chunk.title or chunk.source or "document"
                    relevance = "high relevance" if score >= 0.7 else ("moderate" if score >= 0.4 else "low relevance")
                    lines.append(f"[{i}] ({relevance} | Source: {source})\n{chunk.content}")
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

    # --- External skills (AgentSkills / OpenClaw) ---
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

    use_thinking = config.ENABLE_EXTENDED_THINKING and not image and selected_model != config.FAST_MODEL
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

    _any_round_succeeded = False  # Track cumulative success across tool rounds

    try:
        for tool_round in range(config.MAX_TOOL_ROUNDS):
            if use_thinking:
                thinking_buf = ""
                content_buf = ""
                _stream_tool_calls: list[llm.ToolCall] = []
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
                # Sanitize ALL tool outputs (sanitize_content is idempotent)
                if config.ENABLE_INJECTION_DETECTION:
                    from app.core.injection import sanitize_content
                    output = sanitize_content(output, context=f"tool:{tc.tool}")
                if cache_key is not None:
                    _tool_cache[cache_key] = (output, tool_result)
                return tc, output, tool_result

            results = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])

            assistant_content = result.content or f'[Calling tool: {tool_calls[0].tool}]'

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

            # Build self-attribution with success-aware framing (provider-aware)
            from datetime import datetime as _dt
            _today = _dt.now().strftime("%B %d, %Y")
            _caps = llm.get_provider().capabilities
            _is_ollama = _caps.needs_emphatic_prompts

            if round_succeeded:
                if _is_ollama:
                    attr_prefix = (
                        "I used my tools and they returned real, live results "
                        f"(not simulated, not hypothetical — actual execution on the network). "
                        f"Today is {_today} — this is the real current date:"
                    )
                else:
                    attr_prefix = f"Tool results (executed {_today}):"
            else:
                attr_prefix = f"I executed the tool(s) and received these results. Today is {_today}:"

            messages.append({
                "role": "assistant",
                "content": f"{assistant_content}\n\n{attr_prefix}\n\n{tool_results_text}",
            })

            # User-role synthesis trigger with result evaluation
            # On intermediate rounds, encourage the model to assess and potentially
            # use more tools. On the final round, just synthesize.
            _is_final_round = (tool_round >= config.MAX_TOOL_ROUNDS - 1)
            if round_succeeded:
                if _is_final_round:
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

        # Log complex tool loops (3+ rounds) as tool creation candidates
        if len(gen.tool_results) >= 3 and config.ENABLE_CUSTOM_TOOLS:
            tools_used = [tr["tool"] for tr in gen.tool_results]
            logger.info(
                "Tool creation candidate: query='%s', %d rounds, tools=%s",
                query[:100], len(gen.tool_results), tools_used,
            )

    except LLMUnavailableError as e:
        logger.error("LLM unavailable: %s", e)
        gen.final_content = "I can't reach the language model right now. Please check that the LLM provider is running and try again."
        gen.is_error = True
        yield StreamEvent(type=EventType.ERROR, data={"message": str(e)})


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
    # --- Self-Critique (addendum-based) ---
    # Instead of regenerating the full response on critique failure,
    # generate a short correction addendum that gets appended.
    # This preserves the already-streamed content and avoids UX jank.
    critique_passed = False
    if config.ENABLE_CRITIQUE and final_content and intent == "general":
        from app.core.critique import should_critique, critique_answer, format_critique_for_regeneration
        if should_critique(query, final_content, intent, tool_results, was_planned, kg_facts=kg_facts_text, user_facts=user_facts_text):
            last_critique_issues: list[str] = []
            for critique_round in range(config.MAX_CRITIQUE_ROUNDS):
                try:
                    critique = await critique_answer(
                        query, final_content,
                        sources=retrieved_context,
                        user_facts=user_facts_text,
                        kg_facts=kg_facts_text,
                    )
                    if not critique or critique.get("pass", True):
                        critique_passed = True
                        break
                    logger.info("Critique round %d failed: %s", critique_round + 1, critique.get("issues", []))
                    # Track last critique issues for post-loop reflexion storage
                    last_critique_issues = critique.get("issues", [])
                    critique_msg = format_critique_for_regeneration(critique)
                    if not critique_msg:
                        break

                    # Generate a correction addendum instead of full re-generation.
                    # The original response is already streamed; appending a correction
                    # is cheaper and avoids replacing content the user has already read.
                    issues_list = critique.get("issues", [])
                    addendum_prompt = (
                        "The previous response had these issues:\n"
                        + "\n".join(f"- {issue}" for issue in issues_list)
                        + "\n\nGenerate ONLY a brief correction addressing these specific issues. "
                        "Do NOT repeat the full answer. Start with the corrected information directly."
                    )
                    # Cap messages to prevent unbounded token growth during critique loops.
                    # Smart truncation: keep system msg + tool results + recent turns.
                    if len(messages) > 10:
                        # Keep system/context messages (index 0..2)
                        head = messages[:3]
                        tail = messages[-7:]
                        # Preserve any tool-result messages from the middle
                        middle_tool_msgs = [
                            m for m in messages[3:-7]
                            if m.get("role") == "tool" or (
                                m.get("role") == "system" and "tool" in m.get("content", "").lower()[:100]
                            )
                        ]
                        messages = head + middle_tool_msgs + tail
                    messages.append({"role": "assistant", "content": final_content})
                    messages.append({"role": "user", "content": addendum_prompt})
                    try:
                        retry_result = await llm.generate_with_tools(messages, tools)
                        if retry_result.content and not retry_result.tool_calls:
                            addendum = retry_result.content.strip()
                            if addendum:
                                final_content = final_content + "\n\n---\n[Correction: " + addendum + "]"
                                logger.info(
                                    "Appended correction addendum after critique round %d (%d chars)",
                                    critique_round + 1, len(addendum),
                                )
                            # Re-run critique on the combined content
                            continue
                        else:
                            break
                    except Exception as e:
                        logger.warning("Critique addendum generation failed (round %d): %s", critique_round + 1, e)
                        break
                except Exception as e:
                    logger.warning("Critique failed (round %d): %s", critique_round + 1, e)
                    break

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

            if reflexion_quality is not None and reflexion_quality < 0.3 and reflexion_reason and not critique_passed:
                logger.info("Reflexion critique flagged (%.2f): %s", reflexion_quality, reflexion_reason)
                # Generate a correction addendum instead of full re-generation
                addendum_msg = (
                    f"The previous response had a quality issue: {reflexion_reason}\n\n"
                    "Generate ONLY a brief correction addressing this specific issue. "
                    "Do NOT repeat the full answer. Start with the corrected information directly."
                )
                messages.append({"role": "assistant", "content": final_content})
                messages.append({"role": "system", "content": addendum_msg})
                try:
                    retry_result = await llm.generate_with_tools(messages, tools)
                    if retry_result.content and not retry_result.tool_calls:
                        addendum = retry_result.content.strip()
                        if addendum:
                            final_content = final_content + "\n\n---\n[Correction: " + addendum + "]"
                        reflexion_quality, reflexion_reason = assess_quality(
                            final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query
                        )
                        logger.info("Appended reflexion correction addendum (%d chars, new score: %.2f)",
                                    len(addendum), reflexion_quality)
                except Exception as e:
                    logger.warning("Reflexion critique addendum failed: %s", e)
        except Exception as e:
            logger.debug("Reflexion pre-stream critique failed: %s", e)

    return final_content, reflexion_quality, reflexion_reason


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
) -> AsyncGenerator[StreamEvent, None]:
    """Post-response processing: corrections, facts, KG, reflexion storage, auto skills.

    This is Steps 13–17 of the original think() pipeline.
    Yields LESSON_LEARNED events.
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

    # --- Automatic fact extraction ---
    # Only run when regex signals detect fact-bearing language
    # Blacklist gate: extract facts by default, skip only for pure questions/commands.
    # The LLM extraction prompt returns {} for non-fact messages (false-positive safe).
    _should_extract = svc.user_facts and not is_error and not _is_pure_question_or_command(query)
    # Skip extraction for likely injection attempts
    if _should_extract and config.ENABLE_INJECTION_DETECTION:
        try:
            from app.core.injection import detect_injection
            _inj = detect_injection(query)
            if _inj.score > 0.3:
                logger.info("Skipping fact extraction: injection score %.2f for '%s'", _inj.score, query[:80])
                _should_extract = False
        except Exception:
            pass
    if _should_extract:
        from app.core.memory import _is_explicit_user_statement
        _is_explicit = _is_explicit_user_statement(query)

        async def _safe_fact_extract():
            try:
                facts = await extract_facts_from_message(query, final_content, fact_store=svc.user_facts)
                if facts:
                    for key, fact_data in facts.items():
                        if isinstance(fact_data, dict):
                            value = fact_data.get("value", "")
                            category = fact_data.get("category", "fact")
                        else:
                            value = str(fact_data)
                            category = "fact"
                        confidence = config.FACT_CONFIDENCE_USER if _is_explicit else config.FACT_CONFIDENCE_EXTRACTED
                        await asyncio.to_thread(
                            lambda _k=key, _v=value, _c=confidence, _cat=category: svc.user_facts.set(
                                _k, _v, source="user" if _is_explicit else "extracted", confidence=_c, category=_cat
                            )
                        )
                    logger.info("Extracted %d user fact(s): %s", len(facts), list(facts.keys()))
            except Exception as e:
                logger.warning("Fact extraction failed: %s", e)
        _task = asyncio.create_task(_safe_fact_extract())
        _background_tasks.add(_task)
        _task.add_done_callback(_background_tasks.discard)

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
                    )
                )
        except Exception as e:
            logger.warning("Reflexion storage failed: %s", e)

    # --- Auto skill creation (background) ---
    if (
        config.ENABLE_AUTO_SKILL_CREATION
        and svc.skills
        and len(tool_results) >= 2
        and intent == "general"
    ):
        from app.core.auto_skills import maybe_extract_skill

        async def _safe_skill_extract(q, trs, content, skills):
            try:
                await maybe_extract_skill(q, trs, content, skills)
            except Exception as e:
                logger.warning("Auto-skill extraction failed: %s", e)
        _task = asyncio.create_task(
            _safe_skill_extract(query, tool_results, final_content, svc.skills)
        )
        _background_tasks.add(_task)
        _task.add_done_callback(_background_tasks.discard)

    # --- Curiosity: detect gaps and queue for research ---
    if config.ENABLE_CURIOSITY and svc.curiosity and intent == "general" and not is_error:
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
    if config.ENABLE_CURIOSITY and svc.curiosity and intent == "general":
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
                        logger.debug("Recurring failure check failed: %s", e)
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
    svc = get_services()

    # --- Step 0: Query length validation ---
    if len(query) > config.MAX_QUERY_LENGTH:
        yield StreamEvent(
            type=EventType.ERROR,
            data={"message": f"Query too long ({len(query)} chars). Maximum is {config.MAX_QUERY_LENGTH}."},
        )
        return

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

        # --- Step 4: Gather all context ---
        ctx = await _gather_context(svc, query, intent)

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
                chunk_size = 20
                for i in range(0, len(final_content), chunk_size):
                    yield StreamEvent(type=EventType.TOKEN, data={"text": final_content[i:i + chunk_size]})
            yield StreamEvent(type=EventType.DONE, data={"conversation_id": conversation_id, "ephemeral": True})
            return

        # --- Step 9: Refine (critique + reflexion) ---
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
            if ctx.matched_skill.steps:
                skill_success = (
                    len(gen.tool_results) > 0
                    and not gen.is_error
                    and not any(
                        isinstance(tr.get("output", ""), str)
                        and tr["output"].startswith("[Tool") and "failed" in tr["output"]
                        for tr in gen.tool_results
                    )
                )
            else:
                skill_success = not gen.is_error
            await asyncio.to_thread(svc.skills.record_use, ctx.matched_skill.id, skill_success)

        if is_new_conversation and final_content:
            try:
                title = await _generate_title(query)
                await asyncio.to_thread(svc.conversations.update_title, conversation_id, title)
            except Exception as e:
                logger.warning("Failed to generate title: %s", e)

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
                except Exception:
                    pass

    finally:
        if _conv_lock is not None and _conv_lock_acquired and _conv_lock.locked():
            _conv_lock.release()
            _conv_lock_acquired = False


_SOURCE_CONFIDENCE: dict[str, float] = {
    "Domain Study: Science": 0.75,
    "Domain Study: Technology": 0.75,
    "Domain Study: Finance": 0.70,
    "Domain Study: Current Events": 0.65,
    "World Awareness": 0.60,
    "Curiosity Research": 0.60,
}
_DEFAULT_SOURCE_CONFIDENCE = 0.65


async def _extract_kg_triples(kg, query: str, answer: str, source_name: str = "") -> None:
    """Extract (subject, predicate, object) triples from a Q&A pair.

    Runs as a background task — failures are logged, never raised.
    Includes quality gate (heuristic pre-filter) and contradiction detection.
    """
    from app.core.kg import CANONICAL_PREDICATES, is_garbage_triple

    predicates_str = ", ".join(sorted(CANONICAL_PREDICATES))
    prompt = (
        "Extract factual (subject, predicate, object) triples from this Q&A.\n"
        f"Use ONLY these predicates: {predicates_str}\n"
        "Return a JSON array. Max 5 triples. Only verifiable facts, not opinions.\n"
        "Rate each triple's confidence: 0.3 (uncertain/speculative) to 0.95 (well-established fact).\n"
        'Example: [{"subject": "python", "predicate": "created_by", "object": "guido van rossum", "confidence": 0.9}]\n\n'
        f"Q: {query}\nA: {answer[:1000]}"
    )

    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            json_prefix="[{",
        )
        if raw is None or not raw:
            return

        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict) and "triples" in data:
            data = data["triples"]
        if not isinstance(data, list):
            return

        added = 0
        for triple in data[:5]:
            if not isinstance(triple, dict):
                continue
            s = str(triple.get("subject", "")).strip()
            p = str(triple.get("predicate", "")).strip()
            o = str(triple.get("object", "")).strip()
            if not s or not p or not o or len(s) > 100 or len(o) > 100:
                continue

            # Quality gate: reject obvious garbage before storing
            if is_garbage_triple(s, p, o):
                logger.debug("KG quality gate rejected: %s %s %s", s, p, o)
                continue

            # Resolve confidence: LLM-scored with source-tiered fallback
            raw_conf = triple.get("confidence")
            if isinstance(raw_conf, (int, float)) and raw_conf > 0.0:
                conf = max(0.3, min(0.95, float(raw_conf)))
            else:
                conf = _SOURCE_CONFIDENCE.get(source_name, _DEFAULT_SOURCE_CONFIDENCE)

            # Contradiction check: resolve conflicts before adding
            try:
                safe = await kg.check_and_resolve_contradictions(s, p, o, conf)
                if not safe:
                    continue
            except Exception as e:
                logger.warning("KG contradiction check failed (allowing fact): %s", e)

            if await kg.add_fact(s, p, o, confidence=conf, source="extracted", provenance=source_name):
                added += 1

        if added:
            logger.info("KG: extracted %d triple(s) from Q&A", added)
    except Exception as e:
        logger.debug("KG extraction failed: %s", e)
