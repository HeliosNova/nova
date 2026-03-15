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
from app.core.learning import is_likely_correction
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

async def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool via the registry. Falls back to placeholder if no registry."""
    from app.tools.base import format_tool_error
    svc = get_services()
    if svc.tool_registry:
        timeout = float(config.TOOL_TIMEOUT)
        try:
            return await asyncio.wait_for(
                svc.tool_registry.execute(tool_name, args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Tool '%s' timed out after %ds", tool_name, config.TOOL_TIMEOUT)
            return format_tool_error(tool_name, f"Timed out after {config.TOOL_TIMEOUT} seconds", retriable=True)
        except Exception as e:
            logger.exception("Tool '%s' failed with exception", tool_name)
            return format_tool_error(tool_name, f"Failed: {e}", retriable=True)
    return format_tool_error(tool_name, "Not yet available")


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

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for English, ~1.5 chars/token for CJK."""
    if not text:
        return 0
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    non_cjk = len(text) - cjk_count
    return non_cjk // 4 + int(cjk_count / 1.5)


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
    response_budget = 600  # Reserve tokens for the response

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
                            "Example: 'Discussed X. User prefers Y. Corrected Z.'"
                        ),
                    },
                    {"role": "user", "content": summary_input},
                ],
                max_tokens=150,
                temperature=0.1,
            ),
            timeout=30,
        )
        logger.info(
            "Context managed: %d→%d messages, summary=%d chars (budget: %d/%d tokens)",
            len(history), keep, len(summary),
            system_tokens + sum(_estimate_tokens(m.get("content", "")) for m in recent_messages) + query_tokens,
            config.MAX_CONTEXT_TOKENS,
        )
        return recent_messages, summary.strip()
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
                    timeout=30,
                )
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
    result = await llm.invoke_nothink(
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
    )
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
    re.compile(r"<think>.*?</think>", re.DOTALL),
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
    reflexions_text: str = ""
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
            kg_facts = await asyncio.to_thread(svc.kg.get_relevant_facts, query, 8)
            if kg_facts:
                ctx.kg_facts_text = svc.kg.format_for_prompt(kg_facts)
        except Exception as e:
            logger.warning("KG retrieval failed: %s", e)

    # --- Reflexions (past failure warnings) ---
    if svc.reflexions:
        try:
            reflexions = await asyncio.to_thread(svc.reflexions.get_relevant, query, 3)
            if reflexions:
                ctx.reflexions_text = svc.reflexions.format_for_prompt(reflexions)
        except Exception as e:
            logger.warning("Reflexion retrieval failed: %s", e)

    # --- Success patterns (what worked before) ---
    if svc.reflexions:
        try:
            from app.core.reflexion import ReflexionStore
            successes = await asyncio.to_thread(svc.reflexions.get_success_patterns, query, 2)
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
                    source = chunk.title or chunk.source or "document"
                    score = chunk.score if hasattr(chunk, "score") and chunk.score is not None else 0.0
                    relevance = "high relevance" if score >= 0.7 else ("moderate" if score >= 0.4 else "low relevance")
                    lines.append(f"[{i}] ({relevance} | Source: {source})\n{chunk.content}")
                    ctx.retrieved_sources.append({
                        "title": chunk.title or "",
                        "source": chunk.source or "",
                        "score": round(score, 4),
                    })
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
    system_prompt = build_system_prompt(
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
    )

    # Context window management
    managed_history, conversation_summary = await _manage_context(
        system_prompt, history, query
    )

    if conversation_summary:
        system_prompt = build_system_prompt(
            user_facts_text=ctx.user_facts_text,
            lessons_text=ctx.lessons_text,
            tool_descriptions=_get_tool_descriptions(),
            retrieved_context=ctx.retrieved_context,
            skills_text=ctx.skills_text,
            conversation_summary=conversation_summary,
            kg_facts=ctx.kg_facts_text,
            reflexions=ctx.reflexions_text,
            integrations_text=ctx.integrations_text,
            success_patterns=ctx.success_patterns_text,
            external_skills_text=ctx.external_skills_text,
            matched_external_skill_text=ctx.matched_external_skill_text,
        )
        history = managed_history

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
    if image:
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
    # Skip caching for tools with side effects
    _SIDE_EFFECT_TOOLS = frozenset({
        "file_ops", "email_send", "webhook", "calendar", "reminder",
        "shell_exec", "code_exec", "integration", "delegate", "browser",
        "desktop", "background_task", "tool_create", "monitor",
    })
    _tool_cache: dict[tuple, str] = {}

    try:
        for tool_round in range(config.MAX_TOOL_ROUNDS):
            if use_thinking:
                thinking_buf = ""
                content_buf = ""
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
                except TimeoutError:
                    logger.warning("Streaming generation timed out after %.0fs", _GENERATION_TIMEOUT)
                    if content_buf:
                        content_buf += "\n\n[Response truncated due to timeout]"
                    else:
                        content_buf = "The response timed out. Please try a simpler query or try again."
                content_buf = llm._strip_think_tags(content_buf).strip()
                result = llm.GenerationResult(
                    content=content_buf,
                    tool_call=None,
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
                    result = llm.GenerationResult(
                        content="The response timed out. Please try a simpler query or try again.",
                        tool_call=None,
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
            if result.tool_call is not None:
                tool_calls = [result.tool_call]
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

            for tc in tool_calls:
                yield StreamEvent(
                    type=EventType.TOOL_USE,
                    data={"tool": tc.tool, "args": tc.args, "status": "executing"},
                )

            # Execute ALL tool calls concurrently (with per-conversation cache)
            async def _run_tool(tc):
                if tc.tool == "tool_create" and svc.custom_tools:
                    return tc, await _handle_tool_create(svc, tc.args)
                # Cache lookup for idempotent tools
                if tc.tool not in _SIDE_EFFECT_TOOLS:
                    try:
                        cache_key = (tc.tool, frozenset(tc.args.items()))
                    except TypeError:
                        cache_key = None  # unhashable args
                    if cache_key and cache_key in _tool_cache:
                        logger.debug("Tool cache hit: %s", tc.tool)
                        return tc, _tool_cache[cache_key]
                else:
                    cache_key = None
                output = await _execute_tool(tc.tool, tc.args)
                # Sanitize tool outputs not already handled by the tool itself
                _SELF_SANITIZING_TOOLS = {"web_search", "http_fetch", "browser", "knowledge_search"}
                if (config.ENABLE_INJECTION_DETECTION
                        and tc.tool not in _SELF_SANITIZING_TOOLS
                        and not tc.tool.startswith("mcp_")):
                    from app.core.injection import sanitize_content
                    output = sanitize_content(output, context=f"tool:{tc.tool}")
                if cache_key is not None:
                    _tool_cache[cache_key] = output
                return tc, output

            results = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])

            assistant_content = result.content or f'[Calling tool: {tool_calls[0].tool}]'
            messages.append({"role": "assistant", "content": assistant_content})

            for tc, tool_output in results:
                gen.tool_results.append({
                    "tool": tc.tool,
                    "args": tc.args,
                    "output": tool_output[:2000],
                })
                yield StreamEvent(
                    type=EventType.TOOL_USE,
                    data={"tool": tc.tool, "result": tool_output[:500], "status": "complete"},
                )
                # Use "user" role with structured prefix instead of "system" for cleaner role tracking
                # Truncate to prevent context window blow-up from large tool outputs
                messages.append({
                    "role": "user",
                    "content": f"[Tool Result for '{tc.tool}']:\n{tool_output[:4000]}\n\nUse this result to answer the original question.",
                })

                if not ephemeral:
                    await asyncio.to_thread(
                        lambda _tc=tc, _out=tool_output: svc.conversations.add_message(
                            conversation_id, "tool", _out[:2000], tool_name=_tc.tool
                        )
                    )

        else:
            # Exhausted tool rounds
            if not gen.final_content:
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
) -> tuple[str, float | None, str]:
    """Multi-round critique, plan coverage check, reflexion LLM critique.

    This is Steps 10b–10d of the original think() pipeline.
    Returns: (refined_content, reflexion_quality, reflexion_reason)
    """
    # --- Self-Critique (multi-round) ---
    if config.ENABLE_CRITIQUE and final_content and intent == "general":
        from app.core.critique import should_critique, critique_answer, format_critique_for_regeneration
        if should_critique(query, final_content, intent, tool_results, was_planned):
            for critique_round in range(config.MAX_CRITIQUE_ROUNDS):
                try:
                    critique = await critique_answer(query, final_content, sources=retrieved_context)
                    if not critique or critique.get("pass", True):
                        break
                    logger.info("Critique round %d failed: %s", critique_round + 1, critique.get("issues", []))
                    # Log critique failures as reflexions for future learning
                    _svc = get_services()
                    if _svc.reflexions and critique.get("issues"):
                        try:
                            issues_text = "; ".join(critique["issues"])
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
                    critique_msg = format_critique_for_regeneration(critique)
                    if not critique_msg:
                        break
                    messages.append({"role": "assistant", "content": final_content})
                    messages.append({"role": "system", "content": critique_msg})
                    try:
                        retry_result = await llm.generate_with_tools(messages, tools)
                        if retry_result.content and retry_result.tool_call is None:
                            final_content = retry_result.content
                            logger.info("Regenerated after critique round %d (%d chars)", critique_round + 1, len(final_content))
                        else:
                            break
                    except Exception as e:
                        logger.warning("Critique regeneration failed (round %d): %s", critique_round + 1, e)
                        break
                except Exception as e:
                    logger.warning("Critique failed (round %d): %s", critique_round + 1, e)
                    break

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
                    if retry_result.content and retry_result.tool_call is None:
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
                reflexion_quality, reflexion_reason = await critique_response(query, final_content, tool_results)
            else:
                reflexion_quality, reflexion_reason = assess_quality(final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query)

            if reflexion_quality is not None and reflexion_quality < 0.3 and reflexion_reason:
                logger.info("Reflexion critique flagged (%.2f): %s", reflexion_quality, reflexion_reason)
                regen_msg = (
                    f"[Minor quality note (score: {reflexion_quality})]\n"
                    f"Note: {reflexion_reason}\n"
                    "Adjust your answer if needed. Keep all correct information intact."
                )
                messages.append({"role": "assistant", "content": final_content})
                messages.append({"role": "system", "content": regen_msg})
                try:
                    retry_result = await llm.generate_with_tools(messages, tools)
                    if retry_result.content and retry_result.tool_call is None:
                        final_content = retry_result.content
                        reflexion_quality, reflexion_reason = assess_quality(
                            final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query
                        )
                        logger.info("Regenerated after reflexion critique (%d chars, new score: %.2f)",
                                    len(final_content), reflexion_quality)
                except Exception as e:
                    logger.warning("Reflexion critique regeneration failed: %s", e)
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
        assistant_skip = 1
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
                if msg.content != query:
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
                    lesson_id = await asyncio.to_thread(svc.learning.save_lesson, correction)

                    dpo_query = original_query or query
                    dpo_rejected = prev_answer[:1000]
                    dpo_chosen = correction.correct_answer or correction.lesson_text

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
    # Run on signal match (high confidence) OR any message >50 chars (background, LLM returns {} if nothing)
    _should_extract = svc.user_facts and not is_error and (has_fact_signals(query) or len(query) > 50)
    if _should_extract:
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
                        # Tiered confidence: signal match = 0.8, background extraction = 0.5
                        confidence = 0.8 if has_fact_signals(query) else 0.5
                        await asyncio.to_thread(
                            lambda _k=key, _v=value, _c=confidence, _cat=category: svc.user_facts.set(
                                _k, _v, source="extracted", confidence=_c, category=_cat
                            )
                        )
                    logger.info("Extracted %d user fact(s): %s", len(facts), list(facts.keys()))
            except Exception as e:
                logger.warning("Fact extraction failed: %s", e)
        if has_fact_signals(query):
            await _safe_fact_extract()
        else:
            _task = asyncio.create_task(_safe_fact_extract())
            _background_tasks.add(_task)
            _task.add_done_callback(_background_tasks.discard)

    # --- KG triple extraction (background, with contradiction check) ---
    # Only extract from monitor queries (is_monitor=True) to prevent untrusted user input
    # from poisoning the KG (OWASP ASI06)
    pass  # KG extraction gated — see heartbeat monitors for extraction

    # --- Reflexion — store failures AND high-quality successes ---
    if svc.reflexions and intent == "general" and final_content:
        try:
            if reflexion_quality is not None:
                quality, reason = reflexion_quality, reflexion_reason
            else:
                from app.core.reflexion import assess_quality
                quality, reason = assess_quality(final_content, tool_results, config.MAX_TOOL_ROUNDS, query=query)
            tools_used = [tr["tool"] for tr in tool_results]
            if quality < 0.6 and reason:
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
            elif quality >= 0.8 and tool_results:
                await asyncio.to_thread(
                    lambda: svc.reflexions.store(
                        task_summary=query[:500],
                        outcome="success",
                        reflection=f"Good result (quality={quality:.2f})",
                        quality_score=quality,
                        tools_used=tools_used,
                        revision_count=len(tool_results),
                    )
                )
        except Exception as e:
            logger.debug("Reflexion storage failed: %s", e)

    # --- Auto skill creation (background) ---
    if (
        config.ENABLE_AUTO_SKILL_CREATION
        and svc.skills
        and len(tool_results) >= 3
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
                await asyncio.to_thread(
                    lambda: svc.curiosity.add(query[:200], source="reflexion_failure", urgency=0.7)
                )
        except Exception as e:
            logger.debug("Curiosity failure queueing failed: %s", e)

    # --- Reflexion-to-Action: promote recurring failures to lessons ---
    if svc.reflexions and svc.learning and intent == "general" and final_content:
        try:
            if reflexion_quality is not None and reflexion_quality < 0.6:
                async def _safe_check_recurring(reflexions, task_summary, learning):
                    try:
                        from app.core.reflexion import check_recurring_failures
                        await check_recurring_failures(task_summary, learning)
                    except Exception as e:
                        logger.debug("Recurring failure check failed: %s", e)
                _task = asyncio.create_task(
                    _safe_check_recurring(svc.reflexions, query[:500], svc.learning)
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
        tools.append({"name": "tool_create"})
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
    )

    # --- Step 10: Emit sources + stream tokens ---
    if ctx.retrieved_sources:
        yield StreamEvent(type=EventType.SOURCES, data={"sources": ctx.retrieved_sources})

    if final_content:
        final_content = _sanitize_answer(final_content)
        chunk_size = 20
        for i in range(0, len(final_content), chunk_size):
            yield StreamEvent(type=EventType.TOKEN, data={"text": final_content[i:i + chunk_size]})

    # --- Step 11: Save assistant message + skill usage + title ---
    # Guard against None content from LLM (would cause IntegrityError on NOT NULL column)
    if final_content is None:
        final_content = ""
    await asyncio.to_thread(
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
                    tr.get("output", "").startswith("[Tool") and "failed" in tr.get("output", "")
                    for tr in gen.tool_results
                )
            )
        else:
            skill_success = True
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


async def _extract_kg_triples(kg, query: str, answer: str) -> None:
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
        'Example: [{"subject": "python", "predicate": "created_by", "object": "guido van rossum"}]\n\n'
        f"Q: {query}\nA: {answer[:1000]}"
    )

    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            json_prefix="[{",
        )
        if not raw:
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

            # Contradiction check: resolve conflicts before adding
            try:
                safe = await kg.check_and_resolve_contradictions(s, p, o, 0.7)
                if not safe:
                    continue
            except Exception:
                pass  # On failure, allow the fact through

            if kg.add_fact(s, p, o, confidence=0.7, source="extracted"):
                added += 1

        if added:
            logger.info("KG: extracted %d triple(s) from Q&A", added)
    except Exception as e:
        logger.debug("KG extraction failed: %s", e)
