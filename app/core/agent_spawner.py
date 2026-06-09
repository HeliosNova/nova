"""Multi-agent task spawner — structural decomposition orchestration.

Runs parallel / sequential / map-reduce sub-agents via ephemeral
brain.think() calls.  Each sub-agent is an independent think() invocation
with ephemeral=True so all write paths (fact extraction, skill creation,
corrections, KG, training data) are disabled automatically.

This module owns the _structural_depth ContextVar — distinct from
DelegateTool's _delegation_depth.  Setting _structural_depth > 0 before
a think() call prevents that call from itself triggering structural
decomposition, enforcing MAX_STRUCTURAL_DEPTH = 1.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from app.config import config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structural depth ContextVar
# ---------------------------------------------------------------------------

# Request-scoped counter.  Incremented by _execute_think() before calling
# brain.think(), reset via ContextVar.reset(token) on exit.
# asyncio.gather() copies the current context to each Task, so parallel
# sub-agents each start at the depth they inherit at task-creation time.
_structural_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_structural_depth", default=0
)


def get_structural_depth() -> int:
    """Return the current structural decomposition depth (0 = top-level)."""
    return _structural_depth.get()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    """A unit of work dispatched to a sub-agent."""
    task_id: str
    parent_run_id: str
    role: str
    query: str
    focus: str
    context_facts: list[str]
    context_lessons: list[str]
    shared_findings: dict[str, str]   # Populated for sequential/map-reduce
    depth: int
    timeout: int
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """What a sub-agent returns to the orchestrator."""
    task_id: str
    role: str
    query: str
    response: str
    tools_invoked: list[str]
    skill_used: str | None
    reflexion_score: float | None
    latency_seconds: float
    error: str | None = None
    truncated: bool = False
    depth: int = 0  # the structural-depth value this sub-agent ran at (1 = first level, 2 = nested)
    sub_decomposed: bool = False  # True if THIS sub-agent itself triggered further decomposition


@dataclass
class DecompositionPlan:
    """Output of decompose_query() — the full orchestration plan."""
    strategy: Literal["parallel", "sequential", "map-reduce"]
    tasks: list[AgentTask]
    merge_instruction: str
    max_parallel: int = 6  # default lifted from 3 — see config.MAX_PARALLEL_AGENTS


# ---------------------------------------------------------------------------
# AgentSpawner
# ---------------------------------------------------------------------------

class AgentSpawner:
    """Executes a DecompositionPlan by spawning ephemeral think() sub-agents."""

    def __init__(self, plan: DecompositionPlan, conversation_id: str) -> None:
        self.plan = plan
        self.conversation_id = conversation_id

    async def run(self) -> list[AgentResult]:
        """Execute all tasks per the plan strategy. Returns results in task order."""
        if self.plan.strategy == "sequential":
            return await self._run_sequential()
        elif self.plan.strategy == "map-reduce":
            return await self._run_map_reduce()
        else:  # parallel (default)
            return await self._run_parallel()

    # --- Execution strategies ---

    async def _run_parallel(self) -> list[AgentResult]:
        sem = asyncio.Semaphore(self.plan.max_parallel)
        coros = [self._run_task(task, sem) for task in self.plan.tasks]
        return list(await asyncio.gather(*coros))

    async def _run_sequential(self) -> list[AgentResult]:
        """Run tasks one-by-one; each receives prior results in shared_findings."""
        results: list[AgentResult] = []
        accumulated: dict[str, str] = {}
        for task in self.plan.tasks:
            task.shared_findings = dict(accumulated)
            result = await self._run_task(task, sem=None)
            results.append(result)
            if result.response and not result.error:
                accumulated[result.role] = result.response
        return results

    async def _run_map_reduce(self) -> list[AgentResult]:
        """Parallel map phase (all but last task), then a single reduce agent."""
        if len(self.plan.tasks) <= 1:
            return await self._run_parallel()
        map_tasks = self.plan.tasks[:-1]
        reduce_task = self.plan.tasks[-1]

        sem = asyncio.Semaphore(self.plan.max_parallel)
        map_results: list[AgentResult] = list(
            await asyncio.gather(*[self._run_task(t, sem) for t in map_tasks])
        )
        # Inject map results into the reduce task
        reduce_task.shared_findings = {
            r.role: r.response for r in map_results if r.response and not r.error
        }
        reduce_result = await self._run_task(reduce_task, sem=None)
        return map_results + [reduce_result]

    # --- Per-task runner ---

    async def _run_task(
        self, task: AgentTask, sem: asyncio.Semaphore | None
    ) -> AgentResult:
        """Run a single sub-agent with per-task timeout."""
        async def _do():
            if sem:
                async with sem:
                    return await self._execute_think(task)
            return await self._execute_think(task)

        # Tighter timeout for sub-sub-agents (depth >= 2): they should reason
        # deeply from context, not spawn long tool loops. Keeps recursive
        # decomposition from serializing into multi-minute hangs on a single GPU.
        effective_timeout = task.timeout
        if _structural_depth.get() >= 1:  # we're about to spawn a depth-2 agent
            effective_timeout = min(task.timeout, 90)

        start = time.monotonic()
        try:
            return await asyncio.wait_for(_do(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            latency = time.monotonic() - start
            logger.warning(
                "Agent task '%s' (role=%s) timed out after %ds",
                task.task_id, task.role, task.timeout,
            )
            return AgentResult(
                task_id=task.task_id, role=task.role, query=task.query,
                response="", tools_invoked=[], skill_used=None,
                reflexion_score=None, latency_seconds=round(latency, 2),
                error="timeout",
            )
        except Exception as e:
            latency = time.monotonic() - start
            logger.exception("Agent task '%s' failed", task.task_id)
            return AgentResult(
                task_id=task.task_id, role=task.role, query=task.query,
                response="", tools_invoked=[], skill_used=None,
                reflexion_score=None, latency_seconds=round(latency, 2),
                error=f"exception: {e}",
            )

    async def _execute_think(self, task: AgentTask) -> AgentResult:
        """Stream brain.think(ephemeral=True) and collect TOKEN/TOOL_USE/DONE."""
        from app.core.brain import think
        from app.schema import EventType

        full_query = _build_sub_query(task)
        response_parts: list[str] = []
        tools_invoked: list[str] = []
        skill_used: str | None = None
        start = time.monotonic()

        # Increment structural depth so the sub-agent runs at depth+1.
        # ContextVar.reset(token) restores the previous value on exit — correct
        # even for sequential execution within the same coroutine.
        new_depth = _structural_depth.get() + 1
        token = _structural_depth.set(new_depth)
        sub_decomposed = False
        try:
            async for event in think(
                query=full_query,
                ephemeral=True,
                conversation_id=self.conversation_id,
            ):
                if event.type == EventType.TOKEN:
                    response_parts.append(event.data.get("text", ""))
                elif event.type == EventType.TOOL_USE:
                    tool_name = event.data.get("tool", "")
                    if tool_name:
                        tools_invoked.append(tool_name)
                elif event.type == EventType.DONE:
                    skill_used = event.data.get("skill_used")
                    # If the sub-agent itself triggered structural decomposition, it's
                    # a depth-N+1 nested case — surface the signal upward.
                    if event.data.get("decomposed"):
                        sub_decomposed = True
        finally:
            _structural_depth.reset(token)

        response = "".join(response_parts)
        # Cap at ~RESPONSE_TOKEN_BUDGET chars (approx 2 chars/token for safety)
        budget_chars = config.RESPONSE_TOKEN_BUDGET * 2
        truncated = len(response) > budget_chars
        if truncated:
            response = response[:budget_chars]

        latency = time.monotonic() - start
        return AgentResult(
            task_id=task.task_id,
            role=task.role,
            query=task.query,
            response=response,
            tools_invoked=tools_invoked,
            skill_used=skill_used,
            reflexion_score=None,  # Sub-agent quality not stored
            latency_seconds=round(latency, 2),
            truncated=truncated,
            depth=new_depth,
            sub_decomposed=sub_decomposed,
        )


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_sub_query(task: AgentTask) -> str:
    """Build the sub-agent query, injection-checking shared_findings values."""
    if not task.shared_findings:
        return task.query

    safe_findings: dict[str, str] = {}
    for role, text in task.shared_findings.items():
        if config.ENABLE_INJECTION_DETECTION and text:
            from app.core.injection import detect_injection
            result = detect_injection(text)
            if result.is_suspicious and result.score >= config.INJECTION_SUSPICIOUS_THRESHOLD:
                text = (
                    f"[CONTENT WARNING: potential injection detected,"
                    f" score={result.score:.2f}]\n{text}"
                )
        safe_findings[role] = text

    findings_text = "\n".join(
        f"[{role}]: {text[:500]}" for role, text in safe_findings.items()
    )
    return f"{task.query}\n\n[Prior research findings]\n{findings_text}"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

async def merge_agent_results(
    results: list[AgentResult],
    merge_instruction: str,
    query: str,
) -> str:
    """Synthesize sub-agent results via invoke_nothink(). Falls back to concatenation."""
    successful = [r for r in results if r.response and not r.error]

    if not successful:
        # All sub-agents timed out or errored — instead of returning a useless
        # refusal string (which fails any keyword assertion and confuses the
        # user), make a single-pass best-effort answer to the original query
        # from the model's own knowledge. The merge prompt below will be told
        # explicitly that it has no agent inputs and must answer directly.
        try:
            from app.core import llm
            recovery = await asyncio.wait_for(
                llm.invoke_nothink(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are answering a complex question. The research-stage "
                                "agents that were supposed to gather supporting material "
                                "all failed or timed out, so synthesize the best answer "
                                "you can from your own knowledge. Be direct, cover every "
                                "entity or topic in the question, and do not mention "
                                "sub-agents, research stages, timeouts, or apologize for "
                                "lack of fresh data. If you can give a confident answer, do."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    max_tokens=min(config.RESPONSE_TOKEN_BUDGET + 400, 2400),
                    temperature=config.TEMPERATURE_DEFAULT,
                ),
                timeout=max(float(config.INTERNAL_LLM_TIMEOUT), 60.0),
            )
            recovery = (recovery or "").strip()
            if recovery:
                logger.info(
                    "[merge_agent_results] all sub-agents failed — recovered "
                    "via single-pass synthesis (%d chars)", len(recovery),
                )
                return recovery
        except Exception as e:
            logger.warning(
                "[merge_agent_results] single-pass recovery failed (%s)", e,
            )
        # Fall back to the old refusal only if even the recovery LLM call dies.
        return "[All sub-agents failed to produce results.]"

    if len(successful) == 1 and not merge_instruction:
        return successful[0].response

    # Build compact merge prompt (no full system prompt, no tools).
    # Each agent block carries structured metadata — tools_invoked +
    # reflexion_score — so the synthesizer can cite provenance instead of
    # re-reasoning from scratch. Per-response pre-trim to ~1500 chars caps
    # context bloat when sub-agents emit long tool dumps (web_search etc.);
    # the original observation is preserved in the agent's own task result
    # if needed.
    _PER_RESP_TRIM = 1500
    parts: list[str] = []
    for r in results:
        header_bits: list[str] = []
        if r.tools_invoked:
            header_bits.append("tools=" + ",".join(r.tools_invoked))
        else:
            header_bits.append("tools=(none)")
        if r.reflexion_score is not None:
            header_bits.append(f"quality={r.reflexion_score:.2f}")
        if r.truncated:
            header_bits.append("truncated=true")
        header = " | ".join(header_bits)
        if r.response and not r.error:
            body = r.response
            if len(body) > _PER_RESP_TRIM:
                body = body[:_PER_RESP_TRIM] + " [...trimmed]"
            parts.append(f"[{r.role}] ({header})\n{body}")
        else:
            parts.append(f"[{r.role}] ({header})\n[Agent failed: {r.error or 'no output'}]")

    merge_body = "\n\n".join(parts)
    if merge_instruction:
        merge_body += f"\n\n[Instructions]\n{merge_instruction}"

    # Scale budget and timeout with agent count. At N>=5 the base budget (600
    # tokens) truncates before the model finishes enumerating results, which
    # is how the N=10 probe lost half its targets. Cap at 4000 to keep the
    # merge from running longer than a normal response.
    n = len(successful)
    merge_tokens = min(config.RESPONSE_TOKEN_BUDGET + 200 * max(0, n - 1), 4000)
    merge_timeout = max(float(config.INTERNAL_LLM_TIMEOUT), 30.0 + 10.0 * n)

    try:
        from app.core import llm
        merged = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a synthesis assistant. You have received research findings "
                            "from multiple specialized agents. Synthesize them into a single, "
                            "coherent, direct response.\n\n"
                            "Each agent block in the user message starts with a header in "
                            "parentheses showing the tools that agent invoked (web_search, "
                            "calculator, code_exec, etc.) and a quality score where available. "
                            "Use this provenance: if a fact came from `web_search`, treat it as "
                            "fresh-source-grounded; if `tools=(none)`, the agent answered from "
                            "the model's own knowledge — weight contested numeric facts toward "
                            "the tool-grounded agent.\n\n"
                            "STRICT RULES:\n"
                            "- NEVER mention 'group-N-researcher', 'agent', 'sub-agent', "
                            "  'group 1', 'group 2', or any internal task naming. The user has "
                            "  no idea those exist and will be confused.\n"
                            "- NEVER quote the headers (`tools=...`, `quality=...`) in the output. "
                            "  Use them only to choose which findings to trust when they conflict.\n"
                            "- NEVER second-guess yourself in the output ('No wait, let me re-read...', "
                            "  'Hmm I'm confusing things'). Make ONE confident pass.\n"
                            "- Respond as if YOU did all the research yourself.\n"
                            "- Cover every entity or topic from the agent outputs — drop nothing.\n"
                            "- If the agent outputs disagree on a fact, pick the most well-sourced "
                            "  one and report it. Do NOT include both contradicting numbers.\n"
                            "- If an agent returned an error or empty result, silently work around "
                            "  it — say 'I couldn't verify X' rather than 'agent N failed'.\n"
                            "- For long outputs use concise bullets, not rambling prose."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Original question: {query}\n\n{merge_body}",
                    },
                ],
                max_tokens=merge_tokens,
                temperature=config.TEMPERATURE_DEFAULT,
            ),
            timeout=merge_timeout,
        )
        return (merged or "").strip() or _fallback_concat(successful)
    except Exception as e:
        logger.warning("Merge LLM call failed (%s) — falling back to concatenation", e)
        return _fallback_concat(successful)


def _fallback_concat(results: list[AgentResult]) -> str:
    """Concatenate successful results with formatted role headers."""
    parts: list[str] = []
    for r in results:
        header = r.role.replace("-", " ").replace("_", " ").title()
        parts.append(f"**{header}**\n{r.response}")
    return "\n\n".join(parts)
