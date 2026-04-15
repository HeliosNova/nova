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


@dataclass
class DecompositionPlan:
    """Output of decompose_query() — the full orchestration plan."""
    strategy: Literal["parallel", "sequential", "map-reduce"]
    tasks: list[AgentTask]
    merge_instruction: str
    max_parallel: int = 3


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

        start = time.monotonic()
        try:
            return await asyncio.wait_for(_do(), timeout=task.timeout)
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

        # Increment structural depth so the sub-agent cannot itself decompose.
        # ContextVar.reset(token) restores the previous value on exit — correct
        # even for sequential execution within the same coroutine.
        token = _structural_depth.set(_structural_depth.get() + 1)
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
        return "[All sub-agents failed to produce results.]"

    if len(successful) == 1 and not merge_instruction:
        return successful[0].response

    # Build compact merge prompt (no full system prompt, no tools)
    parts: list[str] = []
    for r in results:
        if r.response and not r.error:
            parts.append(f"[{r.role}]\n{r.response}")
        else:
            parts.append(f"[{r.role}]\n[Agent failed: {r.error or 'no output'}]")

    merge_body = "\n\n".join(parts)
    if merge_instruction:
        merge_body += f"\n\n[Instructions]\n{merge_instruction}"

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
                            "coherent, direct response. Do not mention the agents or their roles. "
                            "Respond as if you did all the research yourself."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Original question: {query}\n\n{merge_body}",
                    },
                ],
                max_tokens=config.RESPONSE_TOKEN_BUDGET,
                temperature=config.TEMPERATURE_DEFAULT,
            ),
            timeout=float(config.INTERNAL_LLM_TIMEOUT),
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
