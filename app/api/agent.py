"""Agent API — exposes the deliberation loop.

POST /api/agent/solve  — Run the agent loop on a query, return final answer + trace.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import require_auth
from app.core.agent_loop import AgentLoop
from app.core.brain import get_services

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"], dependencies=[Depends(require_auth)])


class AgentSolveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    max_iterations: int = Field(default=10, ge=1, le=20)
    sample_n: int = Field(default=1, ge=1, le=5, description="Self-consistency: sample N reasoning chains")
    use_tools: bool = Field(default=True, description="If false, run pure reasoning (no tool calls)")


class AgentStepTrace(BaseModel):
    id: int
    description: str
    status: str
    action: dict | None = None
    observation: str | None = None
    critique: str | None = None
    attempts: int = 0


class AgentSolveResponse(BaseModel):
    query: str
    answer: str
    success: bool
    iterations: int
    duration_seconds: float
    plan: list[AgentStepTrace]
    scratchpad: str
    findings: dict[str, str] = {}


@router.post("/agent/solve", response_model=AgentSolveResponse)
async def agent_solve(request: AgentSolveRequest):
    """Run the deliberation loop: plan → act → observe → critique → revise → synthesize."""
    svc = get_services()
    tools = svc.tool_registry if request.use_tools else None
    n_tools = len(tools._tools) if tools and hasattr(tools, "_tools") else 0
    logger.info("agent.solve: use_tools=%s, registry=%s, n_tools=%d", request.use_tools, type(tools).__name__, n_tools)

    loop = AgentLoop(tools=tools)
    try:
        result = await loop.solve(
            query=request.query,
            max_iterations=request.max_iterations,
            sample_n=request.sample_n,
        )
    except Exception as e:
        logger.exception("agent.solve failed")
        raise HTTPException(status_code=500, detail=f"agent failure: {e}")

    # Pull findings out of scratchpad text. Renderer format is:
    #   "  - <key>: <value>"  where <key> is snake_case (may contain ":" prefix
    # like "prior:foo"). Use a regex so "prior:south_korea_name: Republic of
    # Korea" parses to key="prior:south_korea_name" + value="Republic of Korea"
    # instead of being split at the first colon.
    import re as _re
    findings = {}
    if "FINDINGS:" in result.scratchpad_text:
        chunk = result.scratchpad_text.split("FINDINGS:", 1)[1]
        finding_re = _re.compile(r"^-\s+([a-zA-Z0-9_:]+):\s*(.*)$")
        for line in chunk.splitlines():
            m = finding_re.match(line.strip())
            if m:
                findings[m.group(1)] = m.group(2).strip()

    return AgentSolveResponse(
        query=result.query,
        answer=result.answer,
        success=result.success,
        iterations=result.iterations,
        duration_seconds=round(result.duration_seconds, 2),
        plan=[
            AgentStepTrace(
                id=s.id,
                description=s.description,
                status=s.status,
                action=s.action,
                observation=(s.observation or "")[:1500] if s.observation else None,
                critique=s.critique,
                attempts=s.attempts,
            )
            for s in result.plan.steps
        ],
        scratchpad=result.scratchpad_text,
        findings=findings,
    )
