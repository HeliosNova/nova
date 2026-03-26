"""System endpoints — health check, status, export/import, integrations, access tier, config."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse

from app import __version__
from app.auth import require_auth
from app.config import config, _MUTABLE_FIELDS
from app.database import get_db
from app.schema import HealthResponse, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — minimal status for unauthenticated monitoring."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Provider Health
# ---------------------------------------------------------------------------

@router.get("/providers/health", dependencies=[Depends(require_auth)])
async def providers_health():
    """Check health of all configured LLM providers.

    Returns a map like {"ollama": "healthy", "openai": "unavailable"}.
    Only checks providers that have credentials/URLs configured.
    """
    import asyncio

    results: dict[str, str] = {}

    # Always check the active provider
    from app.core.llm import get_provider
    active_provider = get_provider()
    active_name = config.LLM_PROVIDER

    # Build list of configured providers to check
    providers_to_check: dict[str, object] = {}

    # Ollama: always configured (has default URL)
    if config.OLLAMA_URL:
        if active_name == "ollama":
            providers_to_check["ollama"] = active_provider
        else:
            from app.core.providers.ollama import OllamaProvider
            providers_to_check["ollama"] = OllamaProvider(base_url=config.OLLAMA_URL)

    # OpenAI: configured if API key is set
    if config.OPENAI_API_KEY:
        if active_name == "openai":
            providers_to_check["openai"] = active_provider
        else:
            from app.core.providers.openai import OpenAIProvider
            providers_to_check["openai"] = OpenAIProvider(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)

    # Anthropic: configured if API key is set
    if config.ANTHROPIC_API_KEY:
        if active_name == "anthropic":
            providers_to_check["anthropic"] = active_provider
        else:
            from app.core.providers.anthropic import AnthropicProvider
            providers_to_check["anthropic"] = AnthropicProvider(api_key=config.ANTHROPIC_API_KEY, model=config.ANTHROPIC_MODEL)

    # Google: configured if API key is set
    if config.GOOGLE_API_KEY:
        if active_name == "google":
            providers_to_check["google"] = active_provider
        else:
            from app.core.providers.google import GoogleProvider
            providers_to_check["google"] = GoogleProvider(api_key=config.GOOGLE_API_KEY, model=config.GOOGLE_MODEL)

    if not providers_to_check:
        return {"active": active_name, "providers": {"ollama": "not_configured"}}

    # Run health checks concurrently with a timeout
    async def _check(name: str, provider) -> tuple[str, str]:
        try:
            healthy = await asyncio.wait_for(provider.check_health(), timeout=10.0)
            return name, "healthy" if healthy else "unavailable"
        except asyncio.TimeoutError:
            return name, "timeout"
        except Exception:
            return name, "unavailable"

    tasks = [_check(name, prov) for name, prov in providers_to_check.items()]
    check_results = await asyncio.gather(*tasks)

    # Clean up temporary provider clients (not the active one)
    for name, prov in providers_to_check.items():
        if prov is not active_provider and hasattr(prov, "close"):
            try:
                await prov.close()
            except Exception:
                pass

    for name, status in check_results:
        results[name] = status

    return {"active": active_name, "providers": results}


_ALLOWED_TABLES = frozenset({
    "conversations", "messages", "user_facts", "lessons", "skills", "documents",
    "kg_facts", "reflexions", "custom_tools",
})

# Pre-built SQL queries — no f-string interpolation
_TABLE_QUERIES = {t: f"SELECT COUNT(*) as c FROM {t}" for t in _ALLOWED_TABLES}


@router.get("/status", response_model=StatusResponse, dependencies=[Depends(require_auth)])
async def status() -> StatusResponse:
    """System status — counts of key entities."""
    db = get_db()

    def count(table: str) -> int:
        if table not in _TABLE_QUERIES:
            raise ValueError(f"Table '{table}' is not allowed")
        row = db.fetchone(_TABLE_QUERIES[table])
        return row["c"] if row else 0

    training_examples = 0
    p = Path(config.TRAINING_DATA_PATH)
    if p.exists():
        with p.open() as f:
            training_examples = sum(1 for _ in f)

    return StatusResponse(
        conversations=count("conversations"),
        messages=count("messages"),
        user_facts=count("user_facts"),
        lessons=count("lessons"),
        skills=count("skills"),
        documents=count("documents"),
        training_examples=training_examples,
        kg_facts=count("kg_facts"),
        reflexions=count("reflexions"),
        custom_tools=count("custom_tools"),
    )


# ---------------------------------------------------------------------------
# Integrations / Access Tier / Config Summary
# ---------------------------------------------------------------------------

@router.get("/integrations", dependencies=[Depends(require_auth)])
async def get_integrations():
    """Return all integration templates with configuration status."""
    from app.tools.integration import _registry as integration_registry
    if integration_registry is None:
        return []
    return [
        {
            "name": i.name,
            "auth_type": i.auth_type,
            "auth_env_var": i.auth_env_var,
            "is_configured": i.is_configured,
            "endpoint_count": len(i.endpoints),
            "description": i.description,
        }
        for i in integration_registry.get_all()
    ]


@router.get("/access-tier", dependencies=[Depends(require_auth)])
async def get_access_tier():
    """Return the current access tier and its restrictions."""
    from app.core import access_tiers
    tier = access_tiers._tier()
    descriptions = {
        "sandboxed": "Restricted mode — limited shell commands, no file writes outside workspace",
        "standard": "Standard mode — most tools available, some dangerous commands blocked",
        "full": "Full access — all tools and commands available",
        "none": "No restrictions — sandbox fully disabled, all commands and imports allowed",
    }
    blocked = access_tiers.get_blocked_shell_commands()
    blocked_imports = access_tiers.get_blocked_imports()
    return {
        "tier": tier,
        "description": descriptions.get(tier, "Unknown tier"),
        "blocked_commands": len(blocked),
        "blocked_imports": len(blocked_imports),
        "tool_timeout": config.TOOL_TIMEOUT,
        "generation_timeout": config.GENERATION_TIMEOUT,
    }


@router.get("/config-summary", dependencies=[Depends(require_auth)])
async def get_config_summary():
    """Return key feature flags from the running config."""
    return {
        "LLM_PROVIDER": config.LLM_PROVIDER,
        "LLM_MODEL": config.LLM_MODEL,
        "ENABLE_MCP": config.ENABLE_MCP,
        "ENABLE_DELEGATION": config.ENABLE_DELEGATION,
        "MAX_DELEGATION_DEPTH": config.MAX_DELEGATION_DEPTH,
        "ENABLE_AUTO_SKILL_CREATION": config.ENABLE_AUTO_SKILL_CREATION,
        "ENABLE_MODEL_ROUTING": config.ENABLE_MODEL_ROUTING,
        "ENABLE_EXTENDED_THINKING": config.ENABLE_EXTENDED_THINKING,
        "TOOL_TIMEOUT": config.TOOL_TIMEOUT,
        "GENERATION_TIMEOUT": config.GENERATION_TIMEOUT,
        "SYSTEM_ACCESS_LEVEL": config.SYSTEM_ACCESS_LEVEL,
    }


# ---------------------------------------------------------------------------
# Full Config + Runtime Config Update
# ---------------------------------------------------------------------------

@router.get("/config/full", dependencies=[Depends(require_auth)])
async def get_full_config():
    """Get all config values (sensitive fields redacted)."""
    return config.to_dict(redact_sensitive=True)


from pydantic import BaseModel, Field

class ConfigUpdateRequest(BaseModel):
    """Validated config update request. Only known mutable fields accepted.

    Keep in sync with _MUTABLE_FIELDS in app/config.py.
    """
    model_config = {"extra": "forbid"}

    LLM_PROVIDER: str | None = None
    LLM_MODEL: str | None = None
    OLLAMA_URL: str | None = None
    OPENAI_MODEL: str | None = None
    ANTHROPIC_MODEL: str | None = None
    GOOGLE_MODEL: str | None = None
    VISION_MODEL: str | None = None
    FAST_MODEL: str | None = None
    HEAVY_MODEL: str | None = None
    EMBEDDING_MODEL: str | None = None
    RETRIEVAL_TOP_K: int | None = Field(None, ge=1, le=50)
    CHUNK_SIZE: int | None = Field(None, ge=64, le=4096)
    CHUNK_OVERLAP: int | None = Field(None, ge=0, le=512)
    MAX_HISTORY_MESSAGES: int | None = Field(None, ge=1, le=100)
    MAX_LESSONS_IN_PROMPT: int | None = Field(None, ge=0, le=20)
    MAX_SKILLS_CHECK: int | None = Field(None, ge=0, le=50)
    MAX_CONTEXT_TOKENS: int | None = Field(None, ge=1000, le=128000)
    RECENT_MESSAGES_KEEP: int | None = Field(None, ge=1, le=100)
    CODE_EXEC_TIMEOUT: int | None = Field(None, ge=1, le=300)
    MAX_TOOL_ROUNDS: int | None = Field(None, ge=1, le=20)
    SHELL_EXEC_TIMEOUT: int | None = Field(None, ge=1, le=300)
    BROWSER_TIMEOUT: int | None = Field(None, ge=1, le=300)
    TOOL_TIMEOUT: int | None = Field(None, ge=10, le=600)
    GENERATION_TIMEOUT: int | None = Field(None, ge=30, le=1200)
    INTERNAL_LLM_TIMEOUT: int | None = Field(None, ge=5, le=120)
    ENABLE_PLANNING: bool | None = None
    ENABLE_CRITIQUE: bool | None = None
    ENABLE_CUSTOM_TOOLS: bool | None = None
    ENABLE_EXTENDED_THINKING: bool | None = None
    ENABLE_DELEGATION: bool | None = None
    ENABLE_CURIOSITY: bool | None = None
    ENABLE_VOICE: bool | None = None
    ENABLE_MODEL_ROUTING: bool | None = None
    ENABLE_HEARTBEAT: bool | None = None
    HEARTBEAT_INTERVAL: int | None = Field(None, ge=1)
    ENABLE_PROACTIVE: bool | None = None
    ENABLE_SHELL_EXEC: bool | None = None
    ENABLE_MCP: bool | None = None
    ENABLE_MCP_SERVER: bool | None = None
    ENABLE_AUTO_SKILL_CREATION: bool | None = None
    ENABLE_INJECTION_DETECTION: bool | None = None
    ENABLE_DESKTOP_AUTOMATION: bool | None = None
    ENABLE_WEBHOOKS: bool | None = None
    ENABLE_EMAIL_SEND: bool | None = None
    ENABLE_INTEGRATIONS: bool | None = None
    ENABLE_CALENDAR: bool | None = None
    ALLOWED_ORIGINS: str | None = None
    MAX_SYSTEM_TOKENS: int | None = Field(None, ge=500, le=16000)
    MAX_USER_FACTS: int | None = Field(None, ge=1, le=100)
    MAX_KG_FACTS: int | None = Field(None, ge=100, le=100000)
    MAX_CURIOSITY_PENDING: int | None = Field(None, ge=1, le=500)
    MAX_CURIOSITY_ATTEMPTS: int | None = Field(None, ge=1, le=10)
    MAX_CUSTOM_TOOL_CODE_LENGTH: int | None = Field(None, ge=100, le=50000)
    MAX_CUSTOM_TOOLS: int | None = Field(None, ge=1, le=200)
    RATE_LIMIT_RPM: int | None = Field(None, ge=10, le=1000)
    MAX_KG_FACTS_IN_PROMPT: int | None = Field(None, ge=1, le=50)
    MAX_REFLEXIONS_IN_PROMPT: int | None = Field(None, ge=0, le=20)
    MAX_SUCCESS_PATTERNS_IN_PROMPT: int | None = Field(None, ge=0, le=10)
    MAX_REFLEXIONS: int | None = Field(None, ge=10, le=1000)
    CRITIQUE_ANSWER_LIMIT: int | None = Field(None, ge=100, le=10000)
    CRITIQUE_SOURCES_LIMIT: int | None = Field(None, ge=100, le=10000)
    CRITIQUE_FACTS_LIMIT: int | None = Field(None, ge=100, le=10000)
    MAX_CRITIQUE_ROUNDS: int | None = Field(None, ge=1, le=10)
    DIGEST_HOUR: int | None = Field(None, ge=0, le=23)


@router.patch("/config", dependencies=[Depends(require_auth)])
async def update_config(body: ConfigUpdateRequest):
    """Update config values at runtime. Persists to overrides file.
    Returns {updated: [...], warnings: [...], restart_required: bool}
    """
    # Fields that require LLM provider reinitialization
    PROVIDER_FIELDS = {"LLM_PROVIDER", "LLM_MODEL", "OLLAMA_URL",
                       "OPENAI_API_KEY", "OPENAI_MODEL",
                       "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                       "GOOGLE_API_KEY", "GOOGLE_MODEL",
                       "VISION_MODEL", "FAST_MODEL", "HEAVY_MODEL"}

    # Fields that need full restart
    RESTART_FIELDS = {"DB_PATH", "CHROMADB_PATH", "HOST", "PORT"}

    # Filter to valid, mutable config fields only (uses _MUTABLE_FIELDS from config.py)
    valid_updates = {k: v for k, v in body.model_dump(exclude_none=True).items()
                     if k in _MUTABLE_FIELDS and hasattr(config, k)}

    if not valid_updates:
        return {"updated": [], "warnings": ["No valid config fields provided"], "restart_required": False}

    # Floor RATE_LIMIT_RPM at 10
    if "RATE_LIMIT_RPM" in valid_updates:
        valid_updates["RATE_LIMIT_RPM"] = max(10, int(valid_updates["RATE_LIMIT_RPM"]))

    changed_keys = list(valid_updates.keys())
    warnings = config.update(**valid_updates)
    config._save_overrides(changed_keys)

    needs_restart = bool(set(changed_keys) & RESTART_FIELDS)
    needs_provider_reinit = bool(set(changed_keys) & PROVIDER_FIELDS)

    # Reinitialize LLM provider if needed
    if needs_provider_reinit and not needs_restart:
        try:
            from app.core.llm import close_client, create_provider, set_provider
            await close_client()
            provider = create_provider(config)
            set_provider(provider)
        except Exception as e:
            warnings.append(f"Provider reinitialization failed: {e}")

    return {
        "updated": changed_keys,
        "warnings": warnings,
        "restart_required": needs_restart,
    }


# ---------------------------------------------------------------------------
# Knowledge Graph Facts
# ---------------------------------------------------------------------------

@router.get("/kg/facts", dependencies=[Depends(require_auth)])
async def get_kg_facts(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    search: str = "",
):
    """List KG facts with pagination and optional search."""
    from app.core.brain import get_services
    svc = get_services()
    if not svc.kg:
        return []
    if search:
        facts = svc.kg.search(search, limit=limit)
    else:
        fact_objects = svc.kg.get_all_facts(limit=limit, offset=offset)
        facts = [
            {
                "id": f.id, "subject": f.subject, "predicate": f.predicate,
                "object": f.object, "confidence": f.confidence,
                "source": f.source, "created_at": f.created_at,
                "valid_from": f.valid_from, "valid_to": f.valid_to,
            }
            for f in fact_objects
        ]
    return facts


# ---------------------------------------------------------------------------
# Custom Tools
# ---------------------------------------------------------------------------

@router.get("/custom-tools", dependencies=[Depends(require_auth)])
async def get_custom_tools():
    """List all custom tools."""
    from app.core.brain import get_services
    svc = get_services()
    if not svc.custom_tools:
        return []
    tools = svc.custom_tools.get_all_tools()
    return [{"id": t.id, "name": t.name, "description": t.description,
             "parameters": t.parameters, "times_used": t.times_used,
             "success_rate": t.success_rate, "enabled": t.enabled} for t in tools]


@router.delete("/custom-tools/{name}", dependencies=[Depends(require_auth)])
async def delete_custom_tool(name: str):
    """Delete a custom tool by name."""
    from app.core.brain import get_services
    svc = get_services()
    if not svc.custom_tools:
        raise HTTPException(404, "Custom tools not enabled")
    deleted = svc.custom_tools.delete_tool(name)
    if not deleted:
        raise HTTPException(404, f"Tool '{name}' not found")
    # Unregister from live registry
    if svc.tool_registry:
        tool = svc.tool_registry.get(name)
        if tool:
            svc.tool_registry._tools.pop(name, None)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Curiosity Queue
# ---------------------------------------------------------------------------

@router.get("/curiosity/queue", dependencies=[Depends(require_auth)])
async def get_curiosity_queue():
    """Get pending curiosity research items."""
    from app.core.brain import get_services
    svc = get_services()
    if not svc.curiosity:
        return []
    items = svc.curiosity.get_recent(limit=20)
    return [
        {
            "id": item.id, "topic": item.topic, "source": item.source,
            "urgency": item.urgency, "status": item.status,
            "attempts": item.attempts, "resolution": item.resolution,
            "created_at": item.created_at, "resolved_at": item.resolved_at,
        }
        for item in items
    ]


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

@router.get("/export", dependencies=[Depends(require_auth)])
async def export_all(limit: int = Query(default=10_000, ge=1, le=100_000)):
    """Export all Nova data as JSON.

    Includes: conversations, messages, user_facts, lessons, skills, training_data.
    """
    db = get_db()

    # Conversations + messages
    conversations = []
    conv_rows = db.fetchall("SELECT * FROM conversations ORDER BY created_at LIMIT ?", (limit,))
    for conv in conv_rows:
        msg_rows = db.fetchall(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at LIMIT ?",
            (conv["id"], limit),
        )
        conversations.append({
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": [
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "tool_calls": m["tool_calls"],
                    "tool_name": m["tool_name"],
                    "sources": m["sources"],
                    "created_at": m["created_at"],
                }
                for m in msg_rows
            ],
        })

    # User facts
    fact_rows = db.fetchall("SELECT * FROM user_facts ORDER BY key LIMIT ?", (limit,))
    user_facts = [
        {"key": r["key"], "value": r["value"], "source": r["source"], "confidence": r["confidence"]}
        for r in fact_rows
    ]

    # Lessons
    lesson_rows = db.fetchall("SELECT * FROM lessons ORDER BY id LIMIT ?", (limit,))
    lessons = [
        {
            "id": r["id"],
            "topic": r["topic"],
            "wrong_answer": r["wrong_answer"],
            "correct_answer": r["correct_answer"],
            "lesson_text": r["lesson_text"] if "lesson_text" in r.keys() else "",
            "context": r["context"],
            "confidence": r["confidence"],
            "times_retrieved": r["times_retrieved"],
            "times_helpful": r["times_helpful"],
            "created_at": r["created_at"],
        }
        for r in lesson_rows
    ]

    # Skills
    skill_rows = db.fetchall("SELECT * FROM skills ORDER BY id LIMIT ?", (limit,))
    skills = [
        {
            "id": r["id"],
            "name": r["name"],
            "trigger_pattern": r["trigger_pattern"],
            "steps": r["steps"],
            "answer_template": r["answer_template"],
            "learned_from": r["learned_from"],
            "times_used": r["times_used"],
            "success_rate": r["success_rate"],
            "enabled": bool(r["enabled"]),
            "created_at": r["created_at"],
        }
        for r in skill_rows
    ]

    # Training data (JSONL)
    training_data = []
    p = Path(config.TRAINING_DATA_PATH)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        training_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # KG facts
    kg_rows = db.fetchall("SELECT * FROM kg_facts ORDER BY id LIMIT ?", (limit,))
    kg_facts = [
        {
            "subject": r["subject"], "predicate": r["predicate"], "object": r["object"],
            "confidence": r["confidence"], "source": r["source"],
        }
        for r in kg_rows
    ]

    # Reflexions
    ref_rows = db.fetchall("SELECT * FROM reflexions ORDER BY id LIMIT ?", (limit,))
    reflexions_data = [
        {
            "task_summary": r["task_summary"], "outcome": r["outcome"],
            "reflection": r["reflection"], "quality_score": r["quality_score"],
            "tools_used": r["tools_used"], "revision_count": r["revision_count"],
        }
        for r in ref_rows
    ]

    # Custom tools
    ct_rows = db.fetchall("SELECT * FROM custom_tools ORDER BY id LIMIT ?", (limit,))
    custom_tools_data = [
        {
            "name": r["name"], "description": r["description"],
            "parameters": r["parameters"], "code": r["code"],
            "times_used": r["times_used"], "success_rate": r["success_rate"],
            "enabled": bool(r["enabled"]),
        }
        for r in ct_rows
    ]

    export = {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversations": conversations,
        "user_facts": user_facts,
        "lessons": lessons,
        "skills": skills,
        "training_data": training_data,
        "kg_facts": kg_facts,
        "reflexions": reflexions_data,
        "custom_tools": custom_tools_data,
    }

    return JSONResponse(
        content=export,
        headers={"Content-Disposition": "attachment; filename=nova_export.json"},
    )


@router.post("/import", dependencies=[Depends(require_auth)])
async def import_all(file: UploadFile = File(...)):
    """Import Nova data from a JSON export file.

    Merges with existing data (skips duplicates by ID).
    """
    # Guard against oversized uploads (100 MB max)
    _MAX_IMPORT_SIZE = 100 * 1024 * 1024
    content = await file.read(_MAX_IMPORT_SIZE + 1)
    if len(content) > _MAX_IMPORT_SIZE:
        raise HTTPException(status_code=413, detail="Import file exceeds 100 MB limit")

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if "version" not in data:
        raise HTTPException(status_code=400, detail="Missing 'version' field — not a valid Nova export")

    db = get_db()
    stats = {"conversations": 0, "messages": 0, "user_facts": 0, "lessons": 0, "skills": 0, "training_data": 0, "kg_facts": 0}

    with db.transaction() as tx:
        # Import conversations + messages
        for conv in data.get("conversations", []):
            existing = tx.fetchone("SELECT id FROM conversations WHERE id = ?", (conv["id"],))
            if existing:
                continue
            tx.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv["id"], conv.get("title"), conv.get("created_at"), conv.get("updated_at")),
            )
            stats["conversations"] += 1
            for msg in conv.get("messages", []):
                tx.execute(
                    """INSERT OR IGNORE INTO messages
                       (id, conversation_id, role, content, tool_calls, tool_name, sources, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        msg["id"], conv["id"], msg["role"], msg["content"],
                        msg.get("tool_calls"), msg.get("tool_name"),
                        msg.get("sources"), msg.get("created_at"),
                    ),
                )
                stats["messages"] += 1

        # Import user facts (upsert)
        for fact in data.get("user_facts", []):
            existing = tx.fetchone("SELECT id FROM user_facts WHERE key = ?", (fact["key"],))
            if not existing:
                tx.execute(
                    "INSERT INTO user_facts (key, value, source, confidence) VALUES (?, ?, ?, ?)",
                    (fact["key"], fact["value"], fact.get("source", "imported"), fact.get("confidence", 1.0)),
                )
                stats["user_facts"] += 1

        # Import lessons
        for lesson in data.get("lessons", []):
            existing = tx.fetchone("SELECT id FROM lessons WHERE id = ?", (lesson["id"],))
            if not existing:
                tx.execute(
                    """INSERT INTO lessons
                       (topic, wrong_answer, correct_answer, lesson_text, context, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        lesson["topic"], lesson.get("wrong_answer", ""),
                        lesson.get("correct_answer", ""), lesson.get("lesson_text", ""),
                        lesson.get("context", ""), lesson.get("confidence", 0.8),
                    ),
                )
                stats["lessons"] += 1

        # Import skills
        for skill in data.get("skills", []):
            existing = tx.fetchone(
                "SELECT id FROM skills WHERE name = ? AND trigger_pattern = ?",
                (skill["name"], skill["trigger_pattern"]),
            )
            if not existing:
                steps = skill["steps"]
                if isinstance(steps, list):
                    steps = json.dumps(steps)
                tx.execute(
                    """INSERT INTO skills (name, trigger_pattern, steps, answer_template, learned_from)
                       VALUES (?, ?, ?, ?, ?)""",
                    (skill["name"], skill["trigger_pattern"], steps,
                     skill.get("answer_template"), skill.get("learned_from")),
                )
                stats["skills"] += 1

        # Import KG facts
        for fact in data.get("kg_facts", []):
            existing = tx.fetchone(
                "SELECT id FROM kg_facts WHERE subject = ? AND predicate = ? AND object = ?",
                (fact["subject"], fact["predicate"], fact["object"]),
            )
            if not existing:
                tx.execute(
                    "INSERT INTO kg_facts (subject, predicate, object, confidence, source) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (fact["subject"], fact["predicate"], fact["object"],
                     fact.get("confidence", 0.8), fact.get("source", "imported")),
                )
                stats["kg_facts"] += 1

        # Import reflexions (deduplicate by task_summary + reflection)
        for ref in data.get("reflexions", []):
            existing = tx.fetchone(
                "SELECT id FROM reflexions WHERE task_summary = ? AND reflection = ?",
                (ref["task_summary"], ref["reflection"]),
            )
            if not existing:
                tx.execute(
                    "INSERT INTO reflexions (task_summary, outcome, reflection, quality_score, tools_used, revision_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        ref["task_summary"], ref.get("outcome", "failure"),
                        ref["reflection"], ref.get("quality_score", 0.5),
                        ref.get("tools_used", ""), ref.get("revision_count", 0),
                    ),
                )
                stats["reflexions"] = stats.get("reflexions", 0) + 1

        # Import custom tools (validate code before importing via AST analysis)
        from app.tools.code_exec import _check_code_safety
        for ct in data.get("custom_tools", []):
            existing = tx.fetchone(
                "SELECT id FROM custom_tools WHERE name = ?", (ct["name"],)
            )
            if not existing:
                code = ct.get("code", "")
                # Validate code against access tier restrictions using AST-based check
                safety_error = _check_code_safety(code)
                if safety_error:
                    logger.warning("Rejected imported custom tool '%s': %s", ct["name"], safety_error)
                    continue
                tx.execute(
                    "INSERT INTO custom_tools (name, description, parameters, code, times_used, success_rate, enabled) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ct["name"], ct["description"], ct["parameters"], code,
                        ct.get("times_used", 0), ct.get("success_rate", 1.0),
                        int(ct.get("enabled", True)),
                    ),
                )
                stats["custom_tools"] = stats.get("custom_tools", 0) + 1

    # Training data written to file (outside transaction) with deduplication
    # Uses atomic temp-file + os.replace to prevent corruption on crash.
    training_entries = data.get("training_data", [])
    if training_entries:
        import hashlib
        import os
        import tempfile

        p = Path(config.TRAINING_DATA_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Read existing lines + build hash set for dedup
        existing_lines: list[str] = []
        existing_hashes: set[str] = set()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        existing_hashes.add(hashlib.sha256(stripped.encode()).hexdigest())
                        existing_lines.append(stripped)

        # Compute new lines to append
        new_lines: list[str] = []
        for entry in training_entries:
            entry_line = json.dumps(entry, ensure_ascii=False)
            entry_hash = hashlib.sha256(entry_line.encode()).hexdigest()
            if entry_hash not in existing_hashes:
                new_lines.append(entry_line)
                existing_hashes.add(entry_hash)
                stats["training_data"] += 1

        if new_lines:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(p.parent), suffix=".tmp", prefix=".training_import_",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    for line in existing_lines:
                        f.write(line + "\n")
                    for line in new_lines:
                        f.write(line + "\n")
                os.replace(tmp_path, str(p))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    return {"status": "imported", "stats": stats}
