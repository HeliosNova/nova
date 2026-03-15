"""System endpoints — health check, status, export/import, integrations, access tier, config."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.auth import require_auth
from app.config import config
from app.database import get_db
from app.schema import HealthResponse, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — reports LLM provider and DB connectivity."""
    from app.core import llm

    llm_ok = False
    try:
        llm_ok = await llm.check_health()
    except Exception:
        pass

    db_ok = False
    try:
        db = get_db()
        db.fetchone("SELECT 1")
        db_ok = True
    except Exception:
        pass

    provider = config.LLM_PROVIDER

    return HealthResponse(
        status="ok" if (llm_ok and db_ok) else "degraded",
        version="1.0.0",
        model=config.LLM_MODEL,
        provider=provider,
        llm_connected=llm_ok,
        ollama_connected=llm_ok if provider == "ollama" else False,
        db_connected=db_ok,
    )


_ALLOWED_TABLES = frozenset({
    "conversations", "messages", "user_facts", "lessons", "skills", "documents",
    "kg_facts", "reflexions", "custom_tools",
})


@router.get("/status", response_model=StatusResponse, dependencies=[Depends(require_auth)])
async def status() -> StatusResponse:
    """System status — counts of key entities."""
    db = get_db()

    def count(table: str) -> int:
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"Table '{table}' is not allowed")
        row = db.fetchone(f"SELECT COUNT(*) as c FROM {table}")
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
    from app.integrations.registry import IntegrationRegistry
    registry = IntegrationRegistry()
    return [
        {
            "name": i.name,
            "auth_type": i.auth_type,
            "auth_env_var": i.auth_env_var,
            "is_configured": i.is_configured,
            "endpoint_count": len(i.endpoints),
            "description": i.description,
        }
        for i in registry.get_all()
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


@router.patch("/config", dependencies=[Depends(require_auth)])
async def update_config(updates: dict):
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

    # Filter to valid config fields only
    valid_updates = {k: v for k, v in updates.items()
                     if hasattr(config, k) and not k.startswith('_')}

    if not valid_updates:
        return {"updated": [], "warnings": ["No valid config fields provided"], "restart_required": False}

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
async def get_kg_facts(limit: int = 50, offset: int = 0, search: str = ""):
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
async def export_all():
    """Export all Nova data as JSON.

    Includes: conversations, messages, user_facts, lessons, skills, training_data.
    """
    db = get_db()

    # Conversations + messages
    conversations = []
    conv_rows = db.fetchall("SELECT * FROM conversations ORDER BY created_at")
    for conv in conv_rows:
        msg_rows = db.fetchall(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv["id"],),
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
    fact_rows = db.fetchall("SELECT * FROM user_facts ORDER BY key")
    user_facts = [
        {"key": r["key"], "value": r["value"], "source": r["source"], "confidence": r["confidence"]}
        for r in fact_rows
    ]

    # Lessons
    lesson_rows = db.fetchall("SELECT * FROM lessons ORDER BY id")
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
    skill_rows = db.fetchall("SELECT * FROM skills ORDER BY id")
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
    kg_rows = db.fetchall("SELECT * FROM kg_facts ORDER BY id")
    kg_facts = [
        {
            "subject": r["subject"], "predicate": r["predicate"], "object": r["object"],
            "confidence": r["confidence"], "source": r["source"],
        }
        for r in kg_rows
    ]

    # Reflexions
    ref_rows = db.fetchall("SELECT * FROM reflexions ORDER BY id")
    reflexions_data = [
        {
            "task_summary": r["task_summary"], "outcome": r["outcome"],
            "reflection": r["reflection"], "quality_score": r["quality_score"],
            "tools_used": r["tools_used"], "revision_count": r["revision_count"],
        }
        for r in ref_rows
    ]

    # Custom tools
    ct_rows = db.fetchall("SELECT * FROM custom_tools ORDER BY id")
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
        "exported_at": datetime.now().isoformat(),
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

        # Import custom tools
        for ct in data.get("custom_tools", []):
            existing = tx.fetchone(
                "SELECT id FROM custom_tools WHERE name = ?", (ct["name"],)
            )
            if not existing:
                tx.execute(
                    "INSERT INTO custom_tools (name, description, parameters, code, times_used, success_rate, enabled) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ct["name"], ct["description"], ct["parameters"], ct["code"],
                        ct.get("times_used", 0), ct.get("success_rate", 1.0),
                        int(ct.get("enabled", True)),
                    ),
                )
                stats["custom_tools"] = stats.get("custom_tools", 0) + 1

    # Training data written to file (outside transaction)
    training_entries = data.get("training_data", [])
    if training_entries:
        p = Path(config.TRAINING_DATA_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for entry in training_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["training_data"] += 1

    return {"status": "imported", "stats": stats}
