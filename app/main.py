"""Nova — Sovereign Personal AI.

FastAPI application entry point.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import __version__
from app.auth import _get_client_ip
from app.config import config

# Correlation ID for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
from app.core.brain import Services, set_services
from app.core.learning import LearningEngine
from app.core.llm import close_client, create_provider, set_provider
from app.core.memory import ConversationStore, UserFactStore
from app.core.retriever import Retriever
from app.core.skills import SkillStore
from app.database import get_db
from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.code_exec import CodeExecTool
from app.tools.code_verify import CodeVerifyTool
from app.tools.code_understand import CodeUnderstandTool
from app.tools.file_ops import FileOpsTool
from app.tools.http_fetch import HttpFetchTool
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.memory_tool import MemorySearchTool
from app.tools.browser import BrowserTool
from app.tools.monitor_tool import MonitorTool
from app.tools.active_memory import ActiveMemoryTool
from app.tools.screenshot import ScreenshotTool
from app.tools.shell_exec import ShellExecTool
from app.tools.web_search import WebSearchTool
from app.tools.search_agent import SearchAgentTool
from app.tools.action_email import EmailSendTool
from app.tools.action_calendar import CalendarTool
from app.tools.action_reminder import ReminderTool
from app.tools.action_webhook import WebhookTool
from app.tools.delegate import DelegateTool
from app.tools.background_task import BackgroundTaskTool
from app.core.task_manager import TaskManager

class _CorrelationIDFormatter(logging.Formatter):
    """Formatter that injects request_id from context var, defaulting to empty."""
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get("")
        return super().format(record)

_log_formatter = _CorrelationIDFormatter(
    "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s"
)
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_log_formatter)
# Persist logs under /data so delivery-truth (`digest sent` lines) survives
# container recreation — json-file container logs are destroyed on every rebuild,
# which left 61/63 of a day's deliveries unverifiable (2026-08-14 audit).
_log_handlers: list[logging.Handler] = [_log_handler]
try:
    from logging.handlers import RotatingFileHandler
    os.makedirs("/data/logs", exist_ok=True)
    _file_handler = RotatingFileHandler(
        "/data/logs/nova-app.log", maxBytes=20_000_000, backupCount=5, encoding="utf-8"
    )
    _file_handler.setFormatter(_log_formatter)
    _log_handlers.append(_file_handler)
except Exception:
    pass  # best-effort; never block startup on file logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    handlers=_log_handlers,
)
# Silence ChromaDB's noisy "Delete of nonexisting embedding" + "Add of existing"
# warnings. They fire on every prune or upsert when ChromaDB IDs don't match SQLite,
# which is normal during cleanup; the operations succeed regardless.
logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)
# ChromaDB's posthog telemetry pings still fire even with ANONYMIZED_TELEMETRY=false
# because it imports posthog before reading the flag. Each call logs an exception
# ("capture() takes 1 positional argument but 3 were given"). Silence the logger
# itself so we stop seeing dozens of these per hour.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
# httpx logs every request URL at INFO; for the Telegram bot that URL embeds the
# bot TOKEN (…/bot<token>/getUpdates), leaking it into every log sink ~2×/min
# (2026-08-14 audit: 381 hits in 3h). Silence to WARNING. (Rotate the token — owner.)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # --- Startup ---
    logger.info("Nova starting up...")

    # Auth check — warn prominently if API_KEY is not set
    if not config.API_KEY:
        logger.critical(
            "*** SECURITY WARNING: NOVA_API_KEY is not set — all endpoints are publicly accessible! ***"
        )

    # Validate configuration
    config_warnings = config.validate()
    if config_warnings:
        for warning in config_warnings:
            logger.warning("Config: %s", warning)
    else:
        logger.info("Config validated OK — no warnings")

    # Initialize LLM provider
    provider = create_provider(config)
    set_provider(provider)
    logger.info("LLM provider: %s (model: %s)", config.LLM_PROVIDER, config.LLM_MODEL)

    # Initialize database schema
    db = get_db()
    db.init_schema()
    logger.info("Database initialized at %s", config.DB_PATH)

    # Seed prompt-module baselines (idempotent; no-op if already seeded)
    from app.core.prompt_optimizer import init_prompt_optimizer
    init_prompt_optimizer(db)

    # Restore auth lockout state from DB
    from app.auth import load_lockouts_from_db
    load_lockouts_from_db()

    # Core services
    conversations = ConversationStore(db)
    user_facts = UserFactStore(db)
    learning = LearningEngine(db)
    skills = SkillStore(db)
    try:
        _disabled = skills.revalidate_enabled_skills()
        if _disabled:
            logger.info("Disabled %d stored skill(s) failing current guards", _disabled)
    except Exception as _e:
        logger.warning("Skill revalidation skipped: %s", _e)
    try:
        skills.sync_embeddings()
    except Exception as _e:
        logger.warning("Skill embedding sync failed (ChromaDB may be unavailable): %s", _e)

    # Knowledge graph + reflexion store. to_thread: their first construction
    # runs schema DDL (write lock) — keep it off the event-loop thread
    # (audit 2026-08-23; later constructions skip DDL via the SafeDB memo).
    from app.core.kg import KnowledgeGraph
    kg = await asyncio.to_thread(KnowledgeGraph, db)
    logger.info("Knowledge graph initialized")

    from app.core.reflexion import ReflexionStore
    reflexions = await asyncio.to_thread(ReflexionStore, db)
    logger.info("Reflexion store initialized")

    # KG auto-curation (heuristic pass runs inline, LLM pass runs in background)
    # Note: KG/reflexion decay is handled by the daily maintenance monitor
    kg_curation_task = None
    try:
        curation = await kg.curate(sample_size=0)  # heuristic only — fast
        heuristic_cleaned = curation.get("heuristic", 0)
        if heuristic_cleaned:
            logger.info("KG curation: removed %d garbage facts (heuristic)", heuristic_cleaned)

        async def _bg_kg_curate():
            try:
                result = await kg.curate(sample_size=20, heuristic=False)  # LLM only
                llm_cleaned = result.get("llm", 0)
                if llm_cleaned:
                    logger.info("KG LLM curation: removed %d additional facts", llm_cleaned)
            except Exception as e:
                logger.warning("KG LLM curation failed (non-blocking): %s", e)

        kg_curation_task = asyncio.create_task(_bg_kg_curate())
    except Exception as e:
        logger.warning("KG curation failed: %s", e)

    # Retriever (ChromaDB may fail if not installed — graceful degradation)
    retriever = None
    try:
        retriever = Retriever(db)
        logger.info("Retriever initialized (ChromaDB + FTS5)")
    except Exception as e:
        logger.warning("Retriever init failed (documents won't work): %s", e)

    # Tool registry — each tool is wrapped so one bad init can't crash startup
    registry = ToolRegistry()
    _tool_instances = [
        ("WebSearchTool", lambda: WebSearchTool()),
        ("SearchAgentTool", lambda: SearchAgentTool()),
        ("CalculatorTool", lambda: CalculatorTool()),
        ("HttpFetchTool", lambda: HttpFetchTool()),
        ("KnowledgeSearchTool", lambda: KnowledgeSearchTool(retriever=retriever)),
        ("CodeExecTool", lambda: CodeExecTool()),
        ("CodeVerifyTool", lambda: CodeVerifyTool()),
        ("CodeUnderstandTool", lambda: CodeUnderstandTool()),
        ("MemorySearchTool", lambda: MemorySearchTool(conversations=conversations, user_facts=user_facts)),
        ("FileOpsTool", lambda: FileOpsTool()),
        ("ShellExecTool", lambda: ShellExecTool()),
        ("BrowserTool", lambda: BrowserTool()),
        ("ScreenshotTool", lambda: ScreenshotTool()),
        ("EmailSendTool", lambda: EmailSendTool()),
        ("CalendarTool", lambda: CalendarTool()),
        ("WebhookTool", lambda: WebhookTool()),
    ]
    if config.ENABLE_DELEGATION:
        _tool_instances.append(("DelegateTool", lambda: DelegateTool()))

    # Context detail tool (lazy context retrieval — uses get_services() at call time)
    from app.tools.context_detail import ContextDetailTool
    _tool_instances.append(("ContextDetailTool", lambda: ContextDetailTool()))

    # Background task manager — persists to SQLite so restarts leave an audit trail.
    # to_thread: first construction runs persist-table DDL + interrupted-task
    # hydration (write lock) — keep it off the event-loop thread.
    from app.database import get_db as _get_db
    task_manager = await asyncio.to_thread(
        TaskManager,
        max_concurrent=config.MAX_BACKGROUND_TASKS,
        task_timeout=config.BACKGROUND_TASK_TIMEOUT,
        db=_get_db(),
    )
    _tool_instances.append(("BackgroundTaskTool", lambda: BackgroundTaskTool()))

    for tool_name, tool_factory in _tool_instances:
        try:
            registry.register(tool_factory())
        except Exception as e:
            logger.warning("Failed to register %s: %s", tool_name, e)

    # Desktop automation (optional — requires display server + PyAutoGUI)
    if config.ENABLE_DESKTOP_AUTOMATION:
        from app.tools.desktop import DesktopTool
        try:
            registry.register(DesktopTool())
            logger.info("Desktop automation tool registered")
        except Exception as e:
            logger.warning("Desktop automation tool registration failed: %s", e)

    # Integration templates
    integration_registry = None
    if config.ENABLE_INTEGRATIONS:
        from app.integrations.registry import IntegrationRegistry
        from app.tools.integration import IntegrationTool, set_registry as set_integration_registry
        integration_registry = IntegrationRegistry()
        set_integration_registry(integration_registry)
        configured = integration_registry.get_configured()
        if configured:
            registry.register(IntegrationTool())
            logger.info("Integrations configured: %s", ", ".join(i.name for i in configured))
        else:
            logger.info("Integration templates loaded, none configured (no env tokens set)")

    # MCP tools (external tool servers via Model Context Protocol)
    mcp_manager = None
    if config.ENABLE_MCP:
        from app.tools.mcp import MCPManager
        mcp_manager = MCPManager()
        try:
            mcp_count = await mcp_manager.discover_and_register(registry)
            if mcp_count:
                logger.info("MCP: registered %d external tools", mcp_count)
            else:
                logger.info("MCP enabled, no tools discovered")
        except Exception as e:
            logger.warning("MCP discovery failed: %s", e)
            mcp_manager = None

    logger.info("Tools registered: %s", ", ".join(registry.tool_names))

    # Custom tools (dynamic tool creation)
    custom_tools = None
    if config.ENABLE_CUSTOM_TOOLS:
        from app.core.custom_tools import CustomToolStore, DynamicTool
        custom_tools = CustomToolStore(db)
        loaded = custom_tools.get_all_tools()
        for ct in loaded:
            registry.register(DynamicTool(ct, custom_tools))
        if loaded:
            logger.info("Loaded %d custom tool(s): %s", len(loaded), ", ".join(t.name for t in loaded))
        else:
            logger.info("Custom tools enabled (0 loaded)")

    # Monitor store + monitor tool
    # Active Memory Tool (AgeMem pattern — agent-managed memory)
    registry.register(ActiveMemoryTool(db=db))

    # Trust Manager (Sovereign-OS — earned trust with asymmetric scoring)
    trust_manager = None
    try:
        from app.core.trust import TrustManager
        # to_thread: first construction runs DDL + singleton bootstrap (write lock)
        trust_manager = await asyncio.to_thread(TrustManager, db)
        registry.trust_manager = trust_manager  # Attach to registry for tool gating
        logger.info("Trust system initialized (score: %.0f)", trust_manager.get_score())
    except Exception as e:
        logger.warning("Trust system init failed: %s", e)

    monitor_store = None
    if config.ENABLE_HEARTBEAT:
        try:
            from app.monitors.heartbeat import MonitorStore
            monitor_store = MonitorStore(db)
            registry.register(MonitorTool(monitor_store=monitor_store))
            registry.register(ReminderTool(monitor_store=monitor_store))
            seeded = monitor_store.seed_defaults()
            if seeded:
                logger.info("Seeded %d default monitor(s)", seeded)
            logger.info("Monitor store initialized (%d monitors)", len(monitor_store.list_all()))
        except Exception as e:
            logger.warning("Monitor store init failed (heartbeat disabled): %s", e)
            monitor_store = None

    # Curiosity engine + topic tracker
    curiosity_queue = None
    topic_tracker = None
    if config.ENABLE_CURIOSITY:
        from app.core.curiosity import CuriosityQueue, TopicTracker
        # to_thread: first construction runs schema DDL (write lock)
        curiosity_queue = await asyncio.to_thread(CuriosityQueue, db)
        topic_tracker = await asyncio.to_thread(TopicTracker, db)
        logger.info("Curiosity engine initialized")

    # External skills loader (AgentSkills format)
    external_skills = None
    try:
        from app.core.skill_loader import load_skills
        external_skills = load_skills()
        if external_skills:
            logger.info("Loaded %d external skill(s)", len(external_skills))
    except Exception as e:
        logger.warning("External skill loading failed: %s", e)

    # Assemble services
    svc = Services(
        conversations=conversations,
        user_facts=user_facts,
        retriever=retriever,
        learning=learning,
        skills=skills,
        tool_registry=registry,
        kg=kg,
        reflexions=reflexions,
        custom_tools=custom_tools,
        monitor_store=monitor_store,
        curiosity=curiosity_queue,
        topic_tracker=topic_tracker,
        external_skills=external_skills,
        task_manager=task_manager,
    )
    set_services(svc)

    # Load Nova-authored extensions from /data/extensions/. Modules can
    # register additional tools, monitors, or hooks via a register(svc)
    # function. Failures are non-fatal — one broken extension doesn't break
    # the system.
    try:
        from app.core.extensions import load_all as _load_extensions
        ext_report = _load_extensions(svc)
        if ext_report["loaded"]:
            logger.info(
                "[Extensions] loaded %d at boot: %s",
                len(ext_report["loaded"]), ", ".join(ext_report["loaded"]),
            )
        if ext_report["failed"]:
            for f in ext_report["failed"]:
                logger.warning(
                    "[Extensions] %s failed at boot: %s",
                    f["name"], f["error"],
                )
    except Exception as e:
        logger.warning("Extensions load failed: %s", e)

    # Cleanup old conversations on startup
    try:
        cleaned = conversations.cleanup_old_conversations(days=90)
        if cleaned > 0:
            logger.info("Cleaned up %d old conversations", cleaned)
        else:
            logger.info("Conversation cleanup: 0 old conversations to remove")
    except Exception as e:
        logger.warning("Conversation cleanup failed: %s", e)

    # Reindex lessons into ChromaDB (one-time migration)
    try:
        reindexed = learning.reindex_lessons()
        if reindexed:
            logger.info("Reindexed %d lessons into ChromaDB", reindexed)
    except Exception as e:
        logger.warning("Lesson reindex failed: %s", e)

    # Reindex reflexions into ChromaDB (one-time migration)
    try:
        reindexed_r = reflexions.reindex_reflexions()
        if reindexed_r:
            logger.info("Reindexed %d reflexions into ChromaDB", reindexed_r)
    except Exception as e:
        logger.warning("Reflexion reindex failed: %s", e)

    # Reindex KG facts into ChromaDB for semantic search
    try:
        reindexed_kg = kg.reindex_kg_facts()
        if reindexed_kg:
            logger.info("Reindexed %d KG facts into ChromaDB", reindexed_kg)
    except Exception as e:
        logger.warning("KG reindex failed: %s", e)

    # Decay confidence on stale lessons
    try:
        decayed = learning.decay_stale_lessons(days=30)
        if decayed:
            logger.info("Decayed %d stale lessons", decayed)
        else:
            logger.info("Lesson decay: all lessons are fresh")
    except Exception as e:
        logger.warning("Lesson decay failed: %s", e)

    # Prune dead lessons (never used, too old)
    try:
        deleted = learning.prune_dead_lessons()
        if deleted:
            logger.info("Pruned %d dead lessons on startup", deleted)
    except Exception as e:
        logger.warning("Dead-lesson prune failed: %s", e)

    logger.info("Model: %s", config.LLM_MODEL)
    logger.info("Ollama: %s", config.OLLAMA_URL)
    logger.info("SearXNG: %s", config.SEARXNG_URL)

    # Start channel bots (if tokens configured)
    channel_tasks = []
    discord_bot = None
    telegram_bot = None
    whatsapp_bot = None
    signal_bot = None

    if config.DISCORD_TOKEN:
        from app.channels.discord import DiscordBot
        discord_bot = DiscordBot()
        channel_tasks.append(asyncio.create_task(discord_bot.start()))
        logger.info("Discord bot starting...")

    if config.TELEGRAM_TOKEN:
        from app.channels.telegram import TelegramBot
        telegram_bot = TelegramBot()
        channel_tasks.append(asyncio.create_task(telegram_bot.start()))
        logger.info("Telegram bot starting...")

    if config.WHATSAPP_API_TOKEN:
        from app.channels.whatsapp import WhatsAppBot
        whatsapp_bot = WhatsAppBot()
        app.include_router(whatsapp_bot.get_router())
        channel_tasks.append(asyncio.create_task(whatsapp_bot.start()))
        logger.info("WhatsApp bot starting (webhook mode)...")

    if config.SIGNAL_API_URL and config.SIGNAL_PHONE_NUMBER:
        from app.channels.signal import SignalBot
        signal_bot = SignalBot()
        channel_tasks.append(asyncio.create_task(signal_bot.start()))
        logger.info("Signal bot starting (polling mode)...")

    # Start heartbeat + proactive engines
    def _on_bg_task_done(task: asyncio.Task) -> None:
        """Log unhandled exceptions from monitor background tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("Background task %s died: %s", task.get_name(), exc, exc_info=exc)

    heartbeat_loop = None
    daily_digest = None
    # Hoisted to lifespan scope (2026-08-20 sweep) so shutdown can stop them:
    # both own a background loop with a working stop() that was never called,
    # so on shutdown close_all() closed the DB connections while the daemon
    # kept ticking → "closed database" error spam + no draining of in-flight
    # goal/dream work. The delete-side-of-lifecycle-rots pattern.
    daemon_orch = None
    event_trigger = None
    if config.ENABLE_HEARTBEAT and monitor_store:
        try:
            from app.monitors.heartbeat import HeartbeatLoop
            from app.monitors.proactive import DailyDigest
            heartbeat_loop = HeartbeatLoop(
                monitor_store,
                discord_bot=discord_bot,
                telegram_bot=telegram_bot,
                whatsapp_bot=whatsapp_bot,
                signal_bot=signal_bot,
            )
            task = heartbeat_loop.start()
            task.add_done_callback(_on_bg_task_done)
            svc.heartbeat = heartbeat_loop
            logger.info("Heartbeat loop started")

            if config.ENABLE_PROACTIVE:
                daily_digest = DailyDigest(
                    monitor_store,
                    discord_bot=discord_bot,
                    telegram_bot=telegram_bot,
                    whatsapp_bot=whatsapp_bot,
                    signal_bot=signal_bot,
                    learning_engine=learning,
                    db=db,  # without this _save/_load_last_digest silently no-op:
                            # no durable liveness record + restart double-send/skip
                )
                dtask = daily_digest.start()
                dtask.add_done_callback(_on_bg_task_done)
                logger.info("Daily digest started (hour=%d)", config.DIGEST_HOUR)

            # Start daemon orchestrator
            from app.monitors.daemon import DaemonOrchestrator
            daemon_orch = DaemonOrchestrator(monitor_store._db)
            daemon_orch.start()
            logger.info("Daemon orchestrator started")

            # Start event-driven trigger system
            if config.ENABLE_EVENT_TRIGGERS:
                from app.monitors.event_trigger import EventTrigger, set_event_trigger
                event_trigger = EventTrigger(monitor_store, heartbeat_loop, db)
                et_task = event_trigger.start()
                et_task.add_done_callback(_on_bg_task_done)
                set_event_trigger(event_trigger)
                logger.info("Event trigger system started")
        except Exception as e:
            logger.warning("Heartbeat/proactive startup failed: %s", e)

    # --- Model warmup ---
    # Issue a tiny generation so Ollama loads the model into VRAM before the
    # first user query. Cuts first-query latency from ~30-60s to <1s.
    async def _warmup():
        try:
            from app.core import llm
            t0 = time.monotonic()
            # max_tokens=16 + a prompt with a natural 1-word answer (was "ok"
            # at max_tokens=2). Warmup only needs the weights resident, but
            # cutting it off mid-generation made done_reason="length" fire the
            # [truncation] tripwire on EVERY startup — a guaranteed false
            # positive in the same log the real mid-sentence truncations use.
            # Let it stop on its own so the tripwire keeps meaning something.
            await llm.invoke_nothink(
                [{"role": "user", "content": "Reply with exactly one word: ok"}],
                max_tokens=16, temperature=0.0,
            )
            logger.info("Model warmup complete (%.1fs)", time.monotonic() - t0)
        except Exception as e:
            logger.warning("Model warmup failed (non-fatal): %s", e)
    _warm_task = asyncio.create_task(_warmup())
    _warm_task.add_done_callback(_on_bg_task_done)

    # Startup init is done — from here, a sync DB call on the loop thread is a
    # genuine steady-state offender and the tripwire should be loud.
    from app.database import SafeDB
    SafeDB.end_startup_grace()

    logger.info("Nova ready.")

    yield

    # --- Shutdown ---
    logger.info("Nova shutting down...")

    # Cancel KG curation task if still running
    try:
        if kg_curation_task is not None and not kg_curation_task.done():
            kg_curation_task.cancel()
    except NameError:
        pass  # kg_curation_task was never assigned (curation failed at startup)

    # Stop heartbeat + proactive
    if heartbeat_loop:
        heartbeat_loop.stop()
    if daily_digest:
        daily_digest.stop()
    # Stop the daemon orchestrator + event trigger BEFORE close_all() (2026-08-20
    # sweep) so their loops aren't ticking against closed DB connections.
    if daemon_orch:
        try:
            await daemon_orch.stop()
        except Exception as e:
            logger.warning("Daemon orchestrator stop failed: %s", e)
    if event_trigger:
        try:
            event_trigger.stop()
        except Exception as e:
            logger.warning("Event trigger stop failed: %s", e)

    # Stop channel bots
    if discord_bot:
        await discord_bot.close()
    if telegram_bot:
        await telegram_bot.close()
    if whatsapp_bot:
        await whatsapp_bot.close()
    if signal_bot:
        await signal_bot.close()
    for task in channel_tasks:
        task.cancel()
    if channel_tasks:
        await asyncio.gather(*channel_tasks, return_exceptions=True)

    # Unload Whisper model
    if config.ENABLE_VOICE:
        from app.core.voice import unload_transcriber
        unload_transcriber()

    # Cancel background tasks
    await task_manager.cancel_all()

    # Cancel the model-warmup task if still running — it was the one startup
    # background task without a shutdown handler, so close_all() below could
    # close the DB out from under an in-flight warmup gen (audit 2026-08-23).
    if _warm_task and not _warm_task.done():
        _warm_task.cancel()
        try:
            await _warm_task
        except (asyncio.CancelledError, Exception):
            pass

    # Close MCP sessions
    if mcp_manager:
        await mcp_manager.close()

    # Close retriever (ChromaDB client cleanup)
    if retriever:
        retriever.close()

    # Close HTTP fetch connection pool
    from app.tools.http_fetch import close_http_client
    await close_http_client()

    # Close the headless browser (2026-08-20 sweep): the class-level chromium
    # process tree relied solely on the 10-min idle timeout, which only fires
    # on the NEXT browser call — so shutdown leaked a chromium subprocess
    # (the 2026-07-06 13-zombie incident class).
    # NB: no local `from app.tools.browser import BrowserTool` here. This
    # function-level import (removed 2026-08-29) rebound BrowserTool as a LOCAL
    # of lifespan for the WHOLE function body, so the registration lambda ~430
    # lines above closed over an unbound local instead of the module-level
    # import — "cannot access free variable 'BrowserTool'" on every startup,
    # and the browser tool silently never registered. Use the module import.
    try:
        await BrowserTool._close_session()
    except Exception as e:
        logger.warning("Browser close on shutdown failed: %s", e)

    await close_client()
    from app.database import close_all
    close_all()


_docs_url = None if config.API_KEY else "/docs"
_redoc_url = None if config.API_KEY else "/redoc"

app = FastAPI(
    title="Nova",
    version=__version__,
    description="Sovereign Personal AI",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
)

# Rate limiting — simple in-memory per-IP limiter
# Module-level state so tests can call `_rate_limit_requests.clear()` between runs
_rate_limit_requests: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter. 60 requests/minute per IP. Skips /api/health."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests = _rate_limit_requests
        self._lock = _rate_limit_lock

    async def dispatch(self, request: Request, call_next):
        # Skip health check
        if request.url.path == "/api/health":
            return await call_next(request)

        client_ip = _get_client_ip(request)
        now = time.time()
        cutoff = now - self.window

        _HARD_CAP = 10_000  # Absolute max tracked IPs to prevent memory exhaustion

        async with self._lock:
            # Prune old entries for this IP
            timestamps = self._requests[client_ip]
            self._requests[client_ip] = [t for t in timestamps if t > cutoff]

            # Evict stale IPs periodically to prevent unbounded growth
            if len(self._requests) > 100:
                stale = [ip for ip, ts in self._requests.items() if not ts or ts[-1] < cutoff]
                for ip in stale:
                    del self._requests[ip]

            # Hard cap: if still too many IPs after stale eviction, drop oldest entries
            if len(self._requests) > _HARD_CAP:
                # Sort by most recent request timestamp, keep newest _HARD_CAP // 2
                sorted_ips = sorted(
                    self._requests.items(),
                    key=lambda kv: kv[1][-1] if kv[1] else 0,
                    reverse=True,
                )
                keep = _HARD_CAP // 2
                self._requests.clear()
                for ip, ts in sorted_ips[:keep]:
                    self._requests[ip] = ts

            current_count = len(self._requests[client_ip])

            # Read limit dynamically so runtime config changes take effect
            effective_limit = config.RATE_LIMIT_RPM
            if current_count >= effective_limit:
                # Find earliest expiry for reset time
                reset_time = int(self._requests[client_ip][0] + self.window)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Try again later."},
                    headers={
                        "X-RateLimit-Limit": str(effective_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                    },
                )

            self._requests[client_ip].append(now)
            remaining = effective_limit - current_count - 1
            reset_time = int(now + self.window)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(effective_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        return response


app.add_middleware(RateLimitMiddleware, max_requests=config.RATE_LIMIT_RPM, window_seconds=60)


class UserActivityMiddleware(BaseHTTPMiddleware):
    """Track last user activity for idle detection (dream mode trigger)."""

    async def dispatch(self, request: Request, call_next):
        # Only track user-facing endpoints (chat, voice, actions)
        path = request.url.path
        if any(path.startswith(p) for p in ("/api/chat", "/api/voice", "/api/actions")):
            from app.core.brain import get_services
            try:
                svc = get_services()
                if svc.monitor_store:
                    ts = datetime.now(timezone.utc).isoformat()
                    db = svc.monitor_store._db
                    asyncio.create_task(asyncio.to_thread(
                        db.execute,
                        "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                        ("last_user_activity", ts),
                    ))
            except Exception as e:
                logger.warning("User activity tracking failed: %s", e)
        return await call_next(request)


app.add_middleware(UserActivityMiddleware)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app.add_middleware(SecurityHeadersMiddleware)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Assign a short correlation ID to each request for log tracing."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_var.set(req_id)
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


app.add_middleware(CorrelationIDMiddleware)

# CORS — config-driven origins (default "*" for dev, restrict in production)
_origins = [o.strip() for o in config.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Mount routers
from app.api.system import router as system_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.learning import router as learning_router
from app.api.monitors import router as monitors_router
from app.api.actions import router as actions_router
from app.api.daemon import router as daemon_router

app.include_router(system_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(learning_router, prefix="/api")
app.include_router(monitors_router, prefix="/api")
app.include_router(actions_router, prefix="/api")
app.include_router(daemon_router, prefix="/api")

from app.api.exports import router as exports_router
app.include_router(exports_router, prefix="/api")

from app.api.dossiers import router as dossiers_router
app.include_router(dossiers_router, prefix="/api")

from app.api.agent import router as agent_router
app.include_router(agent_router, prefix="/api")

from app.api.events import router as events_router
app.include_router(events_router, prefix="/api")

if config.ENABLE_VOICE or getattr(config, "ENABLE_TTS", False):
    from app.api.voice import router as voice_router
    app.include_router(voice_router, prefix="/api")
    # Liveness, not just the flag (2026-08-19): TTS was announced "on" while
    # piper wasn't installed AND the voice model file was missing — every
    # synthesize call 503'd. Optional pathways must verify they can actually run.
    _tts_state = "off"
    if getattr(config, "ENABLE_TTS", False):
        try:
            from app.core.voice import _HAS_PIPER
            from pathlib import Path as _P
            _model_ok = _P(getattr(config, "TTS_MODEL_PATH", "") or "").exists()
            if _HAS_PIPER and _model_ok:
                _tts_state = "on"
            else:
                _tts_state = (f"BROKEN (piper installed: {_HAS_PIPER}, "
                              f"model file exists: {_model_ok}) — synthesize will 503; "
                              "install piper-tts + model or set ENABLE_TTS=false")
        except Exception as _e:
            _tts_state = f"BROKEN ({_e})"
    logger.info(
        "Voice API enabled (STT model: %s, TTS: %s)",
        config.WHISPER_MODEL_SIZE,
        _tts_state,
    )
    if _tts_state.startswith("BROKEN"):
        logger.warning("TTS liveness check failed: %s", _tts_state)
