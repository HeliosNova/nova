"""Nova_ — Sovereign Personal AI.

FastAPI application entry point.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

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
from app.tools.file_ops import FileOpsTool
from app.tools.http_fetch import HttpFetchTool
from app.tools.knowledge import KnowledgeSearchTool
from app.tools.memory_tool import MemorySearchTool
from app.tools.browser import BrowserTool
from app.tools.monitor_tool import MonitorTool
from app.tools.screenshot import ScreenshotTool
from app.tools.shell_exec import ShellExecTool
from app.tools.web_search import WebSearchTool
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
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    handlers=[_log_handler],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # --- Startup ---
    logger.info("Nova_ starting up...")

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

    # Core services
    conversations = ConversationStore(db)
    user_facts = UserFactStore(db)
    learning = LearningEngine(db)
    skills = SkillStore(db)

    # Knowledge graph
    from app.core.kg import KnowledgeGraph
    kg = KnowledgeGraph(db)
    logger.info("Knowledge graph initialized")

    # Reflexion store
    from app.core.reflexion import ReflexionStore
    reflexions = ReflexionStore(db)
    logger.info("Reflexion store initialized")

    # KG auto-curation (heuristic pass runs inline, LLM pass runs in background)
    # Note: KG/reflexion decay is handled by the daily maintenance monitor
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

        asyncio.create_task(_bg_kg_curate())
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
        ("CalculatorTool", lambda: CalculatorTool()),
        ("HttpFetchTool", lambda: HttpFetchTool()),
        ("KnowledgeSearchTool", lambda: KnowledgeSearchTool(retriever=retriever)),
        ("CodeExecTool", lambda: CodeExecTool()),
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

    # Background task manager
    task_manager = TaskManager(
        max_concurrent=config.MAX_BACKGROUND_TASKS,
        task_timeout=config.BACKGROUND_TASK_TIMEOUT,
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
    monitor_store = None
    if config.ENABLE_HEARTBEAT:
        from app.monitors.heartbeat import MonitorStore
        monitor_store = MonitorStore(db)
        registry.register(MonitorTool(monitor_store=monitor_store))
        registry.register(ReminderTool(monitor_store=monitor_store))
        seeded = monitor_store.seed_defaults()
        if seeded:
            logger.info("Seeded %d default monitor(s)", seeded)
        logger.info("Monitor store initialized (%d monitors)", len(monitor_store.list_all()))

    # Curiosity engine + topic tracker
    curiosity_queue = None
    topic_tracker = None
    if config.ENABLE_CURIOSITY:
        from app.core.curiosity import CuriosityQueue, TopicTracker
        curiosity_queue = CuriosityQueue(db)
        topic_tracker = TopicTracker(db)
        logger.info("Curiosity engine initialized")

    # External skills loader (AgentSkills / OpenClaw)
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

    # Decay confidence on stale lessons
    try:
        decayed = learning.decay_stale_lessons(days=30)
        if decayed:
            logger.info("Decayed %d stale lessons", decayed)
        else:
            logger.info("Lesson decay: all lessons are fresh")
    except Exception as e:
        logger.warning("Lesson decay failed: %s", e)

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
    heartbeat_loop = None
    daily_digest = None
    if config.ENABLE_HEARTBEAT and monitor_store:
        from app.monitors.heartbeat import HeartbeatLoop
        from app.monitors.proactive import DailyDigest
        heartbeat_loop = HeartbeatLoop(
            monitor_store,
            discord_bot=discord_bot,
            telegram_bot=telegram_bot,
            whatsapp_bot=whatsapp_bot,
            signal_bot=signal_bot,
        )
        heartbeat_loop.start()
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
            )
            daily_digest.start()
            logger.info("Daily digest started (hour=%d)", config.DIGEST_HOUR)

    logger.info("Nova_ ready.")

    yield

    # --- Shutdown ---
    logger.info("Nova_ shutting down...")

    # Stop heartbeat + proactive
    if heartbeat_loop:
        heartbeat_loop.stop()
    if daily_digest:
        daily_digest.stop()

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

    # Unload Whisper model
    if config.ENABLE_VOICE:
        from app.core.voice import unload_transcriber
        unload_transcriber()

    # Cancel background tasks
    await task_manager.cancel_all()

    # Close MCP sessions
    if mcp_manager:
        await mcp_manager.close()

    # Close retriever (ChromaDB client cleanup)
    if retriever:
        retriever.close()

    # Close HTTP fetch connection pool
    from app.tools.http_fetch import close_http_client
    await close_http_client()

    await close_client()
    from app.database import close_all
    close_all()


app = FastAPI(
    title="Nova_",
    version="1.0.0",
    description="Sovereign Personal AI",
    lifespan=lifespan,
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

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window

        async with self._lock:
            # Prune old entries for this IP
            timestamps = self._requests[client_ip]
            self._requests[client_ip] = [t for t in timestamps if t > cutoff]

            # Evict stale IPs periodically to prevent unbounded growth
            if len(self._requests) > 100:
                stale = [ip for ip, ts in self._requests.items() if not ts or ts[-1] < cutoff]
                for ip in stale:
                    del self._requests[ip]

            current_count = len(self._requests[client_ip])

            if current_count >= self.max_requests:
                # Find earliest expiry for reset time
                reset_time = int(self._requests[client_ip][0] + self.window)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Try again later."},
                    headers={
                        "X-RateLimit-Limit": str(self.max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                    },
                )

            self._requests[client_ip].append(now)
            remaining = self.max_requests - current_count - 1
            reset_time = int(now + self.window)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        return response


app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self'"
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

app.include_router(system_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(learning_router, prefix="/api")
app.include_router(monitors_router, prefix="/api")
app.include_router(actions_router, prefix="/api")

if config.ENABLE_VOICE:
    from app.api.voice import router as voice_router
    app.include_router(voice_router, prefix="/api")
    logger.info("Voice API enabled (model: %s)", config.WHISPER_MODEL_SIZE)
