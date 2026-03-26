# Nova — Development Guide

## What This Is

Nova is a sovereign personal AI assistant with multi-provider LLM support (Ollama, OpenAI, Anthropic, Google).
Default: FastAPI backend + Ollama (Qwen3.5:27b) on RTX 3090. Supports MCP (Model Context Protocol) for external tools.
It learns from corrections, remembers user facts, uses tools, and generates DPO training data for fine-tuning.
~79 files, not 238. Learning is the product.

## Architecture (Single Pipeline, No Framework)

```
User query -> brain.think()
  -> load context (history + facts + lessons + skills)
  -> classify intent (regex, no LLM)
  -> retrieve documents if needed (ChromaDB + FTS5 + RRF)
  -> build system prompt (8 prioritized blocks)
  -> generate response (LLM provider: Ollama, OpenAI, Anthropic, or Google)
  -> tool loop if tool call detected (max 5 rounds)
  -> stream tokens via SSE
  -> post-response: corrections, fact extraction, skill creation
```

No LangChain. No LangGraph. Just async Python and httpx to Ollama.

## Key Files

| File | Purpose |
|------|---------|
| `app/core/brain.py` | THE core: `think()` generator -- the entire pipeline |
| `app/core/llm.py` | Provider-agnostic LLM interface: `invoke_nothink()`, `generate_with_tools()`, JSON extraction |
| `app/core/providers/` | LLM backends: `ollama.py`, `openai.py`, `anthropic.py`, `google.py` |
| `app/tools/mcp.py` | MCP client: discovers external MCP tools, wraps as BaseTool |
| `app/mcp_server.py` | MCP server: exposes Nova as MCP server (memory, KG, lessons, docs) |
| `app/core/prompt.py` | System prompt builder (8 blocks with truncation priority) |
| `app/core/memory.py` | ConversationStore + UserFactStore + fact extraction |
| `app/core/learning.py` | Correction detection (regex+LLM), lessons, training data |
| `app/core/skills.py` | Skill store, trigger matching, skill extraction |
| `app/core/retriever.py` | ChromaDB vector + SQLite FTS5 BM25 + RRF fusion |
| `app/core/access_tiers.py` | Tier-aware restrictions: sandboxed/standard/full/none |
| `app/core/injection.py` | Prompt injection detection (heuristic, 4 categories) |
| `app/core/skill_export.py` | Skill import/export with HMAC-SHA256 signing |
| `app/channels/whatsapp.py` | WhatsApp adapter — webhook-based via Business API |
| `app/channels/signal.py` | Signal adapter — polling via signal-cli REST API |
| `app/core/task_manager.py` | Background task manager (submit, cancel, auto-prune) |
| `app/tools/background_task.py` | BackgroundTaskTool — submit/status/list/cancel |
| `app/tools/desktop.py` | Desktop automation (screenshot, click, type, hotkey) |
| `app/core/voice.py` | WhisperTranscriber — local speech-to-text |
| `app/api/voice.py` | Voice API endpoints (transcribe, chat) |
| `app/config.py` | ~85 settings from .env (frozen dataclass) |
| `app/database.py` | SafeDB singleton wrapping sqlite3 |
| `app/tools/base.py` | BaseTool + ToolResult + ToolRegistry |
| `app/api/chat.py` | POST /chat/stream (SSE) + POST /chat (sync) |
| `app/api/learning.py` | Lesson/skill/finetune endpoints |
| `app/api/system.py` | Health, status, export/import |

## Critical Patterns

### invoke_nothink()
`app/core/llm.py` -- Suppresses Qwen3.5 thinking mode via assistant prefix trick.
ALL background tasks (correction extraction, fact extraction, title generation, summarization) use this.
Main responses use `generate_with_tools()` (thinking suppressed for speed) or `stream_with_thinking()`
(thinking enabled for extended reasoning, controlled by `ENABLE_EXTENDED_THINKING`).

### JSON from LLM
- `repeat_penalty` must be **1.1** (not 1.5) for `json_mode=True` -- higher values mangle JSON
- Always pass `format: "json"` to Ollama for structured extraction
- Use `extract_json_object()` as fallback parser (balanced brace matching)

### Tool Calling (Hybrid: Native + Text)
Cloud providers (OpenAI, Anthropic, Google) use native structured tool calls returned in `result.tool_call`.
Tools are now passed through `stream_with_thinking()` for all cloud providers, enabling streaming tool calls.
Ollama uses prompt-based text extraction (Qwen3.5 native tool calling is broken, GitHub #14493):
```
{"tool": "tool_name", "args": {"param": "value"}}
```
`brain.py` checks `result.tool_call` first (structured), then falls back to `_extract_tool_calls()` (text parsing).

### Provider-Aware Prompt Building
`build_system_prompt()` accepts `provider` and `registered_tool_names` params:
- **Date block**: Full emphatic repetition for Ollama (date confusion workaround), condensed for cloud providers
- **Self-attribution**: Emphatic "REAL, live results" framing for Ollama, neutral for cloud
- **Tool examples**: Filtered to only registered tools (no phantom examples)

### Provider Base URLs
All cloud provider base URLs are configurable via config: `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`, `ANTHROPIC_API_VERSION`. Supports self-hosted endpoints and proxy setups.

### Correction Detection (2-stage)
1. **Regex pre-filter** -- `is_likely_correction()` in `learning.py` is the single source of truth
2. **LLM confirmation** -- `detect_correction()` uses `invoke_nothink(json_mode=True)` to extract

Brain.py imports `is_likely_correction` from `learning.py`. Do NOT duplicate patterns.

### History Walking Bug Fix
In `brain.py` step 13, the correction handler must **skip 1 assistant message** because step 11 already saved the new response before the correction handler runs. The second-from-end assistant message is the wrong answer.

### System Prompt Blocks (Priority Order)
```
[NEVER TRUNCATE] Block 1: Identity + Reasoning Methodology
[NEVER TRUNCATE] Block 2: User Facts
[NEVER TRUNCATE] Block 3: Learned Lessons
[NEVER TRUNCATE] Block 8: Date/Time (provider-aware: full for Ollama, condensed for cloud)
[TRUNCATE LAST]  Block 4: Tool Descriptions + Examples (filtered to registered tools only)
[TRUNCATE MID]   Block 5: Skills / Retrieved Context
[TRUNCATE FIRST] Block 7: Conversation Summary
```

### User Fact Source Authority
`memory.py` enforces a source hierarchy when overwriting facts: `user (4) > correction (3) > inferred (2) > extracted (1)`.
Lower-authority sources cannot overwrite higher-authority facts.

### SafeDB.execute() Returns Cursor
Always truthy. Use `fetchone()` / `fetchall()` for SELECTs.

### Access Tiers (`SYSTEM_ACCESS_LEVEL`)
- **sandboxed** (default): Shell blocks system + interpreter commands. File ops only `/data`. Code blocks os/subprocess/socket/httpx/requests.
- **standard**: Shell blocks system commands. File allows `/data`, `/tmp`, `/home/nova`. Code allows pathlib/os.path.
- **full**: Only container-escape commands blocked. Minimal code restrictions.
- **none**: All restrictions disabled. No blocked commands, imports, builtins, or path checks.

### Tool Timeout
`TOOL_TIMEOUT` (default 120s) controls the per-tool execution timeout in `brain.py`.
`GENERATION_TIMEOUT` (default 480s) controls LLM generation timeout.

### Route Ordering
Register `/path/literal` routes BEFORE `/path/{param}` in FastAPI to avoid path conflicts.

## Heartbeat & Self-Improvement

### Monitor System (`app/monitors/heartbeat.py`)
Background loop checks monitors on schedule, detects changes, sends alerts via Discord/Telegram/WhatsApp/Signal.

**51 default monitors** (seeded on first startup):
- **Operational** (6): Morning Check-in (daily), System Health (2h), Self-Reflection (daily), System Maintenance (daily), Fine-Tune Check (weekly), Auto-Monitor Detector (daily)
- **Self-Improvement** (3): Lesson Quiz (6h), Skill Validation (12h), Curiosity Research (1h)
- **Financial Intelligence** (7): Finance (12h), Crypto & Web3 (6h), DeFi & Protocols (8h), Whale Watch (6h), Top Trades (8h), Commodities & Forex (6h), Earnings (8h)
- **International** (6): China Tech (8h), Russia & E.Europe (12h), Middle East (12h), India (12h), Europe & EU (12h), Geopolitics (8h)
- **Science/Tech** (9): Science, Technology, AI & ML, Space, Quantum, Robotics, Physics, Biotech, Semiconductors (8-24h)
- **Policy/Security** (4): US Policy, Cybersecurity, Energy & Climate, Defense & Military (12h)
- **Culture/Local** (5): Sports (6h), Entertainment (12h), Social Media (12h), LA Local (12h), Climate & Weather (12h)
- **Developer/Business** (3): Open Source & GitHub (12h), Developer Ecosystem (12h), Startups & VC (12h)
- **Global** (5): World Awareness (4h), Current Events (8h), Economics & Markets (12h), Supply Chain (12h), Research Frontiers (24h)
- **Geographic** (3): Latin America (24h), Africa & Emerging (24h), Research Frontiers (24h)

All query-type monitors auto-extract KG triples. All prompts anchored to "past 24-48 hours" with today's date injected.

### Self-Improvement Pipeline
1. **Reflexion** (`reflexion.py`): Heuristic + LLM critique after each response. Failures stored and retrieved on similar future queries.
2. **Curiosity** (`curiosity.py`): Gaps detected during conversation → queued → researched by Curiosity Research monitor → findings become KG triples + lessons.
3. **Domain Studies**: Scheduled web searches → results stored as KG triples via `_extract_kg_triples()`.
4. **KG Auto-Curation**: Heuristic + LLM pass at startup removes garbage triples. Daily maintenance decays stale facts.
5. **Success Patterns**: Good responses (quality ≥ 0.8) stored as success reflexions, retrieved for positive reinforcement.
6. **Recurring Failure Promotion**: 3+ similar failures auto-promote to a lesson.

### Key Details
- KG extraction fires for all query-type monitors except Morning Check-in and Self-Reflection
- Auto-monitors use query type (brain.think()) not search (raw web_search)
- Cross-monitor feedback loops run during daily maintenance: quiz failures→curiosity re-research, degrading skills→early validation
- Decay (KG, reflexions, lessons) runs via the daily maintenance monitor, not at startup
- Skill success rate uses EMA (α=0.15) — recent failures degrade quickly
- Lesson confidence uses dampened adjustments — `delta / (1 + times_helpful)`

## Quality Rubric
- **9-10**: Handles edge cases, learns from correction, uses tools naturally
- **7-8**: Correct answer, uses context, conversational tone
- **5-6**: Correct but generic, ignores context or user facts
- **3-4**: Wrong or hallucinated, doesn't use tools when it should
- **1-2**: Broken, crashes, or produces garbage

## Rules
1. Never add features without asking. The rebuild is lean by design.
2. Never add config flags without approval. ~45 settings, not 281.
3. Never rate quality without evidence (test output, logs, actual behavior).
4. If unsure whether something is broken, TEST IT before changing it.
5. Port patterns from nova/ when they're battle-tested. Don't reinvent.
6. No duplicate correction patterns. `learning.py` is the single source of truth.
7. Lessons must have all fields: `topic`, `correct_answer`, `wrong_answer`, `lesson_text`.
8. DPO training pairs: query=original question, chosen=correct, rejected=wrong.
9. Facts are extracted, not hallucinated. Only extract from explicit user statements.
10. Context budget: 6000 tokens max (MAX_SYSTEM_TOKENS in prompt.py). Summarize older messages, keep 6 recent.

## Dependencies

- **Runtime**: FastAPI, uvicorn, httpx, chromadb, sympy, pydantic
- **LLM (default)**: Ollama 0.17.5+ with qwen3.5:27b (17GB VRAM)
- **LLM (cloud, optional)**: OpenAI (gpt-4o), Anthropic (claude-sonnet), Google (gemini-2.0-flash)
- **MCP (optional)**: `mcp` package for Model Context Protocol tool integration
- **Embedding**: nomic-embed-text-v2-moe (0.5GB VRAM)
- **Fine-tuning** (separate venv): unsloth, trl, torch (see `scripts/requirements-finetune.txt`)

## Docker

```
docker compose up          # Start all services
docker compose stop ollama # Free VRAM for fine-tuning
```

Services: nova-ollama (11434), nova-app (8000), nova-searxng (8888)

## Testing

```bash
# In container
docker exec nova-app sh -c "python -m pytest tests/ -v"

# Copy files if needed
docker cp tests/. nova-app:/app/tests/
docker cp pytest.ini nova-app:/app/pytest.ini
```

Mock pattern: `patch("app.core.brain.llm")` for brain, `patch("app.core.memory.llm")` for memory.

## Fine-Tuning

```bash
# Manual fine-tuning
curl http://localhost:8000/api/learning/finetune/status  # Check readiness
docker compose stop ollama                                # Free VRAM
python scripts/finetune.py --dry-run                     # Preview
python scripts/finetune.py --export-gguf                 # Train + GGUF
docker compose start ollama                              # Restart

# Automated pipeline (includes A/B eval)
python scripts/finetune_auto.py --check                  # Check if ready
python scripts/finetune_auto.py                          # Full auto: train + eval + deploy
python scripts/finetune_auto.py --eval-only              # Just run A/B eval
python scripts/finetune_auto.py --force --skip-eval      # Force train, no eval
```

### A/B Evaluation Harness (`scripts/eval_harness.py`)
Compares base vs fine-tuned model on holdout queries. Uses LLM-as-judge with randomized A/B ordering to avoid position bias. Candidate must win >50% and have positive avg preference to be deployed.

### Automated Pipeline (`scripts/finetune_auto.py`)
8-step pipeline: readiness check → load data → stop Ollama → DPO train → GGUF export → restart Ollama → A/B eval → deploy/reject. Records all runs to `run_history.json`.

## MCP Server

Nova exposes its intelligence as MCP tools for external agents (Claude Code, Cursor, etc.):

```bash
python scripts/mcp_server_runner.py                     # Runs over stdio
```

**5 exposed tools**: `nova_memory_query`, `nova_knowledge_graph`, `nova_lessons`, `nova_document_search`, `nova_facts_about`

Sample config for Claude Code: `mcp_configs/nova_mcp.json`

## Channels

4 channel adapters, all following the same pattern: `__init__`, `start()`, `close()`, `send_alert()`, `_handle_query()`.

| Channel | Adapter | Mode | Config Keys |
|---------|---------|------|-------------|
| Discord | `app/channels/discord.py` | Bot (websocket) | `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID` |
| Telegram | `app/channels/telegram.py` | Bot (polling) | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_ALLOWED_USERS` |
| WhatsApp | `app/channels/whatsapp.py` | Webhook (FastAPI router) | `WHATSAPP_API_URL`, `WHATSAPP_API_TOKEN`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_PHONE_ID`, `WHATSAPP_CHAT_ID`, `WHATSAPP_ALLOWED_USERS` |
| Signal | `app/channels/signal.py` | Polling (signal-cli REST API) | `SIGNAL_API_URL`, `SIGNAL_PHONE_NUMBER`, `SIGNAL_CHAT_ID`, `SIGNAL_ALLOWED_USERS`, `SIGNAL_POLL_INTERVAL` |

All channels: phone-number allowlisting (empty = allow all), message splitting for long responses, graceful connection failure handling.

## Temporal Knowledge Graph

KG facts now track change over time:
- `valid_from` / `valid_to` — when a fact was valid
- `provenance` — which conversation/source created it
- `superseded_by` — FK to the fact that replaced it
- Contradicting facts are **superseded** (not deleted), creating a temporal trail
- `query_at(entity, at_time)` — query facts valid at a point in time
- `get_fact_history(subject, predicate)` — all versions of a fact over time
- `get_changes_since(since)` — what changed recently

## Security

### Prompt Injection Detection (`app/core/injection.py`)
Heuristic-based detection on all ingested content (web search, HTTP fetch, external skills). 4 categories:
1. Role override patterns (weight 0.4)
2. Instruction injection patterns (weight 0.3)
3. Delimiter abuse patterns (weight 0.2)
4. Encoding tricks (weight 0.1)

Suspicious content is wrapped with a warning prefix, not stripped. Gated by `ENABLE_INJECTION_DETECTION`.

### Skill Signing (`app/core/skill_export.py`)
Skills can be exported/imported with HMAC-SHA256 signatures:
```bash
python scripts/skill_export.py generate-key --output key.hex
python scripts/skill_export.py export --output skills.json --sign-key key.hex
python scripts/skill_export.py import --input skills.json --verify-key key.hex
```
Set `REQUIRE_SIGNED_SKILLS=true` to reject unsigned skill imports.

### Security Headers
All responses include: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Content-Security-Policy`, `Referrer-Policy`.

### Rate Limiting
60 req/min per IP with `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers.

### Input Validation
All API endpoints validate input lengths, formats, and types. Pydantic validators on request models, regex guards on query parameters.

## Background Tasks (`app/core/task_manager.py`)

In-process `asyncio.create_task` system for long-running work that shouldn't block conversation.

- `TaskManager`: submit(), get_status(), list_tasks(), cancel(), cancel_all()
- Max concurrent limit (`MAX_BACKGROUND_TASKS`, default 5), auto-timeout (`BACKGROUND_TASK_TIMEOUT`, default 300s)
- Auto-pruning keeps last 50 completed tasks
- `BackgroundTaskTool` (`app/tools/background_task.py`): 4 actions — submit, status, list, cancel
- Submit spawns ephemeral `brain.think()` calls for parallel research

## New Config Fields (Deep Audit)
- `MAX_QUERY_LENGTH` (50000) — query length validation in brain.think()
- `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL` — provider base URLs
- `ANTHROPIC_API_VERSION` — Anthropic API version header
- `TRUSTED_PROXY` — enable X-Forwarded-For only when set

## Version Source of Truth
`app/__init__.__version__` is the single source. Imported by system.py and schema.py.

## Desktop Automation (`app/tools/desktop.py`)

PyAutoGUI-based GUI control. Gated by `ENABLE_DESKTOP_AUTOMATION` + access tier (full/none only).

- 6 actions: screenshot, click, type, move, hotkey, scroll
- Rate limiting via `DESKTOP_CLICK_DELAY` (default 0.5s)
- Dangerous hotkey blocking (alt+f4, ctrl+alt+delete)
- Requires X11 display server (`DISPLAY` env var)
- All PyAutoGUI calls run in thread executor (non-blocking)
- Lazy import — gracefully handles missing display or pyautogui

## Voice Interface (`app/core/voice.py`, `app/api/voice.py`)

Local Whisper STT (speech-to-text). Gated by `ENABLE_VOICE`.

- `WhisperTranscriber`: lazy model loading, async via `asyncio.to_thread`
- `POST /api/voice/transcribe` — upload audio → JSON transcription
- `POST /api/voice/chat` — upload audio → transcribe → stream SSE response
- Model size via `WHISPER_MODEL_SIZE` (default "base"), max duration via `VOICE_MAX_DURATION` (300s)
- 25MB file size limit, audio extension validation
- GPU auto-unloaded on shutdown
