# Nova — Development Guide

## What This Is

Nova is a sovereign personal AI assistant. Ollama only — runs entirely on your hardware.
Default: FastAPI backend + Ollama on RTX 3090. Production model is `nova-ft` (9B DPO fine-tune,
v16 deployed 2026-04-22) — set via `LLM_MODEL=nova-ft`. The 9B is what gets fine-tuned; the 27B
is supported as an alternate backend but isn't fine-tuned. Supports MCP (Model Context Protocol)
for external tools. It learns from corrections, remembers user facts, uses tools, and generates
training data for DPO fine-tuning (SimPO was the v1 loss; trl 0.24+ uses sigmoid). ~129 Python
files in `app/`. Learning is the product.

### Reality check (audit 2026-05-30) — read before touching any learning / fine-tune claim
- **The memory loop is the product, not fine-tuning.** Nova "learns" by storing each correction as a
  lesson + a temporal-KG fact that are retrieved into the prompt (in-context). Proven by the
  `memory-learning` eval category (`evals/suite.yaml` + `EvalHarness._run_memory_task`): on the live
  9B it causally fixes the majority of seeded corrections (`memory_causal_fix_rate` ≈ 0.5–0.67 at
  baseline; misses are paraphrase-retrieval + generation-behavior gaps, tracked as work items).
- **Weight fine-tuning is experimental and unproven.** Across every `run_history.json`: 0 successful
  train→A/B→deploy. The one honest A/B (independent cross-family judge `llama3.1:8b`, position-swapped,
  4-dimension) shows `nova-ft` **ties** its base `qwen3.5:9b` (≈8/10 ties, +0.03 pref). Keep
  `ENABLE_AUTO_FINETUNE=false`. Treat FT as a *style/behavior* experiment only — never claim it makes
  the model "smarter."
- **FT deploy gate:** a candidate promotes only if it wins `scripts/eval_harness.py` with
  `EVAL_JUDGE_MODEL` set to a *different-family* local model (no self-judging; swap + multi-dimension on).
- **Config override gotcha:** `/data/config_overrides.json` overrides `.env`. A stale
  `LLM_MODEL=nova-ft-v18-q8` (a fine-tune that never built) sat there and **404'd every generation**;
  fixed to `nova-ft` on 2026-05-30. Keep `LLM_MODEL` pointed at a tag that exists in Ollama.
- Do **not** re-inflate README / MODEL_CARD / docs/launch_posts back to "fine-tunes itself into a
  smarter model" — that claim is disproven and was realigned 2026-05-30.
- **Memory-loop quality (WS2, 2026-05-30):** two fixes raised `memory_causal_fix_rate` 0.67→0.83:
  (1) `claim_validator.build_evidence` now includes `lessons_text` — lesson-grounded answers were
  being silently stripped as "unsupported" (e.g. "Dr. X is based in <city>"), defeating the loop;
  (2) `ENABLE_EXTENDED_THINKING=false` in the override — nova-ft (9B) burned its budget in `<think>`
  and returned EMPTY answers / web-searched fictional facts instead of reading the injected lesson.
  Reasoning category is neutral (6/7 both ways; thinking-on also produced an empty there). Remaining
  miss is paraphrase **retrieval**.
- **EMBEDDER UPGRADE — done properly + measured (2026-06-09):** The earlier (2026-05-30) `nomic`
  swap was reverted because it was a *half-wired* hack (the `EMBEDDING_MODEL` config was dead — every
  collection silently used ChromaDB's 2020-era MiniLM-L6 default; no Ollama embedder was ever called).
  Fixed for real: a 2026 offline bake-off over 80 paraphrase→fact queries measured MiniLM **last** of
  five (`app/core/embedding.py` has the table); **bge-m3** won the deployable slot (symmetric, so it
  drops into ChromaDB's EF; r@3=1.00 vs MiniLM 0.95). Wired a gated `OllamaEmbeddingFunction`
  (`app/core/embedding.py`, falls back to MiniLM if the embedder isn't reachable → giveaway-safe),
  set `EMBEDDING_MODEL=bge-m3`, calibrated `KG_VECTOR_MAX_DISTANCE` from the measured distribution
  (correct matches at cosine 0.08–0.42), and did a clean full reindex (`_reindex_embeddings.py`) that
  also dropped ~3,400 stale/superseded KG vectors (chroma 5,916 → 2,490 active). **Result: kg-retrieval
  causal_fix 0.83 → 1.0** (every paraphrase resolves); lessons retrieval ranks the hardest keyword-free
  paraphrase **#1** (dist 0.386). The lesson vector *gate* was disproven for MiniLM (0.7==0.95).
  Consider conditional thinking (on for complex, off for recall) as a separate refinement.
- **ROOT CAUSE of the memory loop's fragility, found + fixed (2026-05-30):** `MIN_RRF_SCORE=0.015` was
  filtering out **keyword-only** lesson matches — a single keyword hit blends to ~0.0139 (`0.85 × 1/61`)
  < 0.015 — so lesson retrieval depended ENTIRELY on the ChromaDB vector index, which was found **empty**
  (`count=0`; the 175 lessons were never indexed / MiniLM degraded). Net: `get_relevant_lessons` returned
  `[]` for everything → the memory loop was silently broken (the earlier 0.83 only worked while a
  freshly-seeded lesson briefly sat in the vector index). Lowered `MIN_RRF_SCORE` 0.015→0.005 (config.py)
  so keyword retrieval works on its own → `memory_causal_fix_rate` **1.0 (6/6)**, robust without any vector
  index. **FOLLOW-UP:** reindex the lessons vector collection (`LearningEngine.reindex_lessons`) and verify
  the MiniLM embedder actually embeds, so vector *augments* keyword for harder paraphrases.
- **KG retrieval fixed + measured (2026-05-30, new `kg-retrieval` eval category):** `kg.get_relevant_facts`
  was effectively broken. (1) It loaded candidates with `ORDER BY confidence DESC LIMIT 500` while the KG
  had 2,734 valid facts (1,863 at conf≥0.95) — so a relevant fact outside that window returned `[]` (the
  model then hallucinated or said "I don't know", even for direct queries). Fixed: candidate cap → `MAX_KG_FACTS`
  so every valid fact is a candidate (kg.py:1036). (2) The RRF fusion (keyword+vector+PPR) was computed then
  OVERWRITTEN by a keyword-overlap-only pass (the fused ranking was discarded). Fixed: the fused result is now
  authoritative; neighbor-enrichment operates on it (kg.py ~1115-1188). Measured: **0.0 → 0.83**
  `memory_causal_fix_rate` on `kg-retrieval` — restored direct retrieval AND flipped all 3 single-entity
  paraphrases to pass (vector/PPR now actually rank). Remaining: paraphrase recall is bounded by MiniLM/PPR
  quality; one direct flips run-to-run on the model's web-search tool-choice (eval has tool-use variance —
  run it a few times for a stable mean). **The lessons retrieval (`learning.py:~477`) has the SAME latent
  `LIMIT 500`** — not biting yet (261 lessons) but fix it the same way before it grows past 500.

## Architecture (Single Pipeline, No Framework)

```
User query -> brain.think()
  -> load context (history + facts + lessons + skills)
  -> classify intent (regex, no LLM)
  -> retrieve documents if needed (ChromaDB + FTS5 + RRF)
  -> build system prompt (8 prioritized blocks)
  -> generate response (Ollama — local inference)
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
| `app/core/providers/` | LLM backend: `ollama.py` |
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
| `app/config.py` | ~150 settings from .env (frozen dataclass) |
| `app/database.py` | SafeDB singleton wrapping sqlite3 |
| `app/tools/base.py` | BaseTool + ToolResult + ToolRegistry |
| `app/api/chat.py` | POST /chat/stream (SSE) + POST /chat (sync) |
| `app/api/learning.py` | Lesson/skill/finetune endpoints |
| `app/api/system.py` | Health, status, export/import |
| `app/core/agent_loop.py` | Plan→act→critique→synthesize deliberation loop (`AgentLoop.solve()`) |
| `app/core/agent_workspace.py` | Persistent scratchpads keyed by query signature; survives restarts |
| `app/core/cross_monitor.py` | Find themes spanning 3+ different monitors; write KG cross_synthesis facts |
| `app/core/goal_deriver.py` | Mine capability_gap clusters / curiosity / failing skills → mint goals |
| `app/core/auto_tools.py` | Mine capability_gap clusters → ask LLM to write a Python tool → hot-register with live ToolRegistry (no rebuild) |
| `app/core/principles.py` | Distill 3+ agreeing lessons into KG facts that survive lesson decay |
| `app/tools/native_search.py` | Search engine: SearXNG (247 engines) + Wikipedia + DDG with intent-based category routing |

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
Ollama now uses native tool calling (Ollama 0.17+). Text extraction is kept as fallback:
```
{"tool": "tool_name", "args": {"param": "value"}}
```
`brain.py` checks `result.tool_call` first (structured), then falls back to `_extract_tool_calls()` (text parsing).

### Provider-Aware Prompt Building
`build_system_prompt()` accepts `provider` and `registered_tool_names` params:
- **Tool examples**: Filtered to only registered tools (no phantom examples)

### Provider Base URLs

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
[NEVER TRUNCATE] Block 8: Date/Time (date emphasis for local models)
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
`TOOL_TIMEOUT` (default 180s) controls the per-tool execution timeout in `brain.py`.
`GENERATION_TIMEOUT` (default 900s) controls LLM generation timeout.

### Route Ordering
Register `/path/literal` routes BEFORE `/path/{param}` in FastAPI to avoid path conflicts.

## Heartbeat & Self-Improvement

### Monitor System (`app/monitors/heartbeat.py`)
Background loop checks monitors on schedule, detects changes, sends alerts via Discord/Telegram/WhatsApp/Signal.

**50 default monitors** (seeded on first startup):
- **Operational** (5): Morning Check-in (daily), System Health (2h), System Maintenance (daily), Fine-Tune Check (weekly), Auto-Monitor Detector (daily)
- **Self-Improvement** (3): Lesson Quiz (6h), Skill Validation (12h), Curiosity Research (1h)
- **Financial Intelligence** (10): Finance (12h), Crypto & Web3 (6h), DeFi & Protocols (8h), Whale Watch (6h), Top Trades (8h), Commodities & Forex (6h), Earnings (8h), FOMC & Fed Watch (24h), SEC Insider Trading (12h), Economics & Markets (12h)
- **International** (6): China Tech (8h), Russia & E.Europe (12h), Middle East (12h), India (12h), Europe & EU (12h), Geopolitics (8h)
- **Science/Tech** (9): Science, Technology, AI & ML, Space, Quantum, Robotics, Physics, Biotech, Semiconductors (8-24h)
- **Policy/Security** (4): US Policy, Cybersecurity, Energy & Climate, Defense & Military (12h)
- **Intelligence** (7): Hacker News (8h), Product Hunt (24h), FDA Drug Approvals (24h), GitHub Security Advisories (12h), Government Contract Awards (24h), Health & Medicine (12h), Research Frontiers (24h)
- **Developer/Business** (3): Open Source & GitHub (12h), Developer Ecosystem (12h), Startups & VC (12h)
- **Global** (3): World Awareness (4h), Current Events (8h), Supply Chain (12h)
- **Geographic** (2): Latin America (24h), Africa & Emerging (24h)

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
2. Never add config flags without approval. Settings are managed in config.py.
3. Never rate quality without evidence (test output, logs, actual behavior).
4. If unsure whether something is broken, TEST IT before changing it.
5. (Reserved — the pre-rebuild `nova/` repo is no longer referenced; current code is the source of truth.)
6. No duplicate correction patterns. `learning.py` is the single source of truth.
7. Lessons must have all fields: `topic`, `correct_answer`, `wrong_answer`, `lesson_text`.
8. Training pairs: query=original question, chosen=correct, rejected=wrong. Used for SimPO (default) or DPO fine-tuning.
9. Facts are extracted, not hallucinated. Only extract from explicit user statements.
10. Context budget: 6000 tokens max (MAX_SYSTEM_TOKENS in prompt.py). Summarize older messages, keep 6 recent.

## Dependencies

- **Runtime**: FastAPI, uvicorn, httpx, chromadb, sympy, pydantic
- **LLM (default)**: Ollama 0.17.5+ with qwen3.5:27b (17GB VRAM)
- **MCP (optional)**: `mcp` package for Model Context Protocol tool integration
- **Embedding**: bge-m3 (1024-dim, ~1.2GB VRAM) via a gated Ollama EmbeddingFunction; falls back to
  ChromaDB's bundled all-MiniLM-L6-v2 if unreachable. Won a 2026 paraphrase-retrieval bake-off.
- **Fine-tuning** (separate venv): unsloth, trl, torch (see `scripts/requirements-finetune.txt`)

## Docker

```
docker compose up          # Start all services
docker compose stop ollama # Free VRAM for fine-tuning
```

Services: nova-ollama (11434), nova-app (8000), nova-searxng (8888)

### Container freshness — IMPORTANT

`/app` is **not** bind-mounted; it's baked into the image. After ANY change to
`app/**.py`, `evals/**.yaml`, or `scripts/__init__.py` you MUST rebuild:

```
docker compose build nova && docker compose up -d nova
```

Compose marks the container `read_only: true` so `docker cp` cannot patch files
in place — a rebuild is the only path. Verify before assuming changes are live:

```
bash scripts/check_container_freshness.sh
```

**External reindex → RESTART required.** If you rebuild a ChromaDB collection
from a *separate* process (e.g. an embedder migration via a `_reindex_*.py`
script that drops + recreates `kg_facts`/`lessons`/`documents`), the running
app still holds a **stale handle to the dropped collection** — its vector arm
silently returns nothing for pre-existing facts until you
`docker compose up -d nova --force-recreate`. (Symptom seen 2026-06-09: post-
reindex, "what do you know about <entity>?" answered "I don't have facts" with
`kg_facts_used=0`, despite the facts being present and a fresh-client probe
returning them; a restart fixed it.) Fresh installs are unaffected — they build
collections in-process at startup (`reindex_kg_facts`/`reindex_lessons`), so no
stale handle. Only an embedder/collection migration on an existing install hits
this. (Eval/seeded-fact flows are also unaffected: they add + read through the
same handle, so they stay self-consistent.)

Container staleness was a live bug for weeks (channel formatters, monitor
runner, KG fixes — all sat on disk while the container ran old code). Always
rebuild after editing the `app/` tree.

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

### Venv setup

Fine-tuning lives in a dedicated venv (`finetune_env/`) outside the Nova app
container — Unsloth, torch, and CUDA libs are too heavy for the read-only
container and need GPU access. Set up once per machine:

```bash
python -m venv finetune_env
# Windows
finetune_env\Scripts\activate
# Linux
source finetune_env/bin/activate

# Pinned high-level deps (recommended for normal use)
pip install -r scripts/requirements-finetune.txt

# OR bit-exact full freeze (119 pkgs; use when reproducing a past run)
pip install -r scripts/finetune_env.frozen.txt
```

Why pinned: between March (when nova-ft was trained) and May, library drift
(unsloth/trl/transformers) produced a -0.14 multi-agent eval regression with
the *same* data and recipe. Confirmed via baseline-only control train (v19).
`requirements-finetune.txt` is now exact-version pinned (`==`) so future
training runs are at least reproducible against each other. See
`memory/project_v19_clean_ab_2026_05_10.md`.

Two quirks to know about when calling finetune.py directly:

1. **AutoConfig hub-name fails for Qwen3.5-9B**: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")` raises `Unrecognized model in...` even when fully cached.
   Workaround: pass the snapshot folder path directly (`...models--Qwen--Qwen3.5-9B/snapshots/<sha>/`, or any local merged-weights dir with a valid `config.json` like `Qwen/Qwen3.5-9B/merged/`).

2. **Unsloth's `save_pretrained_gguf` is broken** — FIXED in `finetune.py` 2026-05-16 (task #48). Two failure modes hit on v9, v10, v17:
   (a) stale-cache: reuses cached merged shards from a prior run keyed on the snapshot path, producing a "fresh" GGUF that is actually the previous run's weights (symptom: new model's Ollama SHA matches previous run's);
   (b) dynamic-module loading: downloads a temp script that fails with `ModuleNotFoundError: No module named 'conversion'`.
   `export_gguf()` now bypasses Unsloth entirely and uses the manual chain: `PeftModel.from_pretrained → merge_and_unload → save_pretrained → ~/.unsloth/llama.cpp/convert_hf_to_gguf.py --outtype q8_0`. CPU-only merge to free GPU for next training run. See `scripts/finetune.py::export_gguf` and `_find_convert_hf_to_gguf`.

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

### Cadence policy (set 2026-05-14, task #23)
- **Threshold**: `FINETUNE_MIN_NEW_PAIRS=100`. At ~3.3 organic pairs/day this is roughly a monthly cycle.
- **Trigger**: the daily Fine-Tune Check heartbeat monitor reports `FINETUNE READY` once 100+ new pairs accumulate since the last deployed run. Inside the read-only nova-app container the auto-fire path is intentionally blocked (no unsloth/CUDA); the monitor emits a notify-only message with the host-run command (see task #42 / `heartbeat_loop._can_auto_finetune`).
- **Execution**: operator runs `python scripts/finetune_auto.py` from `finetune_env` venv on the host (Python 3.11; see `scripts/finetune_env_compat_patches.py` for the post-install patches required for trl 0.24 + transformers 5.x + llm_blender 0.0.2 to coexist).
- **A/B**: every cycle MUST A/B against the deployed model (`scripts/eval_harness.py`). Smaller deltas under ~5% of total training-pair count almost always A/B to ties — wait for the next cycle rather than burn a deploy.

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

KG facts now track change over time, **bitemporally** (since 2026-05-16 / task #29 —
Memento-style):

**Valid time** (when the fact was true in the world):
- `valid_from` / `valid_to`

**Transaction time** (when WE recorded the fact):
- `created_at` (explicitly set in `add_fact`, no longer the SQLite default)
- `superseded_at` (set when the fact's `superseded_by` is set)

Both timelines support independent queries — see `query_as_of` below.

Other fields:
- `provenance` — which conversation/source created it
- `superseded_by` — FK to the fact that replaced it
- Contradicting facts are **superseded** (not deleted), creating a temporal trail

Query API:
- `query_at(entity, at_time)` — facts world-valid at a point in time (legacy
  uni-temporal API; kept for back-compat)
- **`query_as_of(entity, *, valid_at=None, recorded_at=None)`** — bitemporal:
  - `valid_at=X, recorded_at=Y` → "what records did we have on Y that were
    world-valid on X" (audit query; ignores supersession-after-Y)
  - `recorded_at=Y` alone → "what did we believe on Y" (rows existing by Y
    AND not yet superseded by Y)
  - `valid_at=X` alone → "facts world-valid on X" including currently
    superseded ones
  - neither → currently believed facts (mirrors `query_at(None)`)
- `get_fact_history(subject, predicate)` — all versions of a fact over time
- `get_changes_since(since)` — what changed recently

Use case for bitemporal: reflexion auditing ("what did Nova believe about X
last Tuesday when it made decision Y?") and time-travel debugging.

## Multi-Agent Structural Decomposition

Structural decomposition is a separate path from `DelegateTool`. `DelegateTool` is LLM-driven, tool-call-based delegation. Structural decomposition fires heuristically before the LLM generates, based on query signals.

### Files

| File | Purpose |
|------|---------|
| `app/core/decomposer.py` | Signal scoring, gate logic (`should_decompose`), strategy selection, task extraction (`decompose_query`) |
| `app/core/agent_spawner.py` | `AgentSpawner`: executes `DecompositionPlan` via parallel/sequential/map-reduce `think()` sub-agents; `merge_agent_results()` |

### Config Flags

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_MULTI_AGENT` | `true` | Enable/disable structural decomposition entirely |
| `MULTI_AGENT_TRIGGER_THRESHOLD` | `4` | Minimum signal score to fire decomposition |
| `MAX_AGENT_COUNT` | `10` | Maximum sub-agents per decomposition |
| `AGENT_TASK_TIMEOUT` | `300` | Per-sub-agent timeout in seconds (RTX 3090 + 9B Q8) |
| `MAX_PARALLEL_AGENTS` | `6` | Concurrent sub-agent ceiling (semaphore) |

### Signal Scoring (threshold = 4)

| Signal | Points |
|--------|--------|
| Parallel markers (`compare`, `versus`, `side by side`, …) | +2 |
| Delegation words (`run in parallel agents`, `break this down`, …) | +2 |
| ≥ 3 distinct proper-noun candidates | +1 |
| Multiple question marks in query | +1 |
| Query length > 200 chars AND ≥ 2 tool-type keywords | +1 |
| Query was planned (`was_planned=True`) | +1 |

### Safety Gates (in `should_decompose`)

1. `ENABLE_MULTI_AGENT=false` → never fires
2. `_structural_depth.get() > 0` → sub-agents cannot themselves decompose (max depth = 1)
3. `intent in ("greeting", "correction")` → always skip
4. `score < MULTI_AGENT_TRIGGER_THRESHOLD` → skip

`ephemeral=True` is NOT a gate — the eval harness runs `think(ephemeral=True)` and must be able to test the decomposition path.

### Execution Strategies

- **parallel** (default): all sub-agents run concurrently under `asyncio.Semaphore(MAX_PARALLEL_AGENTS)` (default 6)
- **sequential**: sub-agents run in order; each receives prior results in `shared_findings`
- **map-reduce**: all-but-last tasks run in parallel; last task receives all map results

### SSE Events Emitted

```
AGENT_META    — decomposition plan summary (strategy, task count)
AGENT_START   — fired once per sub-agent at start
AGENT_DONE    — fired once per sub-agent when complete
AGENT_MERGE   — fired before merge LLM call
TOKEN         — merged response tokens (streamed)
TOOL_USE      — re-emitted for each unique tool used by sub-agents
DONE          — includes decomposed=True, agent_count=N
```

### _structural_depth ContextVar

Lives in `agent_spawner.py`, distinct from `DelegateTool`'s `_delegation_depth` in `delegate.py`.

- Set to `depth+1` before each `think()` sub-agent call via `token = _structural_depth.set(...)`
- Restored via `_structural_depth.reset(token)` in `finally` — correct even on exception/timeout
- `asyncio.gather()` copies the context to each Task at creation time, so parallel sub-agents are naturally isolated
- Sequential sub-agents must explicitly reset because they run in the same coroutine

### Eval Regression Probe

Three multi-agent tasks in `evals/suite.yaml` (category `multi-agent`):

1. `multi_agent_parallel_compare` — compare query, asserts `decomposition_fired`
2. `multi_agent_sequential_research` — search+calculate, asserts `decomposition_fired` + `tool_invoked: web_search`
3. `multi_agent_no_decompose` — "What is 2 plus 2?", asserts `answer_contains: 4` + `decomposition_not_fired`

**Regression detection**: `decomposition_rate` metric in the `multi-agent` category. Setting `MULTI_AGENT_TRIGGER_THRESHOLD=1` makes everything decompose → `multi_agent_no_decompose` fails `decomposition_not_fired` → `decomposition_rate` drifts from baseline → regression flagged.

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

## Prompt Self-Modification (`app/core/prompt_optimizer.py`)

Conservative, eval-gated prompt optimization. Modifies instruction strings only — NOT model weights (Ollama GGUF is static).

### How to Enable

```bash
ENABLE_PROMPT_SELF_MOD=true  # default false — opt-in required
```

The "Prompt Optimizer" heartbeat monitor (`check_type="prompt_analyzer"`, daily) is seeded disabled. Enable it alongside the flag:

```sql
UPDATE monitors SET enabled=1 WHERE name='Prompt Optimizer';
```

### Tunable Modules (6 total, hard allow-list)

| Module | Source constant | Tuned when |
|--------|-----------------|------------|
| `critique_prompt` | `reflexion._CRITIQUE_PROMPT` | reflexion calibration drifts |
| `extraction_prompt` | `learning._EXTRACTION_PROMPT` | reasoning/tool-use pass_rate drifts |
| `skill_extraction_prompt` | `skills.py` inline | skill-match hit_rate drifts |
| `merge_instruction_parallel` | `decomposer.py` inline | multi-agent pass_rate drifts |
| `merge_instruction_sequential` | `decomposer.py` inline | multi-agent pass_rate drifts |
| `kg_extraction_prompt` | `brain.py` inline | semantic-match recall drifts |

Everything else (`IDENTITY_AND_REASONING`, harness prompts, safety instructions, `META_PROMPT`) is **hardcoded and unmodifiable by design**.

### Firewall Guarantees

- **Allow-list**: `get_active_module()` returns `None` for any name not in `_SELF_MOD_ALLOWED_MODULES`.
- **Harness internal block**: `quiz_gen/quiz_answer/quiz_grade` are double-blocked by `_HARNESS_INTERNAL_MODULES`.
- **Baseline immutability**: `is_baseline=1` rows are never overwritten; the optimizer only adds new versions.
- **META_PROMPT**: Hardcoded constant, SHA-256 hash verified in `tests/test_prompt_optimizer.py::TestMetaPromptHashStability`.
- **Kill switch**: All writes and promotions check `ENABLE_PROMPT_SELF_MOD` at entry.

### Safety Caps (all configurable)

| Variable | Default | Meaning |
|----------|---------|---------|
| `PROMPT_MOD_MAX_PROPOSALS_PER_DAY` | `2` | Per-module write cap |
| `PROMPT_MOD_MAX_PENDING` | `3` | Max pending candidates per module |
| `PROMPT_MOD_MAX_PROMOTIONS_PER_DAY` | `2` | System-wide daily promotion cap |
| `PROMPT_MOD_MAX_DRIFT` | `0.25` | Jaccard distance from baseline (0=identical, 1=disjoint) |
| `PROMPT_MOD_MIN_IMPROVEMENT_PP` | `2.0` | Min improvement (pp) needed to pass shadow eval |
| `PROMPT_MOD_REGRESSION_TOLERANCE_PP` | `1.0` | Max allowed drop in non-target categories |
| `PROMPT_MOD_LATENCY_OVERHEAD_MAX` | `1.15` | Latency P95 overhead limit (1.15 = +15%) |

### Goodhart Firewall (critique_prompt candidates)

When shadow-testing a `critique_prompt` candidate, `_SCORING_OVERRIDES` pins the **baseline** critique for the reflexion scoring path. The candidate can only improve generation quality — it cannot inflate its own scores.

Calibration check: if `reflexion_p90 > 0.93` or `reflexion_mean > 0.80` during shadow eval, the run is marked `calibration_ok=False` and the candidate is rejected.

### Manual Rollback

```python
from app.database import get_db
from app.core.prompt_optimizer import PromptModuleStore
store = PromptModuleStore(db=get_db())
# Find the active promoted module id:
active = store.get_active("critique_prompt")
store.rollback(active.id)
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `PromptOptimizer: no new candidates` | Not enough eval history (< 3 runs) | Let eval harness run nightly for 3+ days |
| `drift too high` in logs | Candidate text too different from baseline | Reduce `PROMPT_MOD_MAX_DRIFT` or let LLM draft a smaller change |
| `calibration_ok=False` | Candidate inflates reflexion scores | Baseline scorer detected Goodhart loop — reject and quarantine expected |
| Module stuck in `quarantined` | Expiry defaults to 24h | Wait or manually `UPDATE prompt_modules SET status='candidate', quarantined_until=NULL WHERE id=?` |

## RLVR — Verifiable Signal Collection (`app/core/rlvr.py`)

Records ground-truth-style signals during normal operation so the next
GRPO/RLVR fine-tune cycle has reward data without re-grading. Distinct from
reflexion grading (LLM-judged) — these signals come from deterministic checks.

Signal types (hard allow-list): `tool_correct`, `json_valid`, `math_correct`,
`claim_grounded`, `quiz_correct`, `code_passes_tests`, `schema_match`.

Wired-in collectors:
- **brain.py tool dispatch** — records `tool_correct` (1.0/0.0) per tool result
- **brain.py post-validate_claims** — records `claim_grounded` (proportional to strips)
- **heartbeat_loop._execute_quiz** — records `quiz_correct` per quiz outcome

Public API:
```python
from app.core import rlvr
rlvr.record_signal("tool_correct", 1.0, query=..., response=..., evidence=...)
rlvr.aggregate(since_iso=None)        # per-type stats
rlvr.export_grpo_jsonl(path)           # JSONL for GRPO trainer
rlvr.mark_consumed([id, ...])          # called by trainer
```

Config: `ENABLE_RLVR_SIGNALS=true` (default on). Recording is fire-and-forget;
failures degrade silently so production chat is never blocked.

Storage: `verifiable_signals` table (migration 19) with
`consumed_for_training` flag so trainer passes don't double-count.

## Synthesis Sanitizer (`app/core/agent_loop.sanitize_synthesis`)

Runs after `_synthesize()` in AgentLoop. Strips scaffolding leaks the model
occasionally writes despite the tightened SYNTHESIZE_PROMPT:

  * "26/174 scenario", "step 3/5"          (plan IDs)
  * "as indicated by ... search logs"        (retrieval-pipeline refs)
  * "based on the completed analysis plan"   (scaffold name-drops)
  * "marked as requiring live configuration" (status-tag leak)
  * "(not specified here)"                   (parenthetical hedges)
  * "in the provided search results"         (tool-trace refs)

Two-pass design — phrase-level redaction runs FIRST so parentheticals get
redacted instead of triggering whole-sentence deletion; sentence-level
patterns then drop sentences that ARE primarily scaffold. Logs the count
of redactions so we can see when leaks are happening upstream.

Caught live 2026-05-08 in `deliberation_chain_of_reasoning` outputs;
17 unit tests + live deliberation probe verified clean.

## GRPO Trainer (`scripts/grpo_train.py` + `app/core/grpo_dataset.py` + `app/core/grpo_verifier.py`)

Closes the RLVR loop: consumes `verifiable_signals` rows into a fresh LoRA
adapter on the **9B base** (Qwen3.5-9B). Note: only the 9B is fine-tuned —
the 27B remains as the production inference model unless explicitly
trained separately.

### Pipeline
1. `grpo_dataset.build_groups()` reads unconsumed signals, normalizes queries,
   groups by `(normalized_query, signal_type)`. Computes within-group
   advantages = `(r_i - mean) / (std + 1e-8)`.
2. `grpo_train.py` decides:
   - **GRPO path** if ≥ 8 groups have size ≥ 4 and non-degenerate variance
   - **DPO fallback** otherwise: take (best, worst) per group → reuse `finetune.py`
3. Marks signals `consumed_for_training=1` on success.
4. Optional `--export-gguf` produces an Ollama-loadable GGUF.

### Verifier (`grpo_verifier.py`)
For online-rollout scoring inside GRPOTrainer. Replayable signal types:
  - `math_correct` — re-eval arithmetic via safe AST walk
  - `json_valid` — `json.loads` with embedded-object extraction
  - `schema_match` — balanced-brace JSON parse + tool-name match
Non-replayable types (`tool_correct`, `claim_grounded`, `quiz_correct`)
fall through to the precomputed reward from the dataset row.

### Usage
```bash
docker compose stop ollama                                # Free 17 GB VRAM
python scripts/grpo_train.py --dry-run                    # Show stats only
python scripts/grpo_train.py                              # Train (auto picks GRPO/DPO)
python scripts/grpo_train.py --force-grpo --export-gguf   # Insist on GRPO + GGUF
docker compose start ollama                               # Restart
```

### Dataset is small at first
RLVR signal accumulation needs days of normal traffic before there are
enough multi-rollout groups for true GRPO. Dry-run on 2026-05-08 found
3 trainable groups out of 88 from 477 signals — expected ramp.

## Removed experiments (v1.6.0)

A-HMAD debate (`app/core/debate.py`) and MAD-MM memory masking
(`agent_loop._mask_prior_observations`) were removed in v1.6.0: both were
default-off, never enabled in production, and never showed measured value.
The code is preserved on the `experiments/pre-v1.6-archive` branch if a
future eval gives a reason to revisit.

## Procedural Memory Consolidation (`app/core/dream.py::_consolidate_procedural_memory`)

Distinct from `principles.distill` (which only promotes lessons to KG facts).
This rewrites the lessons themselves: clusters near-duplicate lessons by
jaccard(topic_tokens) ≥ 0.6 AND jaccard(answer_tokens) ≥ 0.5, asks the LLM
to write one canonical generalized lesson, demotes the source members so
retrieval prefers the canonical.

Cap: 3 clusters per dream cycle. Persists `cluster_key` in
`procedural_clusters` (migration 19) so re-consolidation is skipped within
7 days.

Config: `ENABLE_PROCEDURAL_CONSOLIDATION=true` (default on).

## Two-Phase Dream Consolidation (`app/core/dream.py::consolidate_nrem`+`consolidate_rem`)

Opt-in SCM/SleepGate-style split of the dream loop's Phase 3 (task #30,
arxiv 2604.20943 + 2603.14517). Splits the kitchen-sink consolidate() into:

- **NREM (structural / deterministic)**: prune low-quality reflexions,
  compact KG chains, disable broken skills, handle failed curiosity,
  refresh stale facts, mine DPO pairs. No LLM calls. Fast (~5s typical).
- **REM (integrative / LLM-driven)**: promote high-quality reflexions to
  lessons, resolve lesson contradictions, procedural-memory generalization.
  All LLM-heavy. Slow (10-60s typical).

Two wins:
1. **Failure isolation** — if REM dies (LLM unreachable, prompt error),
   NREM's deterministic structural commits are preserved. Previously a
   mid-consolidate exception lost everything to Phase 3's single timeout.
2. **Separate time budgets** — NREM gets `PHASE_TIMEOUT` (15s), REM gets
   `DREAM_REM_TIMEOUT_SECONDS` (60s default). Each can run to its own
   ceiling without holding the other hostage.

Config:
- `ENABLE_TWO_PHASE_DREAM` (default `false`) — master switch
- `DREAM_REM_TIMEOUT_SECONDS` (default 60) — ceiling for the REM phase

When off, behavior is unchanged (legacy single-phase consolidate()).

`ConsolidationResult` carries `nrem_seconds`, `rem_seconds`, `nrem_completed`,
`rem_completed` for instrumentation when the flag is on. These stay 0/False
in the legacy path.

## Hybrid Retrieval Config

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_RERANKER` | `true` | Apply composite score reranker after RRF fusion |
| `RETRIEVAL_RRF_K` | `60` | RRF smoothing constant (alias `RRF_K`) |

Reranker: composite heuristic `0.55·vec + 0.30·bm25 + 0.15·coverage`. No external model required.
A cross-encoder path (sentence-transformers) was evaluated empirically on a 300-doc adversarial corpus
and gave 0pp gain over composite on Recall@5/P@1/MRR — deleted (see commit for 4×4 table).

## New Config Fields (Deep Audit)
- `MAX_QUERY_LENGTH` (50000) — query length validation in brain.think()
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

## Automated Eval Harness (`app/monitors/eval_harness.py`)

Self-testing pipeline that runs a curated task suite through the real brain,
computes quality metrics, and flags regressions.  Runs as a nightly heartbeat
monitor (`check_type="eval"`, monitor name "Quality Eval Harness").

### Files

| File | Purpose |
|------|---------|
| `evals/suite.yaml` | 40 evaluation tasks across 8 categories (reasoning, tool-use, skill-match, semantic-match, autonomous-tool, reflexion-calibration, multi-agent, retrieval) |
| `app/monitors/eval_harness.py` | Harness engine — task runner, metrics, regression detection |
| `/data/eval_reports/eval_<ts>.json` | Full structured report (per run) |
| `/data/eval_reports/eval_<ts>.md` | Human-readable markdown summary (per run) |
| `/data/eval_reports/eval_history.jsonl` | Time-series log — one line per run, appended |
| `/data/eval_reports/eval_baseline.json` | Regression baseline (written on first run) |

### Categories

| Category | Tasks | What it tests |
|----------|-------|---------------|
| `reasoning` | 7 | Arithmetic, logic, definitions — no tools, high reflexion expected |
| `tool-use` | 6 | calculator, code_exec, web_search invocation + answer correctness |
| `skill-match` | 6 | Seeded "Eval: *" skills matched by exact regex (eval-probe: prefix) |
| `semantic-match` | 5 | Paraphrase queries that must hit same skill via ChromaDB at threshold 0.65 |
| `autonomous-tool` | 4 | Multi-step queries; metric = fraction using ≥2 tools |
| `reflexion-calibration` | 4 | Score distribution validation — detects inflation/deflation |
| `multi-agent` | 5 | Decomposition gating + sub-agent merge correctness (incl. no-decompose baseline) |
| `retrieval` | 3 | BM25 exact-keyword + semantic paraphrase + reranker regression probe |

### Metrics

- **Per-category**: pass_rate, latency P50/P95, reflexion mean/std/P10/P90
- **skill-match**: hit_rate (fraction of queries that matched any skill)
- **semantic-match**: recall_at_threshold (paraphrases matching at 0.65)
- **autonomous-tool**: multi_tool_rate (fraction using ≥2 tools)
- **Regression flags**: any metric dropping >EVAL_REGRESSION_TOLERANCE (10%) from baseline

### How to add a task

Add an entry to `evals/suite.yaml`:

```yaml
- id: my_task_001          # unique snake_case id
  category: reasoning       # one of the 8 categories above
  query: "What is 2+2?"
  timeout: 45               # seconds (default 60)
  assertions:
    - type: answer_contains
      value: "4"
    - type: reflexion_above
      value: 0.5
```

For a skill-match task with a seeded skill:

```yaml
- id: skill_match_myskill
  category: skill-match
  query: "eval-probe: do my thing"
  seed_skill:
    name: "Eval: My Skill"
    trigger_pattern: "(?i)\\beval-probe[:\\s]+.*do\\s+my\\s+thing\\b"
    steps:
      - tool: web_search
        args_template: {q: "{query}"}
        output_key: result
  assertions:
    - type: skill_matched
```

### How to interpret a drift flag

A `RegressionFlag` in the report JSON means a metric dropped more than
`EVAL_REGRESSION_TOLERANCE` (default 0.10 = 10 percentage points) below the
stored baseline.  Common causes:

- **skill-match.hit_rate drops** — skill patterns broken or SkillStore corrupted
- **semantic-match.recall_at_threshold drops** — `SKILL_SEMANTIC_THRESHOLD` too high
  (was the regression we proved empirically: raising to 0.99 drops recall to 0%)
- **tool-use.pass_rate drops** — tool registry broken or tool unreachable
- **reflexion_mean drifts upward** — score inflation (quality heuristic too lenient)

To update the baseline after intentional improvements:

```python
from app.monitors.eval_harness import EvalHarness
harness = EvalHarness()
# Load any recent report JSON as the new baseline
import json
with open("/data/eval_reports/eval_<ts>.json") as f:
    data = json.load(f)
# Then write it as baseline directly
import shutil
shutil.copy("/data/eval_reports/eval_<ts>.json",
            "/data/eval_reports/eval_baseline.json")
```

### Config flags

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_EVAL_HARNESS` | `true` | Enable/disable the harness monitor |
| `EVAL_SUITE_PATH` | `evals/suite.yaml` | Path to task suite YAML |
| `EVAL_REPORT_PATH` | `/data/eval_reports` | Output directory for reports |
| `EVAL_REGRESSION_TOLERANCE` | `0.10` | Allowed metric drop before flagging |
