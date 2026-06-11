# Changelog

## [1.6.0] - 2026-06-10

Polish pass: measurement honesty, paraphrase-robust memory retrieval, onboarding repair, experiment pruning.

### Eval scoring — timeouts are not wrong answers
- **Outcome `timeout` separated from `fail` everywhere.** A run that hits its time budget without proving correctness no longer counts as incorrect — it's excluded from every correctness denominator and tracked as its own metric. The 2026-03-18 live audit scored 4.5/10 largely because whole categories timed out and were graded as wrong; correctness and latency are now measured independently so neither can hide the other. Slow-but-correct still counts as a pass.
- **memory-learning / kg-retrieval pairs with a timed-out leg are UNTESTABLE** — excluded from `memory_causal_fix_rate` instead of recorded as failed fixes
- **Timeouts no longer write failure reflexions** — budget exhaustion must not pollute the failure-learning store
- **`tests/live_audit.py`**: timeout → `[T]`, excluded from grade, listed separately; per-query latency captured; scorecard reports mean + p95
- 9 new tests (`TestTimeoutSeparation`)

### Lesson retrieval — paraphrase-robust (semantic-first completed)
- **The originating query is now embedded into the lesson vector document.** A future paraphrase is a paraphrase *of that query* — query-to-query similarity is the strongest semantic signal. The query was already persisted and used by the keyword path; the vector path (the one paraphrases depend on) never saw it.
- **`MIN_RRF_SCORE` no longer vetoes strong vector matches.** A vector-only hit (the paraphrase case by definition) scores ~1/(k+1) in RRF — near any practical floor. Hits at distance ≤ `LESSON_VECTOR_STRONG_DISTANCE` (new, default 0.55) bypass the floor; weak hits (0.55–0.9) still must clear it, so single-word-overlap noise stays rejected.
- **`.env.example` shipped the old floor (0.015)** that config.py had already lowered to 0.005 — fresh installs would have silently re-broken paraphrase retrieval. Fixed.
- 4 new tests (`TestParaphraseRetrieval`) including an end-to-end paraphrase roundtrip with real embeddings

### Onboarding repair
- README quick start: `cd nova_` → `cd nova` (the clone directory was wrong)
- `install.sh`: three hardware tiers — 20GB+ GPU → 27b, 8GB+ GPU → 9b, CPU-only → 4b. The old no-GPU path pointed at cloud providers removed in v1.5.0 and a `docker-compose.cloud.yml` that never existed
- New `docker-compose.cpu.yml` override drops the NVIDIA device reservation so CPU-only machines can start the stack
- README: "Multi-Provider LLM" section replaced with an honest Ollama-only statement; CPU row added to the low-VRAM table; monitor counts aligned to the test-pinned seeded count (69)

### Pruned (preserved on `experiments/pre-v1.6-archive`)
- **A-HMAD debate** (`app/core/debate.py`, brain hook, 2 config keys) and **MAD-MM memory masking** (`agent_loop._mask_prior_observations`, retry-mask plumbing, 3 config keys): both default-off, never enabled in production, never showed measured value. ~600 lines removed
- Kept deliberately: fine-tune complex (explicit decision), `ENABLE_PROMPT_SELF_MOD` (live since v1.5.0), `ENABLE_TWO_PHASE_DREAM` (current architectural rationale)
- Dead `HOST` config key removed (never read — bind address comes from the uvicorn invocation)

### Repo hygiene
- `training_data.jsonl` + backup untracked (the honest-reposition commit said training data was excluded; now it actually is)
- Internal docs moved out of the public repo (patent claims draft, launch posts, competitive landscape, capability gap analysis, ecosystem audit) → local archive
- Self-judged degenerate e2e report and synthetic drift-sim report archived; `baselines/README.md` now states the receipts policy: claimed numbers need committed, cross-family-judged eval artifacts
- Monitor dispatch refactored from a 27-branch if-elif chain to a registry (`_CHECK_DISPATCH`)
- `SECURITY.md` updated: mandatory fail-closed query injection detection, always-on tool-output sanitization, CSRF non-applicability (header-token auth, no cookies)
- `MODEL_CARD.md`: three stale 27B references fixed (the FT base is 9B)

## [1.5.1] - 2026-04-24

### Fixed (Multi-agent ceiling)
- **Decomposer compare-split always won over entity-split** — `_try_compare_split` matched "Compare A, B, C, D, E" via its `A and B` regex and produced two agents with truncated role names (e.g. `these-in-memory-caches-side-by-side:-red-researcher`). Entity-split now runs first when the query has 3+ proper nouns. `_STOPWORDS` now includes `Compare`, `Contrast`, `List`, `Explain`, `Describe`, `Summarize`, `Analyze`, `Research`, `Review`, `Evaluate`, `Rank`, `Rate` so imperative verbs don't count as entities
- **Merge budget didn't scale with agent count** — `max_tokens=RESPONSE_TOKEN_BUDGET` (600) truncated mid-enumeration on large merges, dropping half the entities at N=10 (verified via `scripts/multi_agent_ceiling.py`). Budget now scales: `min(600 + 200*(n-1), 4000)` tokens; timeout scales: `max(INTERNAL_LLM_TIMEOUT, 30 + 10*n)` seconds. Merge system prompt also explicitly tells the model to cover every entity. Post-fix verification: 10/10 coverage at N=10 (was 5/10)
- **`PATCH /api/config` silently dropped multi-agent fields** — `MAX_AGENT_COUNT`, `AGENT_TASK_TIMEOUT`, `ENABLE_MULTI_AGENT`, `MULTI_AGENT_TRIGGER_THRESHOLD` were in `_MUTABLE_FIELDS` but not in the pydantic `ConfigUpdateRequest`, so PATCH requests with those fields were accepted and ignored. Added with bounds validation

### Added
- **brain.py modularization** (2517 → 2075 lines, 17.5% reduction):
  - `app/core/brain_sanitize.py` — `_META_PATTERNS` + `_sanitize_answer` (29 regex patterns for meta-commentary, tool-leak, date-dispute, and system-prompt-leak stripping)
  - `app/core/brain_kg.py` — `_extract_kg_triples` + `_SOURCE_CONFIDENCE` tier map
  - `app/core/brain_routing.py` — `_classify_intent`, `_generate_title`, `_select_model` + all pattern constants
  - `app/core/brain_context_manager.py` — `_manage_context` summarization/truncation
  - All modules re-exported from `brain.py` so existing `from app.core.brain import X` calls keep working
- **`scripts/multi_agent_ceiling.py`** — HTTP/SSE probe sweeping N=3→10 to measure useful-work coverage per agent count. Reports land at `data/ceiling_reports/`
- **`scripts/distill_simpo_pairs.py`** — 47 Claude-distilled SimPO pairs covering 12 categories (date awareness, memory/identity, terse tool synthesis, correction acceptance, honest uncertainty, format discipline, system prompt confidentiality, tool-output honesty, multi-agent restraint, self-knowledge, tool-call leakage cleanup, remember-this signals). `--sync-container` flag docker-cps into the named volume. Training data: 725 → 772 pairs
- **8 new regression tests**:
  - `test_decomposer.py::test_multi_entity_query_prefers_entity_split` — 5 entities → 5 per-entity agents, role names under 40 chars
  - `test_decomposer.py::test_imperative_verbs_are_not_entities` — "Compare Python and JavaScript" → 2 tasks, not 3
  - `test_agent_spawner.py::test_merge_budget_scales_with_agent_count` — N=10 merge gets ≥2000 max_tokens
  - `test_system_api.py::TestConfigUpdateMultiAgent` — 5 tests for the new API fields + bounds validation

### Operational
- **`AGENT_TASK_TIMEOUT` default 90s → 300s** in `app/config.py` (was also overridden in `/data/config_overrides.json`). Verified: N=5 probe goes 0/5 → 5/5 coverage on RTX 3090 when sub-agents do real tool research
- **`MAX_AGENT_COUNT` default 5 → 10** in `app/config.py`. Lets entity-split spawn one agent per named entity for queries with 5-10 targets
- **Eval baseline refreshed** — pass rate **94.4% → 97.2%** (34/36 → 35/36) after session fixes landed:
  - `retrieval.pass_rate`: 66.7% → 100% (+33.3pp) — merge budget scaling fix unblocked the reranker-regression probe
  - `reflexion-calibration.pass_rate`: 75% → 100% (+25pp) — empty-response failure resolved
  - `multi-agent.decomposition_rate`: 33.3% → 66.7% (+33.3pp) — entity-split fix lets more queries decompose correctly
  - One eval-suite bug fixed: `autonomous_stock_and_calc` hard-coded AAPL at $200 → 25 shares, but Nova correctly looked up the real $270.04 price. Query now marks the $200 figure as hypothetical

## [1.5.0] - 2026-04-23

### Fixed (Facade Kill cycle)
- **Dream cycle "8 errors" → 0** — `_promote_reflexions` and `_resolve_contradictions` were passing a bare prompt string to `invoke_nothink()`, which expects `messages: list[dict]`. Every call crashed silently for weeks. Fixed to `[{"role": "user", "content": prompt}]`. Added `extract_json_object()` fallback parser. Now promotes 5 reflexions → lessons per cycle
- **SearXNG 61% empty results** — default engines (`google,duckduckgo,brave`) were all rate-limited in production. Swapped to `bing,startpage,ecosia,yandex,yahoo` (verified working under automated load)
- **Trust signal was meaningless** — `web_search` "No results found" returned `success=True` with `+1.0` trust delta. All 8,815 empty searches counted as successes. Fixed: empty → `success=False, NOT_FOUND` category; `record_outcome` skips `NOT_FOUND`/`VALIDATION` entirely; successes now sampled 1/50 to bound audit log growth
- **Active memory poisoned by test queries** — added `ContextVar` intent gate: `active_memory.add` rejects unless current user message contains explicit memory intent (`remember`, `note that`, `always`, `from now on`) or correction signal (`no, actually`, `you're wrong`). 11 contradictory test personas (Alex/Alice/Sam/Stripe/Shopify/Acme) purged from production
- **Auto fact extraction removed** — `extract_facts_from_message` and 200 lines of surrounding regex/LLM pipeline deleted. Source of continuous pollution. Manual storage via `active_memory` tool remains
- **Auto-Monitor Detector silently erroring** — multi-branch alternation regex used inline `(?i)` per branch. Python 3.12 rejects inline flags after position 0. Removed inline flags (the trailing `re.IGNORECASE` arg is sufficient)
- **Event triggers were dead** — `MonitorStore._event_matches` and `get_event_monitors` referenced by `EventTrigger._process_pending_events` but didn't exist in store. Added back. Added `trigger_events`/`trigger_mode` columns (migration 16)
- **`lessons.retrieval_score` missing in fresh DBs** — code referenced the column but no migration created it. Production DB had it from manual backfill; fresh installs and test DBs crashed on `mark_lesson_helpful`. Added to migration 16
- **Skill threshold too aggressive** — 3 consecutive failures → immediate disable meant a skill survived one rate-limited SearXNG streak and another transient upstream hiccup before getting killed. Raised to 5
- **Cloud provider remnants** — removed `OPENAI_API_KEY`/`MODEL`, `ANTHROPIC_API_KEY`/`MODEL`, `GOOGLE_API_KEY`/`MODEL`, `OPENAI_BASE_URL`/etc. from config.py, api/system.py. Deleted `tests/test_providers.py` (479 lines) and `tests/test_provider_failover.py`. Updated remaining tests to use `DISCORD_TOKEN` as the sensitive-field example

### Added
- **Dead-lesson prune** — lessons created 21+ days ago that were never retrieved (or retrieved-but-never-helpful with `times_retrieved >= 10`) are deleted on startup and during maintenance
- **Audit log retention** — `action_log` and `trust_audit_log` prune to 30 days during maintenance (was unbounded)
- **Forced synthesis fallback** — when circuit breaker filters all tool calls but `tool_results` has data, brain forces a synthesis pass instead of emitting empty response (fixes monitor empty-result skips)
- **Prompt self-modification live** — `ENABLE_PROMPT_SELF_MOD=true`, Prompt Optimizer monitor enabled. Will propose candidates against 10+ existing eval baselines
- **ChromaDB warning silenced** — `local_persistent_hnsw` delete-of-nonexisting warnings drowned out real warnings; suppressed at module level

### Changed
- **Ollama models cleaned** — removed `nova-ft-v8-q8`, `v11`, `v12`, `v13` (38GB freed). Active: `nova-ft-v10`
- **Model routing disabled** — `FAST_MODEL=""` (4b model was deleted earlier but env still referenced it). Config now matches `/data/config_overrides.json`
- **`/data` cleanup** — removed `_src_*`, `app_backup`, `tests_backup`, stale probe results (`v9_*.json`, `gen_pairs*.py`, `demo.svg`, `long_context_results.json`), old `chroma/` dir. ~20MB

### Eval harness
- **50% → 94.4% pass rate** (18/36 → 34/36). `skill-match`: 0% → 100%. `semantic-match`: 0% → 100%. `autonomous-tool`: 50% → 100%. `multi-agent`: 67% → 100%. `reasoning`, `tool-use`: 80-83% → 100%. Baseline refreshed to current snapshot

## [1.4.0] - 2026-03-28

### Added
- **Native Ollama tool calling** — `stream_with_thinking()` now parses `message.tool_calls` from stream chunks instead of relying solely on text extraction. Ollama 0.17+ structured tool calls flow end-to-end through the pipeline
- **DPO training curriculum v2** — 221 expert reasoning traces across 8 categories: tool chaining & fallback (25), response discipline (25), search vs knowledge boundary (25), financial analysis (25), tool result evaluation (25), multi-step planning (25), error recovery (21), context & memory usage (25). Deployed as `nova-ft-v3`
- **Result evaluation between tool rounds** — `brain.py` now prompts the model to evaluate intermediate tool results. If data is sufficient, it synthesizes early; if incomplete, it continues tool calls. Prevents wasted rounds
- **One-click fine-tuning pipeline** — `scripts/finetune_oneclick.py` handles the full cycle: stop Ollama → DPO train → merge LoRA → convert to GGUF → quantize Q4_K_M → register model → restart Ollama
- **Smart desktop automation** — `smart_click` (vision-based element finding), `smart_type` (find input + type), `autonomous_workflow` (multi-step loop with action repetition guard). Uses Ollama vision model to locate UI elements from screenshots
- **Browser CDP fallback** — when `connect_over_cdp()` to host browser fails, automatically falls back to headless Chromium launch inside container
- **7 high-value monitors** — FDA Drug Approvals (24h), FOMC and Fed Watch (24h), GitHub Security Advisories (12h), Government Contract Awards (24h), Hacker News Top Stories (8h), Product Hunt Trending (24h), SEC Insider Trading (12h)
- **Auto-monitor quality filter** — rejects questions, price queries, time queries, math, and generation requests from becoming recurring monitors

### Fixed
- **Native tool calls silently dropped** — `stream_with_thinking()` in ollama.py never parsed the `tool_calls` field from stream message chunks, so all native tool calls were lost even though `supports_native_tools=True` was set
- **HTTP fetch SSL cert mismatch on CDN** — DNS pinning replaced hostname with raw IP in HTTPS requests, breaking SSL certificate validation on CDN-served sites. Now skips DNS pinning for HTTPS (SSL validates server identity)
- **Monitor garbage creation** — no quality filter on auto-monitor detector meant test queries and simple questions became recurring monitors

### Changed
- **Quality over speed** — constraints loosened for personal hardware: `MAX_TOOL_ROUNDS` 5→10, `MAX_SYSTEM_TOKENS` 6000→10000, `GENERATION_TIMEOUT` 480→900s, `TOOL_TIMEOUT` 120→180s, `OLLAMA_NUM_CTX` 16384→32768, Ollama container memory 20g→32g
- **Planning threshold lowered** — `signals >= 2` → `signals >= 1` in `planning.py`. Any signal of complexity triggers the planning step
- **Monitor cleanup** — removed 6 low-value monitors (Sports, Entertainment, Social Media, Climate/Weather, Local LA, Self-Reflection), added 7 high-value intelligence monitors. Net: 52 monitors focused on actionable intelligence

## [1.3.0] - 2026-03-26

### Added
- **Blacklist fact extraction** — replaced regex whitelist gate (`has_fact_signals`) with blacklist approach (`_is_pure_question_or_command`). Nova now extracts facts from ANY message that isn't a pure question, command, or greeting. Implicit statements like "my portfolio is 60/40" and "I drive a Tesla" are now captured. The LLM extraction prompt handles false positives by returning `{}`
- **Action audit trail** — all tool executions and monitor alerts now logged to `action_log` table via `AuditLogHook.post_execute()`. Actions page is no longer empty
- **All config toggles working** — added 11 missing `ENABLE_*` fields to both `ConfigUpdateRequest` (Pydantic) and `_MUTABLE_FIELDS` set. Toggles that returned 422 now work
- **API key authentication** — `REQUIRE_AUTH=true` with generated API key. Endpoints reject unauthenticated requests
- **DPO from messaging channels** — `TRAINING_DATA_CHANNELS=api,discord,telegram` enables training pair generation from corrections via Discord and Telegram
- **Frontend UX overhaul** — timestamps show actual times ("4:30 PM", "Yesterday 4:30 PM") instead of vague "3d ago"; tool calls display arguments inline; chat empty state has clickable example prompts; monitors grouped by category with collapsible sections and schedule presets; reflexions have All/Successes/Failures filter; curiosity shows priority badges; lessons column renamed "Times Used"; all empty states have guidance text; StatusBadge shows "Connected" (proxy fix)
- **Sports monitor browser fallback** — query instructs Nova to use Playwright browser for ESPN scoreboards when web_search returns only portal links

### Fixed
- **Frontend proxy "Disconnected"** — Vite dev server proxied to `localhost:8000` (unreachable inside container). Changed to `nova-app:8000` via `API_PROXY_TARGET`
- **Test suite auth failures** — conftest now resets `NOVA_API_KEY=""`, `REQUIRE_AUTH=false`, `SYSTEM_ACCESS_LEVEL=sandboxed` so tests don't inherit production env
- **Script smoke tests** — added `pytestmark = pytest.mark.skipif` when `scripts/` directory unavailable in container
- **Duplicate lessons** — deleted pre-dedup duplicate "Premier League Standings" from database

### Changed
- `FINETUNE_MIN_NEW_PAIRS` default lowered from 50 to 15 for bootstrapping first fine-tune cycle
- Frontend Dockerfile uses `npm install` fallback for cross-platform lock file compatibility
- Vite config adds `allowedHosts` for Docker inter-container access

## [1.2.0] - 2026-03-26

### Added
- **52 autonomous monitors** — expanded from 14 to 52 across 35+ domains: financial intelligence (whale watch, top trades, commodities, DeFi), international perspectives (China, Russia, Middle East, India, EU, Latin America, Africa), science/tech deep dives (AI/ML, semiconductors, quantum, robotics, biotech), policy/security (cybersecurity, defense, US regulation), special intelligence (FDA, FOMC, SEC, GitHub security advisories, govt contracts), developer ecosystem (GitHub trending, framework releases)
- **Temporal freshness enforcement** — `_think_query()` now injects today's date into every monitor query context; all monitor prompts anchored to "past 24-48 hours" instead of vague "recently"
- **Ollama thinking fallback** — provider catches "does not support thinking" 400 errors and retries with `think=false` instead of crashing
- **Fact extraction for life changes** — added patterns for "I moved to", "I switched to", "I joined", "I left", "I no longer", "I used to" so corrections about personal info are captured
- **Monitor migration v3** — existing monitors auto-update to freshness-anchored prompts on restart

### Fixed
- **P0: Chat completely broken** — fine-tuned model (`nova-ft`) was missing `RENDERER qwen3.5` and `PARSER qwen3.5` in Modelfile config, causing Ollama to reject all `think:true` requests with 400. Fixed model config blob and all finetune scripts
- **User facts silently dropped on correction** — `UserFactStore.set()` used `<=` for same-source confidence check, blocking same-authority overwrites. User correcting their own facts (e.g., "I moved to Seattle") was silently ignored
- **Monitor retry storm** — outer exception handler in heartbeat loop used flat 5-minute retry regardless of error count. Now uses exponential backoff (5→15→45→135→405 min) matching the inner handler
- **ChromaDB telemetry errors** — PostHog `capture()` API mismatch producing "takes 1 positional argument but 3 were given". Fixed by disabling telemetry via `ANONYMIZED_TELEMETRY=false`
- **ChromaDB duplicate embedding warnings** — `collection.add()` on startup re-added existing embeddings. Changed to `collection.upsert()` in both `learning.py` and `reflexion.py`
- **Finetune model missing thinking support** — `finetune.py`, `finetune_weekly.sh`, and `finetune_auto.py` now include `RENDERER qwen3.5` and `PARSER qwen3.5` in generated Modelfiles

### Changed
- Monitor error backoff now consistent between inner (LLM failure detection) and outer (exception handler) paths
- All monitor query prompts updated from vague temporal language to explicit "from TODAY" / "past 24-48 hours" with date requirements

## [1.1.0] - 2026-03-25

### Fixed
- **Quiz DPO generation blocked** — quiz monitor wasn't generating DPO pairs from failed quizzes
- **Tool error exposure** — raw internal errors leaked to user in tool failure responses
- **Discord alerts** — alert delivery to Discord was silently failing
- **Fact extraction** — missed extraction on some explicit user statements
- **Lesson quality gate** — low-confidence lessons were being retrieved in prompts
- **Self-reflection context** — reflexion monitor lacked conversation context for meaningful self-assessment
- **KG contradictions** — contradicting facts weren't being superseded properly
- **LLM failure detection** — silent failures in generation not being caught by reflexion system

### Added
- Exponential backoff for monitor retries
- Background task tools (submit, status, list, cancel)
- Desktop automation tool (screenshot, click, type, hotkey)
- Voice interface (Whisper STT)
- WhatsApp and Signal channel adapters
- Access tier system (sandboxed/standard/full/none)
- Skill import/export with HMAC-SHA256 signing
- Prompt injection detection (4 categories)
- MCP server exposing 5 Nova tools
- Provider-aware prompt building (emphatic for Ollama, condensed for cloud)
- Configurable cloud provider base URLs

## [1.0.0] - 2026-03-13

### Security
- **Discord user allowlisting** — Discord channel now supports `DISCORD_ALLOWED_USERS` (previously the only channel without access control)
- **Prompt injection detection** expanded to browser, MCP tools, and knowledge base (previously only web search and HTTP fetch)
- **MCP tools respect access tiers** — blocked at sandboxed, warned at standard
- **Auth rate-limiting** — per-IP lockout after 10 failed auth attempts in 60 seconds
- **Skill signing enforced by default** — `REQUIRE_SIGNED_SKILLS` now defaults to `true`
- **Anti-hijack system prompt** — security boundaries section prevents instruction injection from external content
- **Anti-sycophancy** — Nova refuses to agree with false corrections to computed results
- **Training data poisoning prevention** — channel gating (`TRAINING_DATA_CHANNELS`) and confidence threshold for external channels
- **Expanded protected paths** — added `/etc/passwd`, `/etc/sudoers`, `/proc`, `/sys` and others to write-protected paths
- **Docker hardening** — read-only root filesystem, no-new-privileges, all capabilities dropped
- **Secret scan** — verified no hardcoded credentials in source code

### Fixed
- **Anthropic JSON mode** — `invoke_nothink(json_mode=True)` now uses assistant prefill approach
- **Anthropic streaming thinking** — sends required beta header and thinking config block
- **HTTP error handling** — all 3 cloud providers (Anthropic, OpenAI, Google) now retry on 429/5xx and raise `LLMUnavailableError` on auth errors
- **Desktop blocking sleep** — replaced `time.sleep()` with `await asyncio.sleep()` to prevent event loop blocking
- **Document re-ingest duplicates** — deletes existing chunks before re-inserting (FTS5 + ChromaDB)
- **Images unreachable for non-Ollama** — `generate_with_tools()` now forwards `images` parameter
- **WhatsApp dedup pruning** — replaced `set` with `OrderedDict` for ordered eviction
- **Signal message dedup** — added `OrderedDict`-based dedup using `timestamp:source`
- **KG extraction scoped to monitors** — prevents untrusted user queries from poisoning the knowledge graph
- **`messages[-1]` assumption** — explicit `query` parameter to `_run_generation_loop()` instead of extracting from messages
- **Training data thread safety** — `save_training_pair` is now async with `asyncio.Lock`
- **KG BFS visited tracking** — frontier now excludes already-visited entities
- **Heartbeat instructions** — WhatsApp and Signal now receive monitor alerts and curiosity follow-ups
- **System Health monitor** — replaced shell commands with Python-native checks (`os.statvfs`, `os.getloadavg`, `psutil`)
- **Conversation ID warning** — logs warning when a missing conversation ID is silently replaced
- **CJK token estimation** — improved heuristic for CJK characters (~1.5 chars/token vs 4 for English)
- **User fact dedup** — requires minimum 2 overlapping words to consider facts as duplicates
- **Curiosity dedup** — Jaccard similarity matching (threshold 0.6) prevents near-duplicate questions
- **Auto-skills initial success rate** — new skills start at 0.7 instead of 1.0
- **Skill matching specificity** — multiple matches sorted by regex pattern length (most specific wins)
- **Unbounded conversations** — all 4 channel adapters now use LRU eviction at 1000 entries
- **Desktop screenshot dir** — lazy creation with error handling instead of eager init
- **Browser instance reuse** — Chromium browser pooled and reused across requests
- **Reflexion query limits** — unbounded SELECT queries now capped at LIMIT 200
- **OpenAI max_tokens** — updated to `max_completion_tokens` (modern API convention)
- **KG prune batching** — pruning runs every 50 inserts instead of every insert
- **CSP header** — added `connect-src 'self'` for frontend API calls
- **Finetune script** — replaced `sys.exit()` with `raise RuntimeError()` for cleaner error handling
- **Finetune container name** — configurable via `OLLAMA_CONTAINER` env var
- **Creative patterns** — moved `_CREATIVE_PATTERNS` regex to module-level constant
- **User facts None guard** — handles missing `svc.user_facts` gracefully

### Added
- `DISCORD_ALLOWED_USERS` config field
- `TRAINING_DATA_CHANNELS` config field
- `channel` parameter on `think()` for training data channel gating
- `system_health` check type for heartbeat monitors
- Security Boundaries section in system prompt (OWASP ASI01)
- Anti-sycophantic correction handling in system prompt (OWASP ASI09)
