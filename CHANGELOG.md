# Changelog

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
- **51 autonomous monitors** — expanded from 14 to 51 across 29 domains: financial intelligence (whale watch, top trades, commodities, DeFi), international perspectives (China, Russia, Middle East, India, EU, Latin America, Africa), science/tech deep dives (AI/ML, semiconductors, quantum, robotics, biotech), policy/security (cybersecurity, defense, US regulation), culture (sports, entertainment, social media), developer ecosystem (GitHub trending, framework releases), and local news (Los Angeles)
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
