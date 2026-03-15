# Changelog

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
