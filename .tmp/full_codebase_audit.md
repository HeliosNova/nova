# Nova Full Codebase Audit

**Date:** 2026-04-15  
**Branch:** claude/cranky-newton (1770 passed / 2 skipped / 0 failed)  
**Scope:** Complete codebase — all original modules + 45 new commits  
**Codebase:** ~26,000 lines Python, 80+ modules, React frontend  

---

## Executive Summary — Top 10 by Severity

| # | Severity | Area | Finding | Location |
|---|----------|------|---------|----------|
| 1 | **CRITICAL** | Security | Prompt injection detection is optional — user input reaches LLM without mandatory sanitization | `app/core/brain.py`, `app/core/injection.py` |
| 2 | **CRITICAL** | Security | Custom tool code execution (`DynamicTool`) lacks sufficient sandboxing — user-defined tools can access filesystem/network | `app/core/custom_tools.py` |
| 3 | **CRITICAL** | Security | Shell/code execution tools have incomplete resource cleanup and no per-request isolation | `app/tools/code_exec.py`, `app/tools/shell_exec.py` |
| 4 | **CRITICAL** | Error Handling | Chat streaming, database migrations, and tool execution have unguarded async operations | `app/api/chat.py`, `app/database.py` |
| 5 | **CRITICAL** | Test Coverage | WhatsApp HMAC signature verification is untested — webhooks could be spoofed | `app/channels/whatsapp.py:79-91` |
| 6 | **HIGH** | Arch Debt | 102 blocking calls in async functions (30+ DB ops, 7+ file I/O) — 10-100x throughput loss under concurrency | `app/api/daemon.py`, `app/api/learning.py`, `app/main.py` |
| 7 | **HIGH** | Security | No per-endpoint rate limiting on authenticated endpoints — DoS and resource exhaustion possible | `app/main.py`, `app/api/*` |
| 8 | **HIGH** | Dead Code | 3 security middleware classes (RateLimit, SecurityHeaders, UserActivity) defined but never registered — features silently disabled | `app/main.py` |
| 9 | **HIGH** | Test Coverage | `api/chat.py` has 11 endpoints with ZERO tests; `api/monitors.py` (540 lines, 19 functions) also untested | `tests/` |
| 10 | **HIGH** | Config | 17 unused config keys + provider configs for Anthropic/OpenAI/Google that reference nonexistent providers | `app/config.py` |

---

## 1. Dead Code

### 1.1 Unused Config Keys (17 total)

| Key | File | Notes |
|-----|------|-------|
| `HOST` | config.py | Defined but server always binds to `0.0.0.0` |
| `ANTHROPIC_API_KEY` | config.py | No Anthropic provider implementation exists |
| `ANTHROPIC_API_VERSION` | config.py | No Anthropic provider implementation exists |
| `ANTHROPIC_BASE_URL` | config.py | No Anthropic provider implementation exists |
| `OPENAI_API_KEY` | config.py | No OpenAI provider implementation exists |
| `OPENAI_BASE_URL` | config.py | No OpenAI provider implementation exists |
| `GOOGLE_API_KEY` | config.py | No Google provider implementation exists |
| `GOOGLE_BASE_URL` | config.py | No Google provider implementation exists |
| `TEMPERATURE_*` (multiple) | config.py | Temperature parameters defined but not wired to provider calls |
| Provider-specific model configs | config.py | Corresponding providers not implemented |

**Severity:** HIGH — Users setting `LLM_PROVIDER=anthropic` get silent failure, no error.

### 1.2 Unused Exported Functions (11)

| Function | Module | Severity |
|----------|--------|----------|
| `detect_image_mime()` | core/text_utils.py | LOW |
| `export_skill()` | core/skill_export.py | MEDIUM |
| `sign_skill()` | core/skill_export.py | MEDIUM |
| `import_skills_from_file()` | core/skill_export.py | MEDIUM |
| `export_lesson()` | core/data_export.py | MEDIUM |
| `export_kg_fact()` | core/data_export.py | MEDIUM |
| `should_critique()` | core/critique.py | MEDIUM |
| `is_decomposable()` | core/planning.py | MEDIUM |
| `get_allowed_read_roots()` | core/access_tiers.py | HIGH — security function never called |
| `get_allowed_write_roots()` | core/access_tiers.py | HIGH — security function never called |
| `get_async_db()` | database.py | LOW |

### 1.3 Unused Data Classes (4)

`CuriosityItem`, `CustomToolRecord`, `InjectionResult`, `TranscriptionResult` — defined but never instantiated.

### 1.4 Unregistered Middleware (3)

| Class | Purpose | Impact |
|-------|---------|--------|
| `RateLimitMiddleware` | Per-IP rate limiting | Rate limiting disabled |
| `SecurityHeadersMiddleware` | CSP, X-Frame-Options | Security headers missing |
| `UserActivityMiddleware` | Track user activity | Activity tracking disabled |

**Severity:** HIGH — These are implemented security features that are silently inactive.

---

## 2. Broken Integrations

| Finding | Severity | Details |
|---------|----------|---------|
| Provider config/implementation mismatch | HIGH | Config defines Anthropic, OpenAI, Google provider settings; only Ollama provider is implemented. Silent failure on misconfiguration. |
| Unregistered middleware | HIGH | 3 middleware classes exist but aren't added to FastAPI app |
| Access tier functions unused | HIGH | `get_allowed_read_roots()` / `get_allowed_write_roots()` defined but file_ops.py doesn't call them |

No critical runtime broken imports or missing database tables found — the core import chain is intact.

---

## 3. Missing Error Handling

### Critical

| Finding | Location | Impact |
|---------|----------|--------|
| Chat streaming has unguarded async operations | `app/api/chat.py` | Unhandled exceptions crash SSE streams |
| Database migration errors not caught | `app/database.py` | Failed migration leaves DB in inconsistent state |
| Tool execution missing broad exception handler | `app/core/brain.py` | Tool failure can crash entire conversation |
| Auth lockout state can desync from DB | `app/auth.py` | Module-level dict vs DB persistence race condition |

### High

| Finding | Location | Impact |
|---------|----------|--------|
| HTTP fetch follows redirects without limit validation | `app/tools/http_fetch.py` | SSRF via redirect chains |
| File I/O in learning API lacks error handling | `app/api/learning.py` | Unhandled file errors crash endpoint |
| Shell/code exec resource cleanup incomplete | `app/tools/code_exec.py`, `shell_exec.py` | Leaked subprocesses |
| Database connections not closed on shutdown | `app/database.py` | Connection pool exhaustion on restart |

### Medium

| Finding | Location | Impact |
|---------|----------|--------|
| FTS5 search queries without complexity limits | `app/database.py` | DoS via crafted search queries |
| Input validation incomplete on several endpoints | `app/api/*` | Unexpected data can cause 500 errors |
| Error messages may expose config details | Various | Information disclosure |

---

## 4. Security Boundaries

### Critical

| Finding | Location | Details |
|---------|----------|---------|
| **Prompt injection detection is optional** | `core/injection.py`, `core/brain.py` | The injection scanner exists but is not mandatory — user input flows directly to LLM prompts without guaranteed sanitization |
| **DynamicTool sandboxing gaps** | `core/custom_tools.py` | User-defined tools execute with the same permissions as Nova itself; no filesystem, network, or import restrictions |
| **Code execution tool isolation** | `tools/code_exec.py` | Subprocess execution without proper resource limits, network isolation, or filesystem sandboxing |

### High

| Finding | Location | Details |
|---------|----------|---------|
| No per-endpoint rate limiting | `app/main.py` | RateLimitMiddleware exists but is not registered |
| CSRF protection missing | `app/main.py` | State-changing POST/PUT/DELETE endpoints accept cross-origin requests |
| Symlink/TOCTOU attacks in file ops | `tools/file_ops.py` | Time-of-check/time-of-use race on path validation |
| Access control functions unused | `core/access_tiers.py` | Read/write root restrictions defined but never enforced |

### Medium

| Finding | Location | Details |
|---------|----------|---------|
| Security headers not configured | `app/main.py` | CSP, X-Frame-Options, X-Content-Type-Options missing |
| Case-sensitivity bypass on Windows/macOS | `tools/file_ops.py` | Path validation can be bypassed on case-insensitive filesystems |
| Tool execution lacks request-scoped rate limits | `core/brain.py` | Single conversation could exhaust resources |

**No hardcoded secrets found in source code** — all credentials properly loaded from config/.env.

---

## 5. Config Surface Area

**Total config keys:** ~160 fields in `config.py`

| Metric | Count |
|--------|-------|
| Total config fields | 160 |
| Unused config keys | 17 |
| Missing from .env.example | 12 |
| Env var name mismatches | 1 (`NOVA_API_KEY` vs `API_KEY`) |
| No logical validation | Multiple dependent-feature pairs |

### Missing from .env.example

`EMAIL_RATE_LIMIT`, `USER_TIMEZONE`, `WEB_SEARCH_*`, `ENABLE_CURIOSITY`, `MAX_CURIOSITY_PENDING`, and 7 others are configurable but not documented in `.env.example`.

### Contradictory Defaults

No runtime validation prevents configurations like `ENABLE_CURIOSITY=true` with `MAX_CURIOSITY_PENDING=0`, which silently disables the feature despite it being "enabled."

---

## 6. Architectural Debt

### Blocking Calls in Async (HIGH)

**102 total blocking calls** inside async functions:

| Category | Count | Key Files |
|----------|-------|-----------|
| Synchronous DB operations in async handlers | 30+ | `api/daemon.py`, `api/learning.py`, `api/system.py` |
| File I/O in async context | 7+ | `api/learning.py`, `core/memory.py` |
| Other sync-in-async | 65+ | Various (json.loads on large payloads, etc.) |

**Impact:** 10-100x throughput loss under concurrent load. The event loop blocks while waiting for SQLite/file I/O.

**Fix:** Wrap with `asyncio.to_thread()` or use aiosqlite.

### Memory Leak Patterns

| Pattern | Location | Risk |
|---------|----------|------|
| Auth failure tracking — unbounded dict | `app/auth.py` | O(unique_ips × requests), no cleanup task — DoS vector |
| Rate limiter O(n) eviction at 10K IPs | `app/main.py` | CPU spike when cap hit, sorts all entries |
| Per-conversation locks capped at 500 | `app/core/brain.py` | Already mitigated with LRU eviction |

### SQLite Concerns

WAL mode ✓, foreign keys ✓, busy_timeout ✓ — but `check_same_thread=False` requires careful lock discipline that isn't consistently applied across all async paths.

### Code Duplication

- 9 near-identical tool schema definitions across `app/tools/`
- 11 copy-paste migration patterns in `app/database.py`

---

## 7. Test Coverage Gaps

### Coverage Metrics

| Metric | Value |
|--------|-------|
| App modules with test files | 34/80 (42.5%) |
| Modules with zero test coverage | 46/80 |
| Total test files | 72 |
| Total test lines | ~22,445 |
| Pass/Skip/Fail | 1770 / 2 / 0 |

### Untested Critical Modules

| Module | Lines | Risk | Severity |
|--------|-------|------|----------|
| `api/chat.py` | 11 endpoints | Core user-facing API — completely untested | CRITICAL |
| `api/monitors.py` | 540 lines, 19 functions | Monitor CRUD untested | CRITICAL |
| `api/system.py` | 1200+ lines | 8+ DB operations untested | HIGH |
| `api/daemon.py` | — | Proactive orchestration untested | HIGH |
| `core/llm.py` | — | ToolCall parsing, StreamChunk generation | HIGH |
| `tools/browser.py` | — | Complex user-facing tool | HIGH |
| `core/dream.py` | — | Dream/background processing | MEDIUM |
| `core/trust.py` | — | Trust scoring | MEDIUM |
| `monitors/daemon.py` | — | Monitor daemon orchestration | HIGH |
| `monitors/event_trigger.py` | — | Event trigger processing | MEDIUM |

### Mock-Heavy Tests (Reduced Value)

`test_channels.py` and `test_brain.py` are heavily mocked — they validate mock behavior rather than actual integration. Critical paths (LLM responses, DB writes, channel message delivery) are only tested against mocks.

---

## 8. Channel Integrations

### Status: All 5 Channels Fully Functional

| Channel | Status | Key Features | Test Coverage | Issues |
|---------|--------|-------------|---------------|--------|
| **Discord** | ✅ Full | Messages, reactions, threads, voice | Partial (mocks) | Reconnection logic untested |
| **Telegram** | ✅ Full | Messages, commands, inline, media | Partial (mocks) | Command handlers untested; health check untested |
| **Signal** | ✅ Full | Messages, HTTP polling, dedup | Partial (mocks) | HTTP polling untested; dedup DB ops untested |
| **WhatsApp** | ✅ Full | Messages, HMAC verification, webhooks | Partial (mocks) | **HMAC signature verification untested (SECURITY)** |
| **Base** | Abstract | Activity tracking, message processing | Via subclasses | Activity tracking to system_state untested |

**No hardcoded secrets** — all channel tokens loaded from config.

**Critical gap:** WhatsApp HMAC verification (`whatsapp.py:79-91`) has no test coverage. A spoofed webhook could inject messages.

---

## Prioritized Fix Recommendations

### Immediate (This Week)

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Make prompt injection detection mandatory in `brain.py` chat flow | 2h | Closes critical security gap |
| P0 | Add WhatsApp HMAC verification tests | 1h | Prevents webhook spoofing |
| P0 | Register `RateLimitMiddleware` and `SecurityHeadersMiddleware` in `main.py` | 30min | Enables already-implemented security |
| P0 | Add sandboxing to DynamicTool execution (restrict imports, filesystem, network) | 4h | Prevents tool escape |
| P1 | Add try/except to chat streaming SSE handler | 1h | Prevents stream crashes |
| P1 | Create `test_chat_api.py` with endpoint tests | 4h | Covers most-used API |

### Short-Term (Next 2 Weeks)

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P1 | Wrap 30+ blocking DB calls with `asyncio.to_thread()` | 3-4h | 10-100x throughput improvement |
| P1 | Add periodic auth failure dict cleanup task | 1h | Prevents DoS memory exhaustion |
| P1 | Remove 17 unused config keys from `_MUTABLE_FIELDS` | 30min | Reduces config attack surface |
| P1 | Delete or properly implement Anthropic/OpenAI/Google provider stubs | 2h | Prevents silent misconfiguration |
| P2 | Add CSRF protection to state-changing endpoints | 2h | Prevents cross-origin attacks |
| P2 | Wire `get_allowed_read_roots()`/`get_allowed_write_roots()` into file_ops | 2h | Enables filesystem restrictions |

### Medium-Term (Next Month)

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P2 | Create test files for 46 untested modules (target 70% coverage) | 2-3 weeks | Major reliability improvement |
| P2 | Extract tool schema helpers to reduce 9 duplicate patterns | 2h | Maintainability |
| P2 | Add config.validate() logical invariant checks | 2h | Prevents contradictory configs |
| P3 | Replace mock-only channel tests with integration tests | 1 week | Real channel testing |
| P3 | Add resource limits and network isolation to code_exec | 4h | Sandboxing improvement |
| P3 | Consolidate 11 duplicate migration patterns in database.py | 3h | Maintainability |

---

*Generated by automated codebase audit — 2026-04-15*
