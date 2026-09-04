# Nova Elite Program — Implementation Plan (2026-09-01)

> **For agentic workers:** Execute inline in dependency order (Phase 0 → 4). Each task: failing test → fix → suite → live verification against the stated metric → local commit (NO push, NO Claude trailers — owner rule). Steps use checkbox syntax for tracking.

**Goal:** Turn Nova from "writes good digests on a thrashing box into a forgetful store" into a knowing system: knowledge accumulates instead of being evicted, every digest is primed by and feeds back into its dossier, forecasts are graded honestly and calibrate the next ones, research is verified sentence-by-sentence, chat answers from what Nova knows, and every optional pathway proves it is alive.

**Architecture:** No new frameworks. Fix the accumulation layer (KG eviction, dossier priming, forecasting, curiosity) first because every loop above it is starved by it; then the verification/judging layer so quality becomes measurable; then the interrogation surface; then boundaries and liveness. One rebuild + full suite + live verification per batch.

**Tech Stack:** Python 3.12 / FastAPI / SQLite (WAL) / ChromaDB / Ollama 0.32.15 (qwen3.8:27b synthesis, qwen3.5:9b-q8_0 chat, gemma4:e4b judge, bge-m3 embed) / MiniCheck sidecar / Docker Desktop WSL2.

## Global Constraints

- Quality beats throughput; never downgrade a model or component to save GPU.
- `/app` is baked into the image: every `app/` change needs `docker compose build nova && docker compose up -d nova`; verify with `bash scripts/check_container_freshness.sh`.
- Full suite: `docker exec nova-app sh -c "cd /app && python -m pytest tests/ -q --ignore=tests/slow"`; last green 3334/0. RED runs on the edited tree use an ephemeral container: `docker run --rm -v "$PWD:/app" -w /app nova-app:latest python -m pytest <file> -q` (with `MSYS_NO_PATHCONV=1` in Git Bash).
- No config flags without a reason; overrides live in `/data/config_overrides.json` and outrank `.env`.
- Never run two GPU experiments at once; never rebuild while an A/B is in flight.
- Commits: local only, plain messages, no attribution trailers.
- Measure before claiming; each task ends with its metric checked live.

---

## Phase 0 — Clear the deck

### Task 0.1: Kill the invalid priming A/B
**Why:** the replay primes only 6/16 topics (short-label lookup). GPU is better spent on the fixes.
- [ ] `docker exec nova-app sh -c 'kill 194 195 897'`; confirm no `ceiling_ab.py` / `ab_guardian.py` in `/proc/*/cmdline`.
- [ ] Save the 5 completed on-arm rows from `/tmp/replay_on.log` into `backups/prime_on16_partial_2026-09-01.txt` for the record.
- [ ] Confirm `SELECT COUNT(*) FROM monitors WHERE enabled=1 AND name LIKE 'Domain Study:%'` = 39.

### Task 0.2: Raise the WSL cap
**Files:** `C:\Users\sysadmin\.wslconfig`
- [ ] Back up to `.wslconfig.bak-2026-09-01`; set `memory=44GB`, `processors=12`, keep `swap=8GB`.
- [ ] `wsl --shutdown`; wait for Docker Desktop to relaunch the engine (launch `Docker Desktop.exe` if `docker info` fails after 90 s).
- [ ] Verify: `docker info` shows Total Memory ≈ 43.5 GiB, CPUs 12; all 9 containers healthy; `PRAGMA integrity_check` ok.
- [ ] Metric (24 h): Ollama loads/day < 150 and avg load < 10 s (`docker logs nova-ollama | grep -c "starting llama-server"`, `llama-server started in N seconds`).

### Task 0.3: Commit the previous session's pending work
- [ ] Fold `app/core/storylines.py` word-safe cut, `scripts/ceiling_ab.py` rows, `tests/test_sweep_2026_08_31.py` into batch 1's commit after the suite is green.

---

## Phase 1 — Accumulation layer (correctness)

### Task 1.1: KG stops evicting the newest knowledge
**Files:** Modify `app/core/kg.py` (`_prune`, ~2161; `get_relevant_facts` candidate load ~1661), `app/config.py` (`MAX_KG_FACTS`), `app/database.py` (migration 33: `kg_facts_fts` FTS5 over subject/predicate/object with sync triggers). Test: `tests/test_kg_eviction_2026_09.py`.
- [ ] Failing tests: (a) with cap=10 and 12 live facts, the two evicted are the oldest never-retrieved facts older than 14 days, never a fact < 14 days old; (b) facts with provenance/source in {`principle`, `cross_synthesis`, `curiosity`} are never evicted by cap; (c) `get_relevant_facts` finds a keyword match past 20k live facts (scale test, mirrors `tests/test_retrieval_scale.py`).
- [ ] Implement `_prune`: `WHERE valid_to IS NULL AND created_at < datetime('now','-14 days') AND source NOT IN (...) ORDER BY (times_retrieved = 0) DESC, julianday('now')-julianday(COALESCE(last_retrieved_at, created_at)) DESC, confidence ASC`.
- [ ] Implement FTS5 candidate generation for the keyword arm (fall back to the Python scan if the FTS table is missing); `MAX_KG_FACTS` 5000 → 50000.
- [ ] Live: no `Pruned (retired)` lines in 24 h; 7-day survival query > 90 % after a week; retrieval p95 unchanged in `[LATENCY]` context timings.

### Task 1.2: Dossier priming actually fires
**Files:** Modify `app/monitors/deep_research.py` (~3940-3960 priming block; KNOWN-VS-NEW counter ~4095-4105), `app/core/dossiers.py` (`get_domain_dossier` 376), `app/monitors/domain_study_runner.py` (pass `monitor_name` through `domain_overview`). Test: `tests/test_dossier_priming_key.py`.
- [ ] Failing test: `get_domain_dossier(db, "AI/ML", monitor_name="Domain Study: AI and ML")` returns the dossier keyed `ai-and-ml`; without monitor_name a label alias map still resolves the 26 known profiles.
- [ ] Implement: alias map built from `_DOMAIN_PROFILES` (profile label → monitor slug) + explicit `monitor_name` parameter; prime with the `## Current understanding` and `## Open questions` sections (cap 3,500 chars) instead of `body[:2500]`; counter `max_tokens=200` with a JSON schema `{new:int, updates:int, contradictions:int}`.
- [ ] Live: trigger two Domain Study monitors; both log `[Knowing] KNOWN-VS-NEW: n new | n updates | n contradictions` with integers.

### Task 1.3: Forecasting discipline
**Files:** Modify `app/core/forecasts.py` (`_MAX_DAYS`, `parse_and_store_forecast`, `_gather_evidence`, `_RESOLVE_PROMPT`, `resolve_one`, `create_forecast`), `app/core/dossiers.py` (mint prompt ~122-128 + calibration note 644-657), `app/core/storylines.py` (mint prompt ~128-131, 393-404). Migration: none (reuse `resolves_at`). Test: `tests/test_forecast_discipline.py`.
- [ ] Failing tests: (a) `FORECAST: X | resolves 2026-12-15 | 0.7` stores that date; `| 90 days |` stores +90 d (cap 365); (b) an in-claim deadline ("by Q4 2026", "in 2027", "through November 2026") later than the stored date extends `resolves_at`; (c) evidence entries with `published_date` earlier than `created_at` are dropped before the judge; (d) a resolution whose JSON lacks `evidence_date` inside the window is `unresolvable`; (e) `create_forecast` with a claim at Jaccard ≥ 0.6 to an open claim of the same family records `status='restated'` linked to the original instead of a new open row; (f) `global_calibration_note(db)` returns per-bucket text once ≥ 20 resolved.
- [ ] Implement all six; prompt rule: "Guidance, targets, projections or plans are NOT outcomes — only realized events count; reply unresolvable otherwise."
- [ ] One-off data fix (script under `backups/`, dump first): re-date the 110 open forecasts whose in-claim deadline exceeds `resolves_at`.
- [ ] Live: next Forecast Resolution run logs `evidence_date` per verdict; 0 verdicts citing pre-window evidence; new mints carry explicit dates.

### Task 1.4: Curiosity produces knowledge
**Files:** Modify `app/core/agent_loop.py` (~976 raw INSERT → `CuriosityQueue.add`), `app/monitors/heartbeat_loop.py` (`_execute_curiosity_research` ~1833-1900 provisional path; tension quota), `app/core/dossiers.py` (open-question prompt 183-200 + `Watch for` line), `app/core/brain.py` (~3655-3680 curiosity mint gate), `app/monitors/daemon.py` (urgency threshold 226). Test: `tests/test_curiosity_yield.py`.
- [ ] Failing tests: (a) agent-loop failures go through `CuriosityQueue.add` (dedup + cooldown apply); (b) a provisional resolution string starting with `[provisional] no change` is rejected (item stays pending, attempts +1); (c) a resolution must contain a dated sentence with a host to be banked; (d) every third research run picks a `dossier_tension` item when one is pending; (e) queries matching `_BREVITY_REQUESTED_RE` or Nova-self-reference never mint curiosity from reflexion/tool failure.
- [ ] Implement; dossier prompt: "Open questions must be answerable TODAY by research (present tense). Put future events under `Watch for:` lines instead."
- [ ] Live: over 24 h, curiosity monitor_results show RESOLVED/PROVISIONAL-with-source ≥ 40 % of runs; ≥ 1 tension resolution.

### Task 1.5: One-model digest chain
**Files:** Modify `app/monitors/deep_research.py` (`_overview_angles` ~1085, `_findings` ~1409, `_gap_followup` ~1320, `_verify_lead_claims` ~3460). Test: `tests/test_digest_single_model.py`.
- [ ] Failing test: with `MONITOR_SYNTHESIS_MODEL` set, every LLM call issued during `_synthesize_from_evidence` (fake llm records `model=`) uses that model; `_findings` passes `max_tokens=800`.
- [ ] Implement; keep `syn_model=None` behaviour (config default) unchanged.
- [ ] Live: `docker logs nova-ollama` shows ≤ 1 `starting llama-server` per digest run; 0 `[truncation] ... max_tokens (512)` lines.

### Task 1.6: Tools scoped by task class
**Files:** Modify `app/monitors/heartbeat_loop.py` (`_think_query` ~1232-1348), `app/core/access_tiers.py` (`CURIOSITY_TOOLS` 76, add `RESEARCH_TOOLS`), `app/core/brain.py` (generation loop: taint bit after web-ingesting tool results strips side-effect tools). Test: `tests/test_tool_scoping.py`.
- [ ] Failing tests: (a) `_think_query` runs `think()` under `set_tool_whitelist(RESEARCH_TOOLS)` and restores None after; (b) in the generation loop, once a `web_search`/`http_fetch`/`browser`/`deep_research` result is ingested, `shell_exec`, `file_ops`, `desktop`, `tool_create`, `action_email`, `action_webhook` are absent from the tool list for the rest of the turn and a requested call returns a policy error.
- [ ] Live: `action_log` gains `channel`; 0 side-effect calls with channel=monitor over 7 days; eval `autonomous-tool` unchanged.

### Task 1.7: CI green + honest badge
**Files:** Modify `app/monitors/daemon.py:92,297`, `README.md:4`. Test: extend `tests/test_curiosity_churn_fixes.py`.
- [ ] Failing test: a fresh daemon with `monotonic()` < 1800 still researches when cold.
- [ ] Implement `self._last_curiosity_research: float | None = None` and `last is None or monotonic() - last >= 1800`.

### Task 1.8: Gate `/openapi.json`
**Files:** Modify `app/main.py:667-677`. Test: `tests/test_openapi_gate.py`.
- [ ] Failing test: with `API_KEY` set, `GET /openapi.json` → 404 (or 401).

### Task 1.9: Delete list, part 1
**Files:** `app/monitors/monitor_store.py` (`seed_defaults`: Skill Validation, Capability Review, Goal Derivation, Auto-Tool Synthesis seeded disabled), `app/monitors/heartbeat_loop.py` (quiz credit path ~1427-1690 stops calling `mark_lesson_helpful`; maintenance stops trust decay), `app/core/trust.py` (no DB writes), `app/core/dream.py` + `app/core/learning.py` (remove DPO/training writers), `app/core/salience.py` (drop learned-weight term), archive `app/core/rlvr.py`, `app/core/grpo_dataset.py`, `app/core/grpo_verifier.py`, `app/monitors/domain_study_prompt.py` → `archive/`, plus their tests. Data: delete lessons 75, 92, 147, 431, 433 and goals 2-8 (dump first to `backups/`).
- [ ] Failing tests: seed_defaults leaves the four monitors disabled; a quiz PASS does not change `times_helpful`/`retrieval_score`; trust `record_*` performs no DB write.
- [ ] Implement; move modules and tests; fix imports; full suite green.

### Task 1.10: Lessons with precision
**Files:** Modify `app/config.py` (`LESSON_VECTOR_MAX_DISTANCE` 0.9 → 0.6), `app/core/learning.py` (`get_relevant_lessons` filters confidence < 0.40; `_find_similar_lesson` uses vector cosine + `_answers_conflict` requires numeric disagreement or an explicit judge), `app/core/prompt.py` (lesson line prefers `correct_answer` when `lesson_text` is a provenance string). Test: `tests/test_lesson_precision.py`.
- [ ] Failing tests: an unrelated lesson at distance 0.7 is not injected; a demoted lesson (0.35) is not returned; two paraphrase siblings (topic Jaccard 1.0, answer Jaccard 0.1, cosine 0.93) merge; "Raw GHz is not a reliable metric" vs "depends more on IPC than raw GHz" do not conflict.
- [ ] Live: mean injected lessons/turn on the eval suite ≤ 1.5 (log `Retrieved N lessons`); memory-learning direct tasks unchanged.

---

## Phase 2 — Verification and measurement

### Task 2.1: Fact/analysis contract and full-coverage gate
**Files:** Modify `app/monitors/deep_research.py` (synthesis prompt ~4012 citation rule; `_cite_uncited_sentences` 3305-3372; `_is_analytical` 2921-2942; `_entailment_gate` 3023-3305 `max_checks`; `_check_contamination` backstop 2200-2207; `_repair_dangling_fragments` 1811; retire the verify pass ~4055-4065). Test: `tests/test_fact_analysis_contract.py`.
- [ ] Failing tests: (a) sentences tagged `[analysis]` in the draft are never auto-cited and never sent to the gate; (b) a `(deep analysis: …)` pseudo-citation is rewritten to its outlet or the sentence is marked analysis; (c) with 40 cited fact sentences the gate checks all 40 (time budget, not count); (d) excision removes the whole clause and the result passes the artifact regex (`\bthe (launched|is attempting)\b`, `'s position` after whitespace, `\(deep analysis:\)`).
- [ ] Implement; persist pre-gate draft + per-claim verdicts to `/data/entail_audit/<result_id>.json` for 7 days.
- [ ] Live: per digest, entail drops < 4 with analytical share ≈ 0; artifact-regex hits 0 across the next 20 digests.

### Task 2.2: A judge that can move
**Files:** Modify `app/core/output_eval.py` (full text in chunks; remove floor-to-8 ~195-215; add deterministic canaries: stale-year bullet, artifact grammar, pseudo-citation; novelty = 1 − anchor overlap with the domain dossier). Test: `tests/test_output_eval_v2.py`.
- [ ] Failing tests: a 10k-char digest with a 2020 bullet past char 3000 scores facts ≤ 6 with the canary named in the critique; no floor is applied; novelty computed against a stub dossier.
- [ ] Live: score stdev over the next 40 grades > 1.0.

### Task 2.3: Recency and source class
**Files:** Modify `app/monitors/deep_research.py` (`_fetch_body` 856 returns `(text, published_date)` via OG/JSON-LD/URL-date; `_source_quality` 337; `_TIER1_SUFFIX` 132; `_gate_lead_credibility`), `app/core/source_authority.py` (reference-site class). Test: `tests/test_source_recency.py`.
- [ ] Failing tests: `_extract_published_date(html, url)` reads `article:published_time`, JSON-LD `datePublished`, and `/2026/08/31/` URL paths; wikipedia/britannica/investopedia are `background` class and cannot be a lead; `academia.edu` is not primary; a lead older than 72 h without `background` label is demoted.
- [ ] Live: next 20 digests: share of leads dated ≤ 72 h reported in the coverage line; 0 evergreen bullets.

### Task 2.4: Eval suite that measures the product
**Files:** Modify `evals/suite.yaml` (retire skill-match, semantic-match, retrieval, reflexion-calibration, knowing-plumbing tasks; reword `mem_scheduler_codename_paraphrase`; add deterministic research tasks), `app/monitors/eval_harness.py` (task types `resolver_window`, `priming_key`, `digest_canary` on frozen fixtures under `evals/fixtures/`). Reset `/data/eval_reports/eval_baseline.json` after one clean run.
- [ ] Failing tests for the three new task runners with fixtures.
- [ ] Live: nightly status not REGRESSION; ≥ 3 research categories between 50 % and 95 %; eval GPU seconds halved.

---

## Phase 3 — Interrogation surface and boundaries

### Task 3.1: Knowing-first chat
**Files:** Modify `app/core/brain.py` (`_kg_answers_query` 1593 → also lessons/dossiers/storylines; knowing-first gate at ~4053; dossier limit 1224), `app/core/dossiers.py` (`get_relevant_dossiers` 385-429: no LIMIT 60, 2-3 dossiers, 2,000-char excerpts, Open questions when asked), `app/tools/memory_tool.py` (search lessons + active memories), `app/core/prompt.py` (static blocks first; lessons next to KG with the same "answer from these" framing; `estimate_tokens` ×1.23). Test: `tests/test_knowing_first_chat.py`.
- [ ] Failing tests: a self-referential query with a dossier hit runs the first generation tool-less and cites the dossier; "what do you know about Anthropic" retrieves the Anthropic dossier (89-row store); `memory_search` returns a matching lesson.
- [ ] Live: 10-probe owner set ≥ 8/10 tool-less, median < 20 s; memory paraphrase tasks 5/5 with 0 tool rounds.

### Task 3.2: Talk back from Telegram/Discord
**Files:** Modify `app/monitors/heartbeat_loop.py` (`_send_alert` 3152-3290 records a `digest` assistant turn with `#r<result_id>` footer per channel user), `app/channels/telegram.py` (chat replies `parse_mode=HTML` via `to_telegram_html`), `app/core/brain.py` (correction detection accepts the digest turn as `prev_answer`). Test: `tests/test_channel_talkback.py`.
- [x] Live 2026-09-02: `channel_conversations` has the Telegram owner row and three delivered digests recorded as assistant turns (Pathway Liveness, Forecast Resolution, Dream Consolidation). [ ] A real Telegram correction turning into a lesson still needs the owner to reply to a digest.

### Task 3.3: Blast-radius boundaries
**Files:** `app/tools/file_ops.py` (`_PROTECTED_DIRS` += extensions, logs, chromadb), `app/tools/shell_exec.py` (block `/data/extensions`, `/offsite`, `/backups` writes), `docker-compose.yml` (new `backup` sidecar with `nova_data:ro`, `./backups:rw`, `E:/nova-offsite:rw`; remove those two mounts from `nova`), `app/monitors/heartbeat_loop.py` (maintenance snapshot writes to `/data/backups` only), `scripts/backup_sidecar.sh`, `app/tools/browser.py` (minimal env at launch), `requirements.txt` (playwright → current), `Dockerfile` (browser install), socket-proxy explicit allowlist (custom haproxy cfg or nginx). Tests: `tests/test_blast_radius.py`.
- [x] Live 2026-09-02: nova-app mounts = /data, /data/mcp(ro), /exec_queue, ~/.cache only; `nova-backup` sidecar verifies + syncs (first run copied nothing — lexicographic pick of the stale `nova-premove.db`; fixed to newest-by-mtime with fallback, second run "already present" on both legs); browser env = 6 vars, 0 token-named; proxy: GET json 200, DELETE 403, POST exec 403, GET list 403, POST restart 204. Two compose fixes were needed live: the sidecar script is not in the image (directory mount of ./scripts) and nginx's worker uid could not open docker.sock (workers run as root).

### Task 3.4: Liveness registry and the fake-LLM tick
**Files:** Create `app/monitors/pathways.py` (`PATHWAYS` table: name, enable flag, writer table, filter, cadence, min_rows) + `_execute_pathway_liveness` in the fast lane; `tests/test_pathway_liveness.py`; `tests/test_heartbeat_tick_e2e.py` (scripted fake LLM + fake search, one `_loop` tick, asserts rows in monitor_results, pending_deliveries, kg_facts, storyline_events, dossier_revisions, forecasts); `tests/test_invariants.py` (schema snapshot vs `init_schema`, every `ENABLE_*` read outside config and documented in `.env.example`, every path in CLAUDE.md/README exists, default model agrees across README/.env.example/config).
- [x] Unit: `tests/test_pathway_liveness.py` (18), `tests/test_invariants.py` (13), `tests/test_heartbeat_tick_e2e.py` (2 — one real `_loop` tick on a scripted provider reaches monitor_results, KG, delivery journal, storyline_events, dossier_revisions, forecasts, curiosity; the negative control deletes the `storyline` dispatch key and is caught). Inert flags ENABLE_LORA_CONTINUAL_MERGE / ENABLE_SFT_BOOTSTRAP / ENABLE_MCP_SERVER deleted; 24 undocumented flags added to `.env.example`; 9 dead CLAUDE.md paths repointed at the archive.
- [x] Live 2026-09-02 06:40 UTC: `/api/status` (system router is mounted at `/api`) lists 26 pathways, all alive with last_at/age/window; the "Pathway Liveness" monitor ran on the first tick ("all pathways alive (26 writing)"); the e2e negative control (deleted `storyline` key) fails as designed.

### Task 3.5: Runner context floor (found 2026-09-02 while verifying Phase 1+2 live)
**Files:** `app/core/providers/ollama.py` (`_RUNNER_CTX_FLOOR = 24576`, `_runner_ctx()`, always set `payload["options"]["num_ctx"]`, tripwire uses the floor), `docker-compose.yml` (`OLLAMA_NUM_CTX: "24576"`), `tests/test_runner_ctx_floor.py`.
- [x] Root cause measured: after the WSL fix the 27B was still restarted 92× in 150 min (63 min loading) with only 11 real model swaps — Ollama restarts the runner on every `-c` change (8192 → 16384 → 20480 → 24576 across the digest chain).
- [x] Live 2026-09-02: first hour after deploy = 2 `starting llama-server` lines total (one per model) vs 92 per 150 min before; every slot reports `n_ctx_slot = 24576`.

---

## Phase 5 — Found by measurement 2026-09-03/04 (not in the original program)

Everything here came from looking at what Nova PRODUCED rather than at whether
tests passed. Ordered by measured value.

- [ ] **5.1 Ensemble the forecast probability.** Nova takes ONE verbalized `0.x`
  from a single generation — the weakest method in the literature — and is 15
  points overconfident (hit 0.60 at stated 0.75, legacy regime). Sample the
  probability k times and aggregate. Cheap: the FORECAST line is a short
  generation. Measure inside the new regime (`forecasts.REGIME`) so the effect
  is attributable. Evidence: verbalized confidence is consistently poorly
  calibrated and sampling-based aggregation beats it (arxiv 2412.14737,
  Science Advances adp1528).
- [ ] **5.2 Entailment cascade — 1.55x on the digest bottleneck, zero quality
  cost.** Entailment is 64% of a digest (19.8 of 30.8 min) with the GPU idle.
  Measured on 60 real pairs: a 2,754-char document scores 3.6x faster and
  agrees with the 5,508-char production document 93% of the time — but newly
  DROPS 4 of 60 supported claims, which is a quality cut and therefore refused.
  Crucially nothing narrow-supported was rejected at full width, so scoring
  narrow first and re-checking ONLY the unsupported at full width gives
  verdicts identical to today at ~1.55x. Take the cascade, not the trim.
- [ ] **5.3 Resolution criteria at mint.** Metaculus practice: a threshold and
  an authoritative source in the FORECAST line. NOTE: the obvious version of
  this (reject claims with no number) was measured and REFUSED — no-number
  claims are unresolvable 6% of the time vs 9% for numbered ones. This is about
  informativeness, not rescue.
- [ ] **5.4 Clean-room re-grade (owner approved).** Re-score a sample of pre-
  and post-fix digests with the CURRENT judge so the instrument is constant and
  only the code era differs. ~1h GPU. This is the only honest way to answer
  'is Nova actually better now', because the judge changed on 09-02.
- [ ] **5.5 Learned recalibration — BLOCKED, revisit later.** The best post-hoc
  method (Beta-Bernoulli, arxiv 2605.27668) trains on 7,824 resolved questions;
  Nova has 103 and all are legacy-regime. Revisit at ~30+ resolved in the
  current regime.

### Shipped 2026-09-03/04 (measurement-driven, outside the plan)
- [x] Priming RETIRED on a 16-topic paired A/B (no gain, cost to grounding).
- [x] Self-citation stripper widened — a guard and a prompt written against
      each other let `(deep analysis)` climb from ~10% to 55% of digests.
- [x] Curiosity backpressure (the queue was destroying unread questions).
- [x] Schedule pressure measured and surfaced: 1,646 runs demanded/week, 608
      delivered (37%).
- [x] Entity timelines — the bitemporal trail nothing had ever read.
- [x] Storyline re-attachment; markup stripped at every write boundary.
- [x] Regime stamps on forecasts AND judge scores; `scripts/quality_panel.py`.
- [x] Entailment threads pinned to the cgroup quota (18%); more CPUs measured
      WORSE (13.09 vs 8.26 s/pair) and refused.

## Phase 4 — Structural (after the above are live)

- Task 4.1: Split HeartbeatLoop into scheduler / executors / delivery / maintenance-steps with a `pathway_runs` ledger. — NOT STARTED (the pathway registry + e2e tick now give the safety net for it).
- Task 4.2: Extract `deep_research` fetch.py / verify.py; `brain` context.py / generation.py / post.py. — NOT STARTED.
- Task 4.3: Scheduler v2 — [x] residency classes (`_monitor_class`: digest=27B incl. consolidation/synthesis, judge, other=9B) + [x] model-residency batching (`_batch_by_class` after the starvation floor; swaps bounded by classes−1 per tick; `[Heartbeat] batch of N … (k model swap(s))` log) — `tests/test_scheduler_batching.py`. [x] quiet-window primitive (`app/monitors/quiet.py`, `POST/GET/DELETE /api/monitors/quiet`, tick skips the LLM lane without advancing last_check_at, `/api/status.quiet_until`, `ceiling_ab.py --replay` opens one for itself; `tests/test_quiet_window.py`). [x] **backpressure, 2026-09-03** — two kinds, both found by measurement: (a) the curiosity queue was a treadmill (pinned at 100, ~5/day drained, 152 logged evictions of the OLDEST pending topic — the KG ring-buffer shape again, and `MAX_CURIOSITY_PENDING` was read nowhere) → background work is now refused at the high-water mark, the ceiling evicts by value not age, topics are sanitised of markdown, and the ledger no longer records a refused `-1` as queued (`tests/test_curiosity_backpressure.py`); (b) `schedule_pressure()` measures delivered vs demanded runs — **1,646 demanded/week vs 646 delivered (39%), Curiosity Research at 20% of its declared cadence** — reported in the liveness fields and `/api/status.schedule`, escalating only when severe and only with enough history (`tests/test_schedule_pressure.py`). Closing the 39% gap needs either more capacity or honest cadences: an owner decision, not mine.
- Live 2026-09-02 07:00 UTC (image 363189dc5849): first post-restart tick logged `batch of 23 … (1 model swap(s))`; ledger backfilled from 90 dossiers → 175 questions + 8 belief revisions; the synthetic "Zephyr Semiconductor export saga" (June eval fiction: storyline 11, dossier 67, 4 questions) was found through the ledger and purged (dump in /data/backups); 34 questions belonging to 18 closed storyline dossiers retired → frontier 136 open / 35 retired; `retire_orphaned()` now runs each consolidation cycle; 27/27 pathways alive.
- Task 4.4: [x] Open-questions ledger (`app/core/questions.py`, migration 33: `dossier_questions` open/queued/researched/retired + `belief_revisions`; hooks in consolidation, curiosity feed and curiosity resolve; API `/api/dossiers/questions|beliefs|{id}/questions`; `question_ledger` pathway; `tests/test_open_questions_ledger.py`; e2e tick asserts the fed question is `queued`). [x] storyline re-attachment (a story returning after its thread closed reattaches within 60 days and revives it instead of forking; found live — two Hormuz threads, two global-instability threads; `tests/test_storyline_reattach.py`). [x] **entity timelines, 2026-09-03 — built, but NOT in synthesis.** `app/core/timelines.py` reads the dated trail the bitemporal KG has kept since 2026-05-16 and nothing consumed: facts learned, facts REVISED (`superseded_at` + the replacing object), storyline events naming the subject, consolidation `REVISED:` lines. The defect it actually fixes: `_entity_sources` fed the entity-dossier prompt undated, supersession-stripped facts while that prompt demanded "5-10 dated bullets of the major shifts, oldest→newest" — output judged against a structure it never had, again. Exposed read-only at `GET /api/kg/timeline`. Kept OUT of digest synthesis on purpose (the same-day A/B measured prior context there as a grounding cost); `tests/test_entity_timelines.py` pins that separation. Live check: Iran carries 29 revisions, nvidia 40 events.
- Task 4.5: [x] MCP knowing-tier tools (`nova_dossiers`, `nova_storylines`, `nova_forecasts` — 8 tools). [ ] UI: revision diff, calibration chart, honest counts — deferred (frontend comes last per owner).
- Task 4.6: [x] **DONE 2026-09-03 — and it reversed the feature.** n=16 paired, same frozen evidence, arms run concurrently, 16/16 topics dossier-resolved (the void run managed 6/16). Priming showed no gain on any metric and cost deterministic fact support (0.642 vs 0.690, down on 12 of 16 topics, p=0.099); overall 0.838 vs 0.844 (p=0.61). The PRIOR UNDERSTANDING injection was removed from the synthesis prompt; the dossier is still loaded for the out-of-band KNOWN-VS-NEW count, which runs on the finished briefing. Cost measured: ~38 min/topic, of which 23 is live re-citation search. Harness now appends per-topic rows so a partial run is analysable. Guard: `tests/test_priming_retired.py`.
- Task 4.6 (original): Re-run the priming A/B (n=16) in a quiet window with the fixed key — needs an owner-chosen quiet window (pauses the 39 domain monitors for hours). Recipe: `docker exec nova-app python scripts/ceiling_ab.py --help` (passes `dossier_key=monitor` since Phase 1.2).
