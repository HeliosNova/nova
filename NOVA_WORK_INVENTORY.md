# Nova — Complete Evaluation & Work Inventory
*Compiled 2026-08-29. Every item is either MEASURED (with the number) or explicitly marked NOT YET EVALUATED.*

---

## A. CLOSED — fixed and verified tonight (21 defects, 9 commits)

| # | Defect | Evidence | Commit |
|---|---|---|---|
| 1 | Feed digests rendering as bare links — summarizer confabulating from a 900-char masthead window | `"founded in 2001"` absent from page, p=0.013 at every width; 9→13 summaries | 939e717 |
| 2 | Chat grounding gate 76% inert | 132.1s work on a 60s budget → 18.7s (7.1×), 8/8 supported | 939e717 |
| 3 | KG contradiction judge fail-open 22% | 374-char `reasoning` field truncating JSON → 16 chars, 4/4 clean | 939e717 |
| 4 | BrowserTool never registered (since introduction) | closure shadowing reproduced exactly; now `name='browser'` | 939e717 |
| 5 | dedup: 5 DB ops (3 writes) on event loop | warnings 5/day → 0 | 939e717 |
| 6 | `Planning failed:` / searxng logging blank exceptions | `%s` on empty-str timeouts → `%r` | 939e717 |
| 7 | Warmup tripping the truncation tripwire every startup | false positive gone | 939e717 |
| 8 | Enrich gate timing out → **publishing unverified summaries** | 10-pair batch needs ~110s on a 120s all-or-nothing request → chunked, worst 35.7s vs 90s | c4faf8a |
| 9 | Fail-open sentinel inverted by `min_prob` floor | `-1.0 >= 0.05` = False would DROP items when gate degraded | c4faf8a |
| 10 | KG LLM curation dead — string `"id"` aborted every batch | **0 successes / 1 failure in 48h** → schema-pinned; has since retired 25 junk facts unaided | c4faf8a |
| 11–14 | 4 event-loop DB writes (`kg.curate`, `trust.decay`, lesson chain ×3) | tripwire → 0 | 3b67a33 |
| 15–20 | 6 more event-loop writes (`forecasts`, `output_eval`, `kg_communities`, `custom_tools`×3, `heartbeat`, `storylines`) | found by re-reading the tripwire after each deploy | 6ae8d99, 89d9b87 |
| 21 | Quality grader scoring on absent evidence since **June** | freshness 7.48→**9.18**, format 7.55→**10.0**, facts flat (control) | 89d9b87, 0e92cd0 |
| 22 | Operator sanity probes auto-spawning research monitors | `"reply with exactly: operational"` → 12-hourly research job; 7/7 probes blocked, 7/7 real topics kept | 4c7a154 |
| 23 | KG pipeline-artifact triples: 0.3% of facts, **9.6% of all retrieval** | 13/13 blocked, 0 false positives on 4944 live facts | pending |
| 24 | **Memory retrieval buries newly-learned lessons** | d=0.386 rank-1 match cut at rank 6/9; `retrieval_score` 0.5 (new) vs 0.96 (established) | pending |

**Suite: 3089/0** (from 3057 — 32 new tests, no regressions across 11 full runs).

---

## B. OPEN — measured, root-caused, not yet fixed

Ranked by (impact × confidence) / effort.

| P | Item | What's known | Effort |
|---|---|---|---|
| ~~B1~~ | ~~Cold-Q = 42.4%~~ **CLOSED — investigated, not a defect** | Cold memory is NEWER (20.1d vs 49.7d) and HIGHER confidence (0.949 vs 0.85). Cold rate by age: 0-2d **90%**, 2-7d 85%, 7-30d 71%, **30d+ only 18%** — facts warm up over weeks, so the headline number is an AGE artifact. KG ranks by `confidence`, never by `times_retrieved`, so there is **no rich-get-richer bias** here (unlike lessons, defect #24 — hypothesis checked and disproved). Genuinely stale set is just **527 facts (10.7%)**, and sampling shows real knowledge, not junk. Lessons are healthy: 44/50 ever-helpful, 2 never retrieved. **A learned memory controller would have been large effort against a non-problem.** RoMeRL's 84% reduction targets agent *trajectory* memory, not a curated bitemporal KG | — |
| **B1b** | Predicate-level cold skew worth a look | `acquired` **83% cold** (95 facts), `invested_in` 69%, `located_in` 59% (545 facts). Either over-extraction or the retriever never surfaces them for relevant queries. Feeds into E2 | S |
| **B2** | ~~Goals subsystem dead~~ **ROOT-CAUSED + FIXED (pending test)** | TWO defects: (1) **ordering** — `_execute_capability_review` marked every gap `reviewed=1` and THEN called `derive_goals()`, which selects `WHERE reviewed=0`; the review consumed its own input, so the capability-gap source could **never** mint a goal (live: 18 gaps, all reviewed, 9 in 7d, 0 unreviewed). (2) **cluster key by position** — `words[:3]` filtered stop words *after* slicing, so the key came from whatever opened the query; research queries all open with an imperative, producing goals literally titled "capability gap: does / higher / clock" from one CPU question, all failed | done |
| **B3** | `related_to` junk dominating retrieval | Largest predicate: 892 facts, **35,041 retrievals**. Top-60 sample: only **44% of retrieval** goes to facts that look real; rest is quantity/date/fragment residue. Existing gate insufficient | M |
| **B4** | Residual bare links in feeds | HN self-posts / comment permalinks / `hnrss.org` have genuinely **no article** (0 bytes). Decide: suppress the row (like the contracts special-case) or accept title+link | S |
| **B5** | `web_search` circuit breaker | Fired 17×/48h, `MAX_SAME_TOOL_CALLS=3`. Dedup layer already stops true loops; the cap may be starving multi-part research. **Do not raise blindly** — confirm which path trips it | S |
| **B6** | Lazy-embedder cold-start race | A lesson written seconds after restart may not be immediately queryable (bit my own verification). Unproven in production | S |
| **B7** | Orphaned ChromaDB vectors | 51 vectors vs 50 lessons. Delete-side of the lesson lifecycle leaks | S |
| **B8** | Dream Phase 2b (REM) timeouts | 3× in 48h | S |
| **B9** | Playwright `TargetClosedError` ×15 | Browser teardown race, unretrieved futures. Noisy, likely benign | S |
| **B10** | ~35 remaining sync-DB **reads** on event loop | Reads take no write lock (WAL snapshot) → latency only, not the freeze class | M |

---

## C. DORMANT / SUSPECT — liveness measured, needs judgement

*This is the category I'd been missing by fixing leaves.*

| Subsystem | Live signal (7d) | Verdict |
|---|---|---|
| KG facts | **+614** | healthy |
| Storyline events | **+393** | healthy |
| Curiosity | +91 created, **0 open** | churning — created and closed, is it producing value? |
| Forecasts | 45 resolved, **0 pending** | resolving but is it *minting*? |
| Lessons | +15 | low but alive |
| Reflexions | +11 | low but alive |
| Skills | 6 total, 5 ever used | **near-dormant** — 2 organic skills ever created |
| **Goals** | **0 active, 7/8 failed** | **DEAD — see B2** |
| Dossiers | 85 / 605 revisions | alive (no timestamp column to verify recency) |

---

## D. NEVER EVALUATED — no test file, no liveness check, not looked at tonight

**21 core modules have no test file at all:**
`access_tiers` `agent_loop` `agent_workspace` `auto_tools` `brain_context_manager` `brain_kg` `brain_routing` `brain_sanitize` `critique` `cross_monitor` `extensions` `goals` `llm` `memory` `output_eval` `prompt_optimizer_baselines` `skill_loader` `source_authority` `task_manager`

**Never exercised or audited this session:**
- `voice` — voice pipeline, entirely unverified
- `injection` / `access_tiers` — prompt-injection defence and permission tiers (security-relevant)
- `backup` / `data_export` — DR path; last restore drill was 2026-08-24
- `rlvr`, `grpo_dataset`, `grpo_verifier` — RL scaffolding, status unknown
- `gsw` — global summary/working memory
- `ppr` — personalized PageRank (part of retrieval fusion, never measured)
- `principles`, `salience`, `calibration`, `quality` — never checked for liveness
- `platform`, `extensions`, `skill_export`, `auto_skills`
- **Frontend** (`frontend/`) — not touched at all
- **74 monitors** — only ~6 inspected individually

---

## E. STRATEGIC — the 2026 frontier gap

**Memory is the one place Nova is genuinely behind.** Three 2026 papers converge on *learned* memory controllers: Memory-R1 (ACL 2026), RoMeRL (arXiv 2608.02508, Aug 2026), NEMORI (Apr 2026). Memory-R1's critique names Nova's design: *"static and heuristic-driven, lacking a learned mechanism for deciding what to store, update, or retrieve."*

**Smaller than first thought:** the utility telemetry already exists. The work is a *controller* on existing signal, not new instrumentation.

- **E1** Memory utility controller — use `times_retrieved` / `times_helpful` / `retrieval_score` to prune, promote and bound memory (targets B1's 42.4% cold)
- **E2** Retrieval ranking is now known-fragile (defect #24). Audit the whole fusion: PPR arm never measured, `related_to` pollutes, Q-value blend has a rich-get-richer bias
- **E3** Grounding is **at parity** — arXiv:2607.04223 (2026) confirms chunk-level NLI still matches newer methods. No model change needed; the problem was gates that couldn't finish
- **E4** Deep-research: the "2026 SOTA" comparison was partly fiction — Tongyi DeepResearch's repo dates to **2025-09-16**. Re-benchmark against genuinely current work before adopting anything

---

## F. OWNER ACTIONS

- `git push origin main` — commits pending
- Monitor 77 `Auto: reply with exactly: operational` — **disabled**, not deleted; your call
- `backups/_*` scratch scripts — removable
- Two new monitors live: **[78] Macroeconomics**, **[79] Microeconomics** — first digests due within 12h
- Old Ollama images `latest` + `0.32.13` (~8.4GB) — unused but `0.32.13` is your rollback point. `0.30.11` **must stay** (pinned for nova-embed)

---

## RECOMMENDED ORDER

1. **B2 goals** + **C skills/curiosity liveness** — whole loops dead or churning; biggest blind spot
2. **B1 Cold-Q → E1 controller** — the real frontier work, and the telemetry already exists
3. **E2 retrieval audit** — defect #24 proved this layer is fragile; PPR arm never measured
4. **D security modules** (`injection`, `access_tiers`) — never audited, and security-relevant
5. **B3–B5** monitor/KG cleanup — bounded, known
6. **D test coverage** — 21 core modules with zero tests
