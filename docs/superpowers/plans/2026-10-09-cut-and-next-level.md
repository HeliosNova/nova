# Cut and Next Level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the second round of subsystems the record shows nobody uses, then ship the five measured changes from the 2026-10-09 research sweep that move the knowing tier: store-level temporal guards, a planting-probe audit of the entailment rescue path, directive search refusals with a repaired engine ladder, extraction-first source counting, and curiosity pointed inward at Nova's own records.

**Architecture:** Every change lands as one change class per deploy, test-first, with its decision rule pre-registered below and the measurement run on the live record (SQLite at `/data/nova.db`, logs at `/data/logs/nova-app.log*`) before and after. GPU-bound measurements run inside a quiet window (`POST /api/monitors/quiet`). Cuts remove modules, their wiring, their flags, their seeds and their tests together, pinned by a guard test the way `tests/test_cut_2026_10_09.py` pins round one.

**Tech Stack:** Python 3.12 / FastAPI / SQLite (SafeDB), Ollama 0.32.15 (qwen3.8:27b synthesis, nova-ft 9B, gemma4:e4b judge), MiniCheck flan-t5-large sidecar (CPU), SearXNG, Docker Compose; tests run with the tree mounted: `MSYS_NO_PATHCONV=1 docker run --rm -v "F:\Helios Project\nova_:/app" -w /app nova-app:latest python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/slow`.

## Global Constraints

- Nothing outside the repo and its containers changes: no host, Docker Desktop, Windows, GPU-reserve or drive settings. Ollama stays pinned at 0.32.15 in this plan.
- Monitor cadences are not relaxed. The VRAM reserve stays. MiniCheck stays on CPU unless a task's measurement says otherwise and the owner decides.
- One change class per deploy. Deploy only into a quiet window after in-flight digests drain (recipe: `POST /api/monitors/quiet {"hours": 2}` → wait for digest-class results → `docker compose up -d nova` → `DELETE /api/monitors/quiet`).
- Never name a forbidden string in a prompt; say what is wrong semantically.
- Every `ENABLE_*` flag has a reader and a `.env.example` line, or neither (`tests/test_invariants.py`). Every seeded `check_type` has a dispatch handler. A dropped table means regenerating `tests/schema_snapshot.json` with `UPDATE_SCHEMA_SNAPSHOT=1`.
- Every backticked path in `CLAUDE.md`/`README.md` must exist or sit within two lines of "removed/retired/moved".
- No silent writes: a swallowed exception around a write logs, or is listed with a reason in `tests/test_silent_writes.py`.
- Run `python scripts/sanity_scan.py` (undefined names, duplicate defs) over every touched module before the suite.
- Commits carry no Claude attribution. LF line endings only (`.gitattributes`).
- The suite proves the code does what was written; a prompt or gate change ships with a check on real stored output.

## Pre-registered decision rules

Written before any measurement runs. A change ships only if its rule passes; otherwise the instrument stays and the change does not.

| # | Change | Measurement | Ships if |
|---|---|---|---|
| G1 | Re-read/revert guard on `add_fact` | A-B-A chains in the live `kg_facts` (A superseded by B, B later superseded by A, reviving source not newer than B's) | count ≥ 1 |
| G2 | Aspect-preserving extraction | Share of a 200-fact random sample of `source='researched'` live facts whose predicate or object was flattened to simple present where the digest sentence carried an ongoing/as-of form | share ≥ 30% |
| P1 | Entailment rescue path | Planting-probe rate (citation migrates to a doctored uncited passage) on 200 rescued sentences vs 200 first-pass-passed sentences | if rescued − first-pass > 5 pp, the rescue layers are disabled pending redesign; if ≤ 5 pp they stay and the number is recorded |
| R1 | Directive search refusals | Digests and curiosity answers emitted from runs where every engine returned zero results, one week before vs one week after | emitted-from-nothing count falls and zero-result runs mark a gap instead |
| R2 | Engine ladder (`default_lang`, `braveapi`) | Per-engine non-empty-result rate over 48 h before vs after | any engine goes from 0% to > 50% non-empty, no engine regresses |
| X1 | Extraction-first source counting for digests | Distinct-origin sources vs cited URLs on 50 stored digests (shingle near-dup), and fact support on 16 frozen topics via `scripts/ceiling_ab.py --replay` | support within 0.02 of baseline and cited-copy share falls |
| X2 | Extraction-first context for forecast minting | Brier on the 144-claim resolved replay (`scripts/forecast_backtest.py`), k ≥ 8 | Brier improves by ≥ 0.02 (the measured noise floor) |
| C1 | Inward curiosity for dossier tensions | Open tensions closed from Nova's own spans, each verified by a second independent judge | ≥ 50% closed and ≤ 10% later judged unsupported |
| C2 | Answerability triage for the curiosity backlog | AUROC of the two probes (resolvable now? evidence sufficient?) against actual resolve/fail on the next 40 attempts | AUROC > 0.85 |
| K1..K5 | Round-two cuts | Full suite green on the cut image; freshness 0 stale; every pathway alive/idle/off with the owner's cadences intact; delivered runs per active hour not down vs the week before (`scripts/throughput_panel.py`) | all four hold |

Noise floor to respect: at k=3, T=0.7 two estimates of the same claim differ by |0.115|; paired counts near 72/144 and Brier deltas under 0.02 are unreadable. Use k ≥ 8 or more claims.

---

## Program order

Cuts first, because every later measurement runs faster on a smaller surface and suite. Guards second, because they cost no model time. The audit third, because it decides whether the gate is honest before anything leans on it. Search and context fourth. Inward curiosity last, because it is the largest and depends on the guards and the audit.

| Phase | Tasks | Change class |
|---|---|---|
| K | Round-two cuts: exec sidecar, custom/auto tools, agent loop, skills, quiz/reflexions | removal |
| G | Temporal guards: A-B-A measurement, revert guard, aspect audit, aspect-preserving extraction | store rules + extraction prompt |
| P | Planting-probe audit of the rescue path | instrument + possible gate change |
| R | Directive refusals + engine ladder | search tool + SearXNG config |
| X | Extraction-first source counting (digest, then forecasts) | prompts |
| C | Inward curiosity: answerability triage, tension resolver over own spans, dependency-tracked dossier repair | curiosity runner + dossiers |

Tasks follow, written from the six subsystem maps (2026-10-09 workflow, verified
against the live DB). Line numbers are as of commit ad9dd9d; re-grep before editing.

## What the maps changed about the plan

- **Tensions never reach a dossier.** `_numeric_tensions` output is only appended to
  the Knowledge Consolidation result string (dossiers.py:1016-1058); 0 of 107 dossier
  bodies carry the marker, 31 of 62 consolidation results do, and nothing reads them
  back. The retired resolver web-searched the literal tension sentence
  (deep_research.py:1143-1161 cannot find a question in it). C1 is a NEW module over
  Nova's own stored digests, not a repair of the old path.
- **The KG has no world time.** valid_from == created_at on 25,089 of 25,091 rows
  because no extractor passes a publication date (only eval does). Every digest is
  extracted TWICE (deep_research._learn_facts source='researched' and the post-digest
  heartbeat pass source='extracted') under different provenance strings, which
  add_fact counts as independent corroboration (kg.py:1038-1046: trust +0.2 and
  quarantine release). 37 A-B-A chains exist, 30 of them on `leads` with
  entity-variant churn ("Federal Reserve" / "FOMC" / "Federal Reserve (nominee)").
  Revival (kg.py:1107-1127) wipes the revived row's history. G2 becomes "pass the
  publication date as valid_from and stop double-extracting"; the tense audit is
  closed: predicates are canonical snake_case and 0 of 2,973 live researched facts
  carry a tense marker, so there is nothing to preserve.
- **No gate tuple is stored anywhere.** The planting probe needs a persistence hook
  at the three cascade call sites first (deep_research.py:3490, 3540, 3578).
- **Nothing reaches the 27B on zero results in the monitor path** (fixed strings at
  deep_research.py:4113-4118, 4720-4722; domain_study_runner.py:1290-1291). The real
  search defect is invisibility: native_search never reads SearXNG's
  `unresponsive_engines`, so "searxng:general returned 16" can mean bing+yandex only
  while brave, google, startpage sit suspended (1,658 upstream failures in 31 h);
  three engines in settings.yml never load (ecosia, presearch, reddit). R1 is
  therefore per-engine visibility plus a directive string on the chat/think tool
  path; R2 is the settings repair.
- **The extraction-first seam exists.** Per-article findings already come from
  `_findings` (deep_research.py:1546-1589); the cross-source "distinct readings"
  block goes between `_annotated_evidence` and `_deep_analyze` (4467-4481), where
  `_independence_clusters` and `_figure_support` already compute which clusters
  state which figure.
- **Long-memory on the 10-09 run (pre-deploy of afdb491):** temporal 0/1, aggregate
  0/1, multi-valued 0/1, all because the brain web-searched a fictional entity
  instead of answering from the seeded facts. The eval baseline (2026-09-07) predates
  the category, so it is never regression-compared.

## Cut order and the cross-cut edges that force it

(a) exec sidecar and (b) custom/auto tools go together: custom_tools.py:33 imports
code_exec. (c) agent loop goes with them: agent_loop.py:936/1327 writes
auto_tool_candidates and builds CustomToolStore. (d) skills and (e) quiz/reflexions go
together: auto_skills.py:433 reads reflexions, _execute_quiz:1714 writes one.
Two deploys: K1 (a+b+c) and K2 (d+e).

---

### Task K1: Cut the agent tool layer (exec sidecar, custom/auto tools, agent loop)

**Files:**
- Delete: `app/exec_runner.py`, `app/tools/code_exec.py`, `app/tools/code_verify.py`,
  `app/tools/shell_exec.py`, `app/core/custom_tools.py`, `app/core/auto_tools.py`,
  `app/core/tool_triggers.py`, `app/core/self_improvement.py`, `app/core/agent_loop.py`,
  `app/core/agent_workspace.py`, `app/tools/delegate.py`, `app/api/agent.py`
- Create: `app/core/reasoning_cues.py` (the three predicates brain imports from
  agent_loop: `_is_math_logic_word_problem` agent_loop.py:141, `_is_hard_reasoning_query`
  :149, `_is_deliberation_query` :526 with `_DELIBERATION_QUERY_RE` :514, moved verbatim)
- Create: `tests/test_cut_2026_10_10_agent_layer.py` (guard, modelled on
  `tests/test_cut_2026_10_09.py`)
- Modify: `app/main.py` (imports :38-39,49,56; registrations :206-207,211,218-219;
  custom tools block :247-258; Services custom_tools :320; agent router :801-802)
- Modify: `app/core/brain.py` (TYPE_CHECKING import :26; Services.custom_tools :460;
  `_handle_tool_create` :489-520; TOOL_CREATE_DESCRIPTION :563-566; tool_create parse
  :1545-1558; dispatch :2242-2243; autonomous trigger :2448-2462; tool schema :4157-4171;
  `_SIDE_EFFECT_TOOLS`/`_TAINT_STRIPPED_TOOLS` :82-98 drop shell_exec/code_exec/
  tool_create/delegate; fallback lists :560,:580 drop code_exec; `[MATH TOOL ROUTING]`
  :1465-1482 keep calculator only; substantive list :4549-4555; agent_loop imports
  :263, :988, :1807, :3368 to `app.core.reasoning_cues`; deliberation gate :4112-4147,
  `_run_deliberation_path` :2569-2630, `_DELIBERATION_PATTERNS`/`_should_use_deliberation`
  :2542-2566; workspace :1233-1234, :4493-4495, `_ThinkContext.workspace_text` :807;
  delegate strip :4172-4173)
- Modify: `app/core/prompt.py` (:50,:70,:75,:80,:86,:102,:191-197,:280,:327,:329,:360,
  :362,:371: remove code_exec/shell_exec/code_verify/tool_create/delegate text and the
  sqlite3-via-code_exec guidance; keep calculator)
- Modify: `app/core/trust.py` (:29,:31,:33), `app/core/access_tiers.py` (drop
  `set_access_tier_override`/`get_access_tier_override` :28-41,:104-105; delegate was the
  only user; keep `research_scope`)
- Modify: `app/core/curiosity.py` (:77 CURIOSITY_TOOLS name list), `app/core/reflexion.py`
  (:257,:260 names), `app/tools/file_ops.py:42`, `app/tools/base.py:238` (names only)
- Modify: `app/core/learning.py:995-999` (custom_tools SQL out of get_learning_summary)
- Modify: `app/monitors/maintenance.py:257-275` (auto_tools prune block out)
- Modify: `app/core/dream.py:302-318` (self_improvement prune out)
- Modify: `app/api/system.py` (:104,:141 counts; :152-162 self-improvement route;
  :239-240,:293 delegation config; :282,:284,:299 exec timeouts/flag; :291 custom tools
  flag; :573-600 custom-tools routes; :798,:820,:840,:981-1002 export/import)
- Modify: `app/schema.py:182` StatusResponse.custom_tools
- Modify: `app/config.py` (:44-45,:47-48,:51-52 name lists; CODE_EXEC_TIMEOUT :247,
  SHELL_EXEC_TIMEOUT :257, ENABLE_SHELL_EXEC :264, ENABLE_CUSTOM_TOOLS :309,
  ENABLE_AUTONOMOUS_TOOL_CREATION :369, AUTO_TOOL_CREATION_THRESHOLD,
  MAX_CUSTOM_TOOL_CODE_LENGTH, MAX_CUSTOM_TOOLS, ENABLE_DELEGATION :331,
  MAX_DELEGATION_DEPTH :332, ENABLE_DELIBERATION :338)
- Modify: `.env.example` (:48-49,:54,:82-83,:96,:101-103,:139-141,:248)
- Modify: `docker-compose.yml` (ENABLE_SHELL_EXEC :152; `exec_queue:/exec_queue` :167-168;
  service exec-runner :275-299; top-level volume :452; pids_limit comment :186-190)
- Modify: `app/database.py` (CREATE TABLE custom_tools :133-144; migration 18
  agent_workspace :750-785 removed; regenerate `tests/schema_snapshot.json`)
- Modify: `app/monitors/pathways.py:144-146` (agent_workspace row out; :175 exclusion
  list gains agent_workspace), `tests/test_pathway_coverage.py:51-55` becomes an absence
  assertion (precedent :45-48)
- Modify: `app/monitors/eval_checks.py:165-178` (rq_tool_scoping: assert
  `file_ops`/`email_send` are taint-stripped and denied under research_scope instead
  of shell_exec/tool_create), `evals/suite.yaml` (remove tool_code_sum :153-162 and
  tool_code_primes :164-173; drop the `deliberation-route` tag on
  deliberation_chain_of_reasoning :314-330 and set its timeout to 300)
- Modify: `app/monitors/heartbeat_loop.py` (`_CHECK_DISPATCH['command']` :981 and
  `_execute_command` :1031-1037 out; `app/api/monitors.py:31-32` drop 'command')
- Modify: `tests/conftest.py:70-73` (EXEC_QUEUE_DIR monkeypatch out)
- Modify: `frontend/src/lib/api.ts` (:516-522 custom tools), `frontend/src/lib/types.ts`
  (:229, :558 CustomToolInfo): a dead call to a 404 is a lie in the UI
- Delete tests: `test_exec_tools.py`, `test_code_exec_sandbox.py`, `test_code_verify_sandbox.py`,
  `test_custom_tools.py`, `test_tool_triggers.py`, `test_generated_tool_network_screen.py`,
  `test_sub_agent_isolation.py`, `test_blast_radius.py`, `test_delegate.py`,
  `test_synthesis_sanitizer.py`, `test_self_improvement.py`, `test_self_improvement_closure.py`,
  `test_explicit_tool_request.py`, `test_math_tool_routing.py` (read first; re-point to
  calculator if calculator-only), `test_eval_residue_2026_08_31.py` (workspace parts)
- Edit tests by name: `test_tool_execute.py`, `test_tool_scoping.py`, `test_tools.py`,
  `test_auth.py`, `test_config.py`, `test_exam_alignment.py`, `test_security.py`,
  `test_tool_call_parsing.py`, `test_reasoning_detection.py` (import from reasoning_cues),
  `test_brain.py`, `test_curiosity_yield.py`, `test_sweep_2026_08_31.py`,
  `test_loop_thread_writes_2026_08_31.py`, `test_brain_chat_fixes.py`,
  `test_synthetic_traffic_gates_2026_08_30.py`, `test_silent_writes.py` (accepted-swallow
  list shrinks), `test_delete_list_2026_09.py`
- Docs: `CLAUDE.md` Key Files rows (agent_loop, agent_workspace, auto_tools, custom_tools
  mentions), "Code Execution Sandbox" section :1571-1593, Access Tiers section;
  `README.md` tool table rows (code_exec, shell_exec, delegate, tool_create)

**Interfaces:**
- Produces: `app.core.reasoning_cues._is_math_logic_word_problem(q) -> bool`,
  `_is_hard_reasoning_query(q) -> bool`, `_is_deliberation_query(q) -> bool`; same
  signatures as before; brain.py:263 becomes a module-level import (it was unguarded).

- [ ] Step 1: Write `tests/test_cut_2026_10_10_agent_layer.py`: GONE_MODULES (the 12
      files), GONE_FLAGS (ENABLE_SHELL_EXEC, CODE_EXEC_TIMEOUT, SHELL_EXEC_TIMEOUT,
      ENABLE_CUSTOM_TOOLS, ENABLE_AUTONOMOUS_TOOL_CREATION, AUTO_TOOL_CREATION_THRESHOLD,
      MAX_CUSTOM_TOOLS, MAX_CUSTOM_TOOL_CODE_LENGTH, ENABLE_DELEGATION, MAX_DELEGATION_DEPTH,
      ENABLE_DELIBERATION), compose carries no exec-runner/exec_queue/ENABLE_SHELL_EXEC,
      no tool list names code_exec/shell_exec/code_verify/tool_create/delegate, main wires
      none of it, `reasoning_cues` exports the three predicates, `brain` imports them from
      there, `_CHECK_DISPATCH` has no 'command'.
- [ ] Step 2: Run it: `... pytest tests/test_cut_2026_10_10_agent_layer.py -q` then FAIL.
- [ ] Step 3: Create `app/core/reasoning_cues.py` by moving the three predicates and their
      regexes out of agent_loop.py (keep docstrings); update the four brain imports.
- [ ] Step 4: Apply the deletions and edits above with a scripted, anchor-asserting edit
      pass (precedent: the round-1 cut script); run `python scripts/sanity_scan.py`.
- [ ] Step 5: `UPDATE_SCHEMA_SNAPSHOT=1 pytest tests/test_invariants.py -k snapshot`.
- [ ] Step 6: Run the guard test + `tests/test_invariants.py` + `tests/test_pathway_coverage.py`
      + every edited test file to green; then the full suite; fix or delete
      by the rule "subject gone: delete; name only: edit".
- [ ] Step 7: Commit: `Cut the agent tool layer: exec sidecar, custom/auto tools, agent loop`.

### Task K2: Cut skills, the Lesson Quiz and the reflexion store

**Files:**
- Delete: `app/core/skills.py`, `app/core/skill_loader.py`, `app/core/auto_skills.py`,
  `app/core/skill_export.py`, `scripts/skill_export.py`, `scripts/migrate_skills.py`,
  `scripts/migrate_skills_v2.py`, `scripts/verify_skills_v2.py`
- Keep `app/core/reflexion.py` but reduce it to `assess_quality` (:119-135, pure),
  `should_use_llm_critique` (:434), `critique_response` (:451, the chat judge) and
  `_CRITIQUE_PROMPT` (prompt_optimizer baseline reads it); delete `ReflexionStore`
  (:611-1148), `check_recurring_failures` (:1150), `sweep_recurring_failures` (:1258)
- Create: `tests/test_cut_2026_10_10_self_improvement.py` (guard)
- Modify: `app/main.py` (:34,:150-160,:300-308,:316,:324 skills; :169-171,:319,:367-373
  reflexion store)
- Modify: `app/core/brain.py` (Services.skills/external_skills :456,:465,
  Services.reflexions :459; skills :903-905,:921-960,:1369-1376,:1410,:3696-3724,
  :3783-3805,:4327,:4442-4470,:3845 capability-gap check uses matched_skill; reflexion
  store :912-917,:1169-1174,:1212-1216,:1407,:1509,:3746-3780,:3812-3828,:4692-4697,
  :2773,:4661; keep `critique_response`/`assess_quality` calls in `_refine_response`)
- Modify: `app/core/prompt.py` (:318,:424-430,:442,:546-556,:827-836 skills blocks;
  :171,:191,:265,:426,:524-531 reflexion block 5d), `app/core/planning.py:165-174`
- Modify: `app/core/calibration.py:97-108` (return defaults; the reflexion read goes)
- Modify: `app/tools/context_detail.py:25-31,:68-69,:218-233` (reflexion category out)
- Modify: `app/core/dream.py` (:128-129,:229-237,:292-293,:458-468 skills; :118-121,
  :169-190,:252,:288-289,:327-328,:393-424,:532-605,:1023-1026,:1052,:1115,:1144,
  :1158-1170 reflexions incl. the dead distill_principles call)
- Modify: `app/monitors/maintenance.py` (:189-199,:656-668,:762-774 skills; :223-245,
  :670-679 reflexions; :777-791 Loop C quiz count)
- Modify: `app/monitors/heartbeat_loop.py` (`_CHECK_DISPATCH` 'quiz' :984, 'skill_quality'
  :993; `_execute_quiz` :1482-1729; `_UNQUIZZABLE_ANSWER_RE` ~:130-160; :856-857 override)
- Modify: `app/monitors/health_checks.py:308-354` (skill quality handler), :165-166 counts
- Modify: `app/monitors/monitor_store.py` (names :62,:69; types :83,:93; `_CORE_ENABLED`
  :516,:546; seeds :682-689 quiz, :775-776 skill quality; keep the migration comments)
- Add migration 39 in `app/database.py`: delete monitors named 'Lesson Quiz' and
  'Skill Quality Monitor' and their monitor_results (dump to
  `/data/backups/cut_monitors_2026-10-10.json` first, precedent 7b8d97c)
- Modify: `app/database.py` (CREATE TABLE skills :73-86 and migrations :537-572,
  :1139-1160; CREATE TABLE reflexions :120-130,:265-266 and migration 20 :842-857;
  keep migration 5 lessons.last_quizzed_at/quiz_failures since lessons stays; regenerate
  the snapshot)
- Modify: `app/monitors/pathways.py` (:149-152 skills + reflexions rows out; add both to
  the exclusion comment), `tests/test_pathway_coverage.py`
- Modify: `app/monitors/eval_harness.py` (:53 seed_skill field; :711-741,:778-810,
  :844-870 seeding/cleanup; :230-240 skill assertions; :146-148,:383-389 category
  metrics; :653-655,:1668-1680 margin tripwire; :1697-1698 cfg snapshot; :1231-1254
  failure-reflexion write; keep :926 assess_quality), `evals/suite.yaml` (skill-match
  :191-219 and semantic-match :221-247 out; header counts)
- Modify: `app/core/prompt_optimizer.py` (:52 skill_extraction_prompt module; :774,:776
  drift rules; keep critique_prompt), `app/core/prompt_optimizer_baselines.py`
- Modify: `app/core/learning.py` (:987-993,:1008-1013,:1062,:1078 skills in summary;
  :1001-1006,:1047-1050 reflexions), `app/monitors/proactive.py` (:145-173,:195-208
  skills; :155-160,:184-192 reflexions/quiz), `app/core/backup.py:45-46` KEY_TABLES,
  `app/core/dedup_metrics.py:14,:38`, `app/api/monitors.py:32`
- Modify: `app/api/learning.py` (:155-218 skill routes; :545-564 reflexions route),
  `app/api/exports.py:61-67,:106-117`, `app/core/data_export.py:289,:301,:359-365`,
  `app/api/system.py` (:103,:110,:136,:140,:241,:300,:730,:786-807,:804,:836,:839,
  :924-939,:962-978), `app/api/chat.py:131,:148,:192` + `app/schema.py:39,:181`
  reflexions_used
- Modify: `app/config.py` (MAX_SKILLS_CHECK, ENABLE_AUTO_SKILL_CREATION,
  ENABLE_SEMANTIC_SKILL_MATCHING, SKILL_SEMANTIC_THRESHOLD, SKILL_EMA_ALPHA,
  SKILL_STALE_DAYS, SKILL_SUCCESS_QUALITY, SKILLS_DIR, REQUIRE_SIGNED_SKILLS;
  MAX_REFLEXIONS, MAX_REFLEXIONS_IN_PROMPT, MAX_SUCCESS_PATTERNS_IN_PROMPT,
  REFLEXION_DECAY_DAYS/AMOUNT, REFLEXION_FAILURE_THRESHOLD/SUCCESS_THRESHOLD), `.env.example`
- Modify: `frontend/src/lib/api.ts` (:223-233 skills, :351 reflexions),
  `frontend/src/lib/types.ts:228`, `frontend/src/cosmos/SystemsPanel.tsx:20`
- Delete tests: `test_auto_skills.py`, `test_semantic_skills.py`, `test_procedure_skills.py`,
  `test_skill_dup_judge_2026_08_30.py`, `test_skill_export.py`, `test_skill_induction.py`,
  `test_skills_hardening.py`, `test_reflexion.py`, `test_reflexion_failure_sweep.py`,
  `test_promotion_taproot_2026_08_31.py`, `test_quiz_false_failure_engine_2026_08_30.py`,
  `test_quiz_gate.py`, `test_lesson_counter_invariant_2026_08_29.py` (read first: lessons
  counters may stay), `test_forecast_skill.py` (read first: it is the forecast "skill"
  score; likely KEEP)
- Edit tests: `test_eval_harness.py`, `test_learning.py`, `test_core.py`,
  `test_system_health_monitors.py`, `test_audit_fixes_20260817.py`, `test_learning_api.py`,
  `test_monitor_format.py`, `test_eval_checks.py`, `test_prompt_optimizer.py`,
  `test_calibration.py`, `test_correction_e2e.py`, `test_dream_two_phase.py`,
  `test_context_detail.py`, `test_heartbeat.py`, `test_chat_api.py`, `test_proactive.py`,
  `test_embedding_collections.py`, `test_scheduler_batching.py`, `test_delete_list_2026_09.py`,
  `test_eval_pair_timeout_cleanup.py`, `test_cut_2026_10_09.py` (skills tool-name list)
- Docs: CLAUDE.md Key Files rows, "Skill Signing" :1595-1600, "Self-Improvement Pipeline"
  :297-304, "Reflexion" mentions, monitor catalog line; README tool/learning rows

- [ ] Step 1: Guard test (GONE_MODULES, GONE_FLAGS, no 'quiz'/'skill_quality' in dispatch
      or seeds, `reflexion.assess_quality` and `critique_response` still importable,
      `ReflexionStore` gone, Services has no `skills`/`reflexions`/`external_skills`).
- [ ] Step 2: Run, FAIL.
- [ ] Step 3: Scripted edit pass + migration 39 + snapshot regen + sanity_scan.
- [ ] Step 4: Full suite; delete/edit tests by the rule.
- [ ] Step 5: Commit: `Cut skills, the Lesson Quiz and the reflexion store`.

### Task K3: Deploy both cuts and verify

- [ ] Build; full suite on the built image; open a 2 h quiet window; wait for in-flight
      digest-class results (count dispatches, not a fixed number: the 10-09 poll killed
      three digests); `docker compose up -d --remove-orphans nova` (nova-exec must go);
      confirm `docker ps` has no nova-exec, freshness 0 stale, Telegram/Discord up,
      27B placement GPU, `/api/status` pathways none dead; close the window.
- [ ] After 24 h: `scripts/throughput_panel.py` runs/active-hour vs the prior day; the
      Quality Eval Harness run on the new image (tool-use 6/6, reasoning unchanged,
      long-memory per task: the afdb491 history change gets its first reading here).
- [ ] Record in memory + plan status.

---

### Task G1: Stop extracting every digest twice

**Files:** `app/monitors/heartbeat_loop.py:801-818` (post-digest extraction),
`app/monitors/deep_research.py:2881-2956` (`_learn_facts`), new
`tests/test_digest_extracted_once.py`

Measured: a Domain Study digest is banked by `_learn_facts` (source='researched',
provenance 'deep_research:<topic>') and again by the heartbeat pass (source='extracted',
provenance=monitor.name) on the same text; kg.py:1038-1046 treats the second as
corroboration (trust +0.2, quarantine released). Cost: one 27B extraction call per
digest and a false corroboration signal.

- [ ] Step 1: Test: a query monitor whose runner already banked facts does not get the
      heartbeat extraction; a think-path query monitor still does.
- [ ] Step 2: Run, FAIL (both extracted today).
- [ ] Step 3: `_learn_facts` records the banked count on a module-level per-run registry
      keyed by monitor name (`deep_research.banked_by_run`); `_check_monitor` skips the
      post-digest pass when the runner banked > 0 for this run.
- [ ] Step 4: Green; commit `Bank a digest's facts once, not twice`.
- [ ] Measurement after a day: kg_facts added per digest (expect roughly half), and the
      corroboration-promotion count from the `[kg]` log line.

### Task G2: World time: pass the publication date as valid_from

**Files:** `app/monitors/deep_research.py` (`_PUB_DATES` :343, `_record_pub_date`
:357-366, `_learn_facts` :2881-2956, `_FACT_PROMPT` :537-551), `app/core/kg.py:952-995`
(`add_fact` already takes valid_from), new `tests/test_facts_carry_world_time.py`

- [ ] Step 1: Test: `_learn_facts` given findings whose articles have recorded pub
      dates passes `valid_from=<earliest pub date among the cited articles>` to
      add_fact; with no date it passes the digest date (`today`), never None.
- [ ] Step 2: Run, FAIL.
- [ ] Step 3: Implement; the triple prompt gets one line "AS OF: <date>" so the model
      reads the evidence as of that date (no tense instruction: the schema is tenseless).
- [ ] Step 4: Green; commit. Rule G2: after one day, share of new researched facts with
      valid_from differing from created_at is at least 50%.

### Task G3: Refuse a revert that the evidence does not date past the supersession

**Files:** `app/core/kg.py:1100-1127` (revival branch), new
`tests/test_kg_revert_guard.py`

Rule from arXiv 2610.07715: a retired value re-read from an older source must not
reactivate. With G2 in place, a revival request carries the evidence's valid_from.

- [ ] Step 1: Tests: (a) A superseded by B (B live, valid_from D_B); add_fact(A) with
      valid_from at or before D_B returns False, A stays retired, log `[kg] revert refused`;
      (b) with valid_from after D_B it revives A and supersedes B (today's behaviour);
      (c) no superseding row (A retired by prune/judge only) revives as today;
      (d) a multi-valued predicate is unaffected.
- [ ] Step 2: Run, FAIL. Step 3: implement in `_sync_add_fact` before the UPDATE at
      :1118. Step 4: green; commit.
- [ ] Measurement: `[kg] revert refused` count per day; the live A-B-A count
      (37 today, SQL in the map) must stop growing.

### Task P1: Persist gate tuples and build the planting probe

**Files:** `app/monitors/deep_research.py` (`_check_cascade` :3419-3488; call sites
:3490, :3540, :3578), new `app/monitors/gate_record.py` (append-only JSONL writer at
`/data/entail/gate-YYYY-MM-DD.jsonl`, fields: ts, label, site, claim, hosts, doc, prob,
supported, narrow), new `scripts/planting_probe.py`, new
`tests/test_gate_record.py`, `tests/test_planting_probe.py`

- [ ] Step 1: Tests: every scored pair at every site is appended with the full doc
      window and verdict; the writer never raises into the gate (swallow + log);
      records rotate daily; `planting_probe.sample(records)` picks N rescued
      (site clause/alt, supported) and N first-pass (site gate, supported) claims;
      `planting_probe.plant(claim, uncited_doc)` inserts the claim's key phrase into a
      passage from a NON-cited article and the probe reports whether the verdict
      migrates (scored via the sidecar).
- [ ] Step 2: Run, FAIL. Step 3: implement. Step 4: green; commit
      `Record every entailment verdict with its evidence window; add the planting probe`.
- [ ] Run: after at least 200 rescued records exist (341 anchored per 31 h today),
      `scripts/planting_probe.py --n 200` inside a quiet window (CPU sidecar only).
      Rule P1: rescued minus first-pass > 5 pp disables clause/alt rescue (`_CLAUSE_TOTAL=0`,
      `_ALT_TOTAL=0`) pending redesign; else record the number in the plan.

### Task R1: Make engine suspensions visible, and refusals directive on the tool path

**Files:** `app/tools/native_search.py` (`_search_searxng` :464-533 reads
`unresponsive_engines`; `search()` :622-728 log line gains `unresponsive=[...]`;
`search_health` :38-47 gains per-engine counters `engine_health()`),
`app/tools/web_search.py:79-88` (zero results: output
`SEARCH REFUSED: every engine returned nothing for '<q>'. Do not infer an answer; say
the search found nothing.`), `app/monitors/engineering_report.py` (field
`search_engines`: per-engine non-empty rate + top unresponsive), new
`tests/test_search_visibility.py`

- [ ] Step 1: Tests: SearXNG JSON with `unresponsive_engines` is logged and counted;
      `engine_health()` returns {engine: {calls, nonempty, unresponsive}}; web_search
      zero result output starts with `SEARCH REFUSED`; the report field renders.
- [ ] Step 2: FAIL, implement, green, commit.
- [ ] Rule R1: curiosity think-path answers that cite nothing after a refused search
      stay 0; the engineering report names the suspended engines daily.

### Task R2: Repair the SearXNG engine list

**Files:** `searxng/settings.yml.example` and `searxng/settings.yml` (drop ecosia,
presearch, reddit which log "Cannot load engine" at every start, and wikidata; keep the
brave HTML scraper as is; set `search.default_lang: en-US`; enable `duckduckgo` for
the trial), `tests/test_searxng_settings.py`

- [ ] Step 1: Test: the two files differ only on secret_key; no engine listed twice;
      the engines that fail to load are absent.
- [ ] Step 2: edit; `docker compose restart searxng`; commit.
- [ ] Rule R2 (48 h, from R1's counters): any engine from 0% to >50% non-empty, none
      regresses; otherwise revert the DDG line.

### Task X1: Distinct readings before analysis (digest chain)

**Files:** `app/monitors/deep_research.py` (:4467-4481 seam; `_figure_support`
:2602-2656; `_independence_clusters` :2472-2536; `_deep_analyze` prompt :4271-4283),
new `tests/test_distinct_readings.py`, `scripts/ceiling_ab.py` (arm flag
`--set DISTINCT_READINGS=off` for the baseline)

- [ ] Step 1: Test: given findings from 3 articles of which 2 are one syndication
      cluster, `distinct_readings(findings, clusters)` lists ONE line per independent
      cluster per figure (`host · figure · date`), copies collapse, and the block is
      prepended to the analysis prompt with the instruction "reason from the readings,
      not from the count of articles".
- [ ] Step 2: FAIL, implement, green, commit `List distinct readings before reasoning`.
- [ ] Measurement (quiet window, GPU): `scripts/ceiling_ab.py --replay` on the 16 frozen
      topics, arm on vs off, same evidence. Rule X1: fact support within 0.02 of the
      baseline AND cited-copy share (same-cluster citations / total) falls. Refuse
      otherwise and keep the function unwired.

### Task X2: Distinct readings in the forecast mint context

**Files:** `app/core/forecasts.py:401-425` (ctx assembly), `scripts/forecast_backtest.py`
(arm `--context distinct`), test in `tests/test_forecast_context_readings.py`

- [ ] Build the arm; run the 144-claim replay with k=8 (noise floor). Rule X2: Brier
      improves by at least 0.02; else refuse and record.

### Task C1: Resolve dossier tensions from Nova's own spans

**Files:** new `app/core/tensions.py`, new table `tension_resolutions` (migration 40:
id, tension_text, dkey_a, dkey_b, figure_a, figure_b, span_a, span_b, verdict
(a|b|both|unresolvable), reasoning, judge_model, verifier_model, verifier_agrees,
created_at), `app/core/dossiers.py` (:1016-1058 call the resolver; write the
resolution into the losing dossier's Key facts bullet and a `belief_revisions` row via
`questions.record_revisions`; the tension line in the consolidation result becomes
"Tension resolved: ..." or "Tension open: ..."), `app/monitors/pathways.py` (row
`tension_resolutions`), `app/api/dossiers.py` (GET /dossiers/tensions), new
`tests/test_tensions.py`

Design (MemDream/MemCodex applied): the resolver NEVER searches the web. For each
tension from `_numeric_tensions`: load the last 3 stored digests of each domain
(`_domain_sources` :577-611 already does this), locate the sentence(s) carrying each
figure (exact figure match, then the `_fmt_qty` form), build a span pack
`[domain · digest date · sentence]` per side, and ask the 27B (judge temp 0) with ONLY
those spans + today: are the two figures the same quantity? if yes which is current
(by date and wording); answer JSON {verdict, reasoning, winner_date}. Then a second,
independent judge (gemma4:e4b) sees the same spans and the verdict and answers
agree/disagree. Admit the resolution only when the verifier agrees; otherwise leave
the tension open with both judges' reasons stored.

- [ ] Step 1: Tests: span location finds the sentence for '5.501%' and for '2 billion';
      a tension whose figures are different quantities (10-year vs 30-year yield)
      resolves `both`; same quantity with dated spans resolves to the later one;
      verifier disagreement leaves it `open`; the losing dossier bullet is rewritten
      and a belief_revisions row is written; the consolidation result line changes.
- [ ] Step 2: FAIL, implement, green, commit `Resolve a tension from Nova's own
      records, admitted only when a second judge agrees`.
- [ ] Measurement: replay the 31 stored tension lines (monitor_results of Knowledge
      Consolidation, 2026-09-03..10-09) through the resolver in a quiet window. Rule
      C1: at least 50% closed; of the closed, a third independent pass (same spans,
      different prompt wording) disagrees on at most 10%.

### Task C2: Answerability triage for the curiosity backlog

**Files:** `app/monitors/heartbeat_loop.py` (`_research_one_curiosity` :1879 before
the path split :1918), `app/core/curiosity.py` (columns `answerable`, `evidence_ok`,
`probe_at` via migration 41), new `tests/test_curiosity_triage.py`,
`app/monitors/engineering_report.py` (field `curiosity_triage`)

- [ ] Step 1: Tests: two yes/no probes on the fast model ("Is this question settleable
      from public sources available now?", "Would a web search plausibly surface a
      dated source for it?") are recorded on the item before research; a `no/no`
      item is deferred with reason `not_answerable` and its attempt is not burned;
      the pick order prefers `yes/yes` over aging urgency.
- [ ] Step 2: FAIL, implement, green, commit.
- [ ] Rule C2 (after 40 attempts): AUROC of the probes against resolve/fail > 0.85
      (`scripts/curiosity_triage_auroc.py`); else the probes stay as instruments and
      the pick order reverts.

### Task Z: Close out

- [ ] Plan status entry per task with the measured number and the rule outcome.
- [ ] Memory: one file for this program with every rule result.
- [ ] CLAUDE.md: the removed-2026-10-10 section, the world-time note on the KG, the
      gate record, the tension resolver.

## Status 2026-10-11 (execution log)

| Task | State | Commit | Measured / rule |
|------|-------|--------|-----------------|
| K1 agent tool layer | shipped to tree | f17c2c6 | guard 31 tests; full suite 3510/0 after the access-tier gate tables went too |
| K2 skills, Lesson Quiz, reflexion store | shipped to tree | 878c7dc | guard 46 tests; full suite 3226/0; migration 39 dumps + deletes the three retired monitors |
| K3 deploy | DONE 2026-10-11 05:12 UTC (nova-app:k2 = 791f530b5933; nova-exec removed; freshness 118/0; 27B 100% GPU; migration 39 dumped 2 monitors + 147 results; no startup errors). **24 h readings 2026-10-12:** throughput_panel 10-11 = 105 runs, 5.2 runs/hr (10-10: 6.5) — confounded by the deploy drain and the 14.5 h X1 replay; 10-12 so far 5.7. Quality Eval Harness on the new image (ran 10-11 22:42 UTC, DURING the replay): 45/50 = 91.8% (10-10: 51/56 = 91.1%); tool-use 5/6, reasoning 6/8 with 1 timeout (GPU contention), **long-memory 4/5 (was 2/5) — the afdb491 KG-history change's first reading: multi-session and multi-valued recall now pass; temporal-previous still fails**; flagged regressions reasoning −0.14 / tool-use −0.17 need the clean daily run on 10-12 to confirm or clear | 791f530b5933 | clean harness read due 2026-10-12 ~22:42 UTC |
| G1 bank once | shipped; **read 2026-10-12 05:51 UTC (24 h)**: facts per Domain Study digest 7.51 (day before, 73 digests, 548 facts, 269 source='extracted') → 4.25 (36 digests, 153 facts, 11 'extracted'); 39 'skipping the second extraction' log lines; the extracted-source stream is gone as designed. Digest count halved that day because the 14.5 h X1 replay held the GPU | 85d9bad | expectation (~half) met |
| G2 world time | shipped; **rule G2 read 2026-10-12 05:51 UTC: PASS** — 147 of 147 researched facts since deploy carry valid_from ≠ created_at (100% vs the ≥50% bar) | 4621ad2 | PASS |
| G3 revert guard | shipped; first 24 h: 0 `[kg] revert refused` lines (no revert attempted yet); re-count the A-B-A chains with the map SQL in a week | 59e8c53 | pending (no events yet) |
| P1 gate record + planting probe | shipped; recorder live (9,652 verdicts in the first 24 h, ~45 MB/day → retention cut to 7 days); **rule P1 read 2026-10-12 06:54 UTC: rescue NOT disabled** — 200 rescued vs 200 first-pass claims planted into uncited passages: migration 76.0% vs 85.5%, delta −9.5 pp (bar: > +5 pp disables). Caveat: the probe plants the whole claim sentence, so the sidecar sees it verbatim and both arms sit near the ceiling; a key-phrase-only planting would be the sharper instrument (follow-up, not built) | 32c8d98, cfcabf9 | rule P1: keep rescue |
| R1 search visibility | shipped | dd1635f | report field `search_engines`; refusals directive |
| R2 engine list | shipped + searxng restarted | d013852, 7815c29 | first live call after restart: bing + yandex answered; brave/google/mojeek suspended; startpage + duckduckgo CAPTCHA → rule R2 at 48 h likely reverts DDG |
| X1 distinct readings | function shipped, flag stays OFF; **rule X1 READ 2026-10-11 21:41 UTC: REFUSED** — same-evidence replay, 16 frozen topics, 27B, judge gemma4:e4b: support 0.672 off → 0.691 on (within 0.02, met), but cited-copy share 0.000 → 0.010 (did NOT fall: 0/258 vs 3/300 same-cluster citations — the baseline has no copies to remove). Side readings: overall 0.827 → 0.842, race 4.47 → 4.53, core coverage 0.724 → 0.679. The premise (copies inflate the count the model reasons from) does not show in the citations; the block buys judge score at a coverage cost. Not wired | e6f2e31 | rule X1 not met (copy share must fall) |
| X2 forecast readings arm | not started | — | depends on X1's verdict; k=8 replay, rule X2 Brier ≥ 0.02 |
| C1 tension resolver | shipped; **rule C1 READ 2026-10-11 07:05 UTC: FAIL as measured** — 56 stored tension lines replayed against a DB copy (judge qwen3.8:27b, verifier gemma4:e4b): 9 closed (16%, all 'both' = different quantities), 30 'both' total of which the verifier refused 21 (mostly one side without spans), 26 unresolvable, 30 with NO span on either side (digests aged past the 3-per-domain window), 0 a/b verdicts; third-pass disagreement 0/9. Resolver stays (admission is conservative, no harm) with a no-span short-circuit; the real test is NEW tensions at consolidation where both digests are still stored | 8659230 + follow-up | ≥50% closed NOT met (16%); ≤10% disagreement met (0%) |
| C2 curiosity triage | shipped; **first 24 h: 15 of 15 probed items deferred** — measured on the 27B, the original wording answers NO to the Fed's September decision and to Bitcoin's price (the model will not vouch for post-training documents). Probes replaced 2026-10-12 (d75a374): 'would public reporting normally cover this' + 'specific enough for one dated source'; validated on 9 questions (yes for Fed/Bitcoin/sanctions, no for Deno share/DeFi status/unicorns). Redeployed 2026-10-12 07:00 UTC (nova-app:k3 = 20f939f2c918, same quiet-window recipe, freshness 118/0, no startup errors); the 15 deferred items were returned to pending. The new wording still deferred the next 3 (open-ended 'current status of…' questions), and a deferred item never yields an outcome — so deferral is now behind ENABLE_CURIOSITY_TRIAGE_DEFER (default off): probes recorded, yes/yes ranked first, everything researched until rule C2 is read (nova-app:k4, 2026-10-12 ~07:30 UTC). Rule C2 clock restarts | 423c57d, d75a374 | AUROC after 40 probed attempts |
| Z close-out | CLAUDE.md program section written (4521b6d); memory + plan numbers after the 24 h readings | — | — |

Deviation recorded: C1's `tension_resolutions` pathway row was NOT added — the
writer is conditional on a tension existing, the same shape as the
`capability_gaps` false-dead case the pathways file warns about.

## Follow-up program 2026-10-12 (owner: "do all 4 in order")

| Item | State | Commit | Measured / rule |
|------|-------|--------|-----------------|
| 1 Search coverage via feeds | shipped | 5b731cd | Probed all 232 curated feeds from inside nova-app: 6 dead + 12 blocked (403) replaced with outlets that answered, Barron's dropped, 7 in-domain duplicates removed; Macroeconomics + Microeconomics (search-only, 2 sources/digest) given 12 + 7 feeds. New overview log line carries distinct-host count; report field `sources_per_digest`. Baseline 10-09..10-12: 18.8 sources/digest, 5.8 distinct hosts (first six), 8 digests ≤3 sources in 3 days. Read after 48 h: median hosts and the thin list |
| 2 Curiosity question quality | shipped | ad51712, 5595843 | 40 failed vs 20 resolved dossier questions: failures ask for the unpublished (undisclosed conditions, specific mechanisms, patch rollout, yield). Prompts now demand a PUBLIC settleable object; `_is_unpublished_question` skips those shapes at minting (logged). Rule: dossier_open_question resolve share over the next 40 attempts > the 33% lifetime (20/60) |
| 3 Ollama runtime bake-off | **READ 2026-10-12 17:30 UTC: REFUSED** — same model store, same entrypoint, 27B on the 3090. Baseline 0.32.15: 0 leaks, JSON 15/15, 52.5 tok/s, 113 s cold load, 100% GPU. Candidate 0.35.1: correctness identical (0 leaks, thinking field empty, JSON 15/15, GPU discovery OK) but 2.8 tok/s cold and 1.5 tok/s WARM with the card fully allocated (24.3 GB of 24.6, 1% utilization): the 0.35 runner adds an 8 GB prompt cache (`cache state … limits: 8192 MiB`) on top of the 24K KV, overflowing the card and the owner's 3 GB reserve. Stay on 0.32.15. Possible second trial, not run: 0.34.x with the prompt cache capped. Gotcha: an orphaned killed script removed the first candidate mid-probe; run probes as detached containers | — | throughput −95%: refuse |
| 4 Gate drop rate | measured; rule staged | 6a1eec1, e19db9b | `scripts/gate_miss_classes.py` over 3,557 primary pairs: 80% rejected at the primary pass (54% after rescue); 34% low overlap, 25% analytical, 15% number absent, 7% possible false negatives, 3% borderline; evidence quality equal across accepted/rejected; median rejection p=0.03 → the gate is right, the synthesis mis-attributes. Citation-discipline rule staged behind ENABLE_CITATION_DISCIPLINE (off). Rule: same-evidence replay, primary rejection share (baseline 0.798 on the x1off window) down ≥10 pp with support within 0.02 and coverage within 0.03 |

Items 1, 2 and 4 deployed 2026-10-12 ~18:55 UTC as nova-app:k5 (full suite 3302/0; quiet-window recipe; freshness 118/0; 27B 100% GPU). The citation-discipline replay arm (`cite_on`, flag on, same frozen evidence as `x1off`) was launched inside the new container right after; its rule is read with `/data/ceiling/cite_ab.py cite_on <start> <end>`.
