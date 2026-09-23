# Next level — research and program (2026-09-22)

Owner: "cut 1-4, reset forecast from 0, add it to your focus order and do the
research on what we need to move to the next level."

## Where the record stands (measured today)

| area | evidence |
|---|---|
| digests | 5,557 chars, 8.3 sources, 0 self-citations, 0 leaks; 97 % of banked facts survive; ~40/day at 6 runs per active hour |
| delivery | whole briefings since today; every alert-path rewrite and the bundler cap on knowing-tier writers removed |
| knowing tier | 97 dossiers, 85 storylines, ledger reconciling; priming measured as a grounding cost and retired |
| curiosity | lifetime: dossier questions 17 resolved / 22 failed, agent failures 5 / 0, every other source 0 — those sources are cut |
| forecasting | 94 hit / 50 miss with **no skill over the base rate** (skill −0.06), 48 unresolvable, 17 restated; record reset to zero today |
| GPU | 32 % idle across a 2 h sample of three-wide digests (CPU entailment is the wait) |

## What the field did since June (sources at the end)

1. **OpenForecaster (Dec 2025 → 2026).** An 8B model (Qwen3-8B base) trained
   with GRPO on a hybrid accuracy + Brier reward over ~52k questions generated
   from news, matches 100B+ models on Brier. Two findings transfer directly:
   - **The validator is the lever.** Of 745k candidate questions only ~7 %
     survived a validator that demanded future-facing framing, an explicit
     resolution criterion (source of truth + answer format), a deadline, and
     no answer leakage. Nova mints a free-text claim + date and validates
     nothing — hence 48 unresolvable and a judge left to guess what settles a
     claim.
   - **Retrieval is bounded to the pre-resolution corpus** so the forecaster
     cannot look the answer up. Nova's resolver already drops post-creation
     evidence for grading; the estimator replay is claim-only (handicapped).
2. **Calibration.** RL with a Brier reward (RLCR) is training-time; the
   inference-time result that transfers is **source-summary averaging**:
   forecasting from several evidence summaries and averaging cut ECE 0.142 →
   0.120. Nova's ensemble already takes k=3 samples of the same context; vary
   the evidence view per sample instead.
3. **HINDCAST (Jul 2026).** Leak-free evaluation of LLM forecasters: grade as
   of a past date against a frozen archive and against the market price at
   that date. Nova's backtest is the same shape; keep it the only yardstick
   and never quote a number pooled across regimes.
4. **Verifiers.** No MiniCheck successor; Bespoke pivoted. "Verifying the
   Verifiers" found ~16 % of benchmark labels ambiguous or wrong and that small
   verifiers improve with synthetic multi-hop training data. Nova's 83–87 %
   first-pass rejection on synthesis claims is the expected shape; the
   clause/re-cite/de-cite rescues are the right tools. Nothing to adopt now.
5. **Models.** Qwen3.8-27B (Aug 14) is what Nova runs. **Gemma 4 26B-A4B**
   (MoE, ~4B active) runs ~2× faster on a 3090 at the same VRAM; benchmark
   aggregates put Qwen3.8-27B ahead on quality (45.0 vs 29.6). Only an A/B on
   grounded synthesis decides; the ceiling harness exists.
6. **Agent memory.** Independent LongMemEval: Zep's temporal graph 63.8 %,
   Mem0 49 %. Nova's KG is Zep-shaped (bitemporal since May). New work:
   memory contamination (MemGuard), memory as an action space, typed memory.

## Program, in order

1. **Forecasting from zero** (this session): a mint-time **validator** on the
   resident model — binary, falsifiable, an explicit resolution criterion
   (what source settles it, in what form), a date consistent with the claim,
   not a restated projection, not already knowable — stored as
   `forecasts.criterion` and handed to the resolver's judge; a **reference
   class / base rate** step before the probability; REGIME bump so the new
   record is measured on its own. Yardstick: `skill` on the backtest once the
   new record has ≥ 50 resolutions; unresolvable share (was 5 %).
2. **Forecaster model bake-off** (offline, no residency cost): OpenForecaster-8B
   (pulled: `hf.co/mradermacher/OpenForecaster-8B-GGUF:Q8_0`) vs the 27B
   estimator on the same claims and contexts, Brier on the new record. Adopt
   only if it wins; it would be a third 9B-class residency.
3. **Synthesis model A/B**: Gemma 4 26B-A4B (pulled: `gemma4:26b-a4b-it-q4_K_M`)
   vs Qwen3.8-27B with `scripts/ceiling_ab.py` over the frozen 16 topics
   (support, coverage, fabrication, chars). Needs a quiet window (~20 h
   serial, run the arms concurrently). Adopt only if support holds.
4. **Curiosity**: the two surviving sources are dossier questions
   (evidence-first, 27B) and agent failures (memory path, 5/5). Measure
   resolved/day for a week; the loop is now clean enough to read.
5. **Memory against the field**: a LongMemEval-style subset through the KG +
   dossiers, so "Zep-shaped" becomes a number.

## Status

- Step 1 shipped 2026-09-22 (`5c490d4`): validator + criterion + regime
  `2026-09-22-criterion`; record reset to zero; migration 38.
- Steps 2 and 3 scheduled for a 3-hour quiet window at 02:07 local
  2026-09-23: `scripts/forecast_bakeoff.py` (27B vs OpenForecaster-8B on the
  144 resolved claims of the dumped record) then `scripts/ceiling_ab.py
  --replay --model gemma4:26b-a4b-it-q4_K_M` on 8 frozen topics against the
  `prime_off_n16` baseline arm. Pre-registered rules: adopt Gemma only if fact
  support holds within 0.02 and fabricated stays 0; adopt OpenForecaster only
  if its Brier beats the 27B on ≥60 % of paired claims.
- Cuts landed (`7b8d97c`): the curiosity queue now has two sources; step 4's
  measurement starts from here.
- Step 1 verified on its first live consolidation (2026-09-23 01:46 UTC):
  7 candidates, 1 minted with a stored criterion, 6 refused for stated
  reasons (a restated consensus, a two-outcome bundle, an outcome already
  under way, an undefined term, an action already taken). Acceptance 14 %
  against OpenForecaster's 7 %. Expect roughly one mint per consolidation;
  the record grows slowly by design. A refusal used to trip the parser-drift
  warning; fixed (`1abb198`, `forecasts.REJECTED`).
- Residency, the precondition for every bake-off number meaning anything:
  two more leaks closed the same night — the digest's detached KG extraction
  outliving the class gate (`bbcba3d`) and startup's own KG LLM curation and
  chat-model warmup evicting the resident 27B on every restart (`63cbedb`).
  Cold loads measured from the T7 once the page cache is gone: 9B 4 min,
  27B 7 min; warm 27B 17 s. Each swap is therefore minutes, not seconds, and
  the quiet-window jobs must be the only thing touching the card.
- Deployed `1abb198` at 09:02 UTC 2026-09-23 inside the quiet window;
  startup evicted nothing.
- **Step 2 result — OpenForecaster-8B REJECTED** (2026-09-23 09:17 UTC,
  `scripts/forecast_bakeoff.py`, n=144 resolved claims of the dumped record,
  claim text only, k=3, rows in `/data/ceiling/forecast_bakeoff_2026-09-23.rows.jsonl`):

  | arm | Brier | skill | mean conf | closer (paired) |
  |---|---|---|---|---|
  | base rate (hit rate 0.653) | 0.2267 | 0 | — | — |
  | stated as minted (saw the digest) | 0.2287 | −0.009 | — | — |
  | qwen3.8:27b, claim only | 0.3061 | −0.350 | 0.475 | 77/144 |
  | OpenForecaster-8B Q8, claim only | 0.3285 | −0.450 | 0.373 | 67/144 (47 %) |

  The rule was ≥60 % of paired claims; it reached 47 %. Its low band
  (n=112) said 0.28 and happened 0.63 — it answers "no" on a record that
  resolved 65 % "yes". Caveats apply to both arms equally (self-graded
  outcomes, no digest, JSON-mode answer without its reasoning). The finding
  that matters more: the stated number, written with the digest in context,
  beats both blind re-estimators by 0.08 Brier. Context is the lever, not
  the model — which is what HINDCAST and the evidence-summary averaging
  papers say, and what the validator + criterion step already builds on.
- Harness rule learned on step 3 (2026-09-23 09:59 UTC): a replay with a
  non-config model must also pass `--set MONITOR_SYNTHESIS_MODEL=<model>`.
  `deep_research._verify_lead_claims` reads the config model instead of the
  chain's `syn_model`, so the Gemma arm pulled the 27B onto the card
  mid-topic — a third residency per topic, each a cold load. Harmless in
  production (same model), invisible in the baseline arm (same model), and
  it cost the first Gemma run its second topic.
- **Step 3 result — Gemma 4 26B-A4B REJECTED as synthesis model**
  (2026-09-23 11:35 UTC, `ceiling_ab.py --replay`, 8 frozen topics, same
  evidence in both arms, judge gemma4:e4b, `/data/ceiling/results/gemma4_n8.rows.jsonl`
  vs `prime_off_n16`, exact sign-flip p):

  | metric | Gemma 26B-A4B | Qwen3.8-27B | delta | won | p |
  |---|---|---|---|---|---|
  | fact support | 0.645 | 0.751 | −0.105 | 3/8 | 0.16 |
  | fabricated | 0.000 | 0.000 | 0 | — | — |
  | core coverage | 0.627 | 0.690 | −0.062 | 2/6 | 0.28 |
  | judge (RACE) | 4.53 | 4.63 | −0.09 | 0/3 | 0.25 |
  | overall | 0.828 | 0.873 | −0.045 | 3/8 | 0.09 |
  | chars | 5,363 | 7,859 | −2,496 | 0/8 | 0.008 |

  The rule was support within 0.02 and fabrication 0; fabrication held,
  support did not. Gemma won China tech, economics and finance by a hair
  and lost cybersecurity by 0.14 and AI/ML by 0.11. The one significant
  difference is length: Gemma writes a third less on identical evidence.
  It is 2.3× faster on generation (118.8 vs 51.3 tok/s, clean probe on an
  idle card; prompt 422 vs 136) and loads warm in 8 s — a real throughput
  lever if grounding were equal. It is not, and grounding is the product.
  Keep the tag pulled for a future quantization or prompt retest; do not
  retest without a change that could plausibly move support by 0.1.
- **Step 1 follow-up in flight (2026-09-23 17:2x UTC):** because the
  bake-off said context is the lever, two mint-time context levers were
  built unwired and are being measured leak-free on the same 144 claims
  (`scripts/forecast_context_backtest.py`, `4b7502a`): web evidence dated
  on or before the claim (undated dropped, the HINDCAST rule) and an
  explicit reference-class + base-rate step. Adoption rules are in the
  script's docstring.
- **Step 1 follow-up result — both levers refused, mint unchanged**
  (2026-09-23 18:46 UTC, n=144, 27B, k=3,
  `/data/ceiling/forecast_context_2026-09-23.rows.jsonl`):

  | arm | Brier | closer than claim-only |
  |---|---|---|
  | stated as minted (saw the digest) | 0.2287 | — |
  | claim only | 0.3172 | — |
  | + prior-dated evidence | 0.3217 | 69/144 (48 %) |
  | + reference class / base rate | 0.3190 | 59/144 (41 %) |
  | + both | 0.3220 | 61/144 |

  Evidence: strictly prior-dated pages were findable for **1 claim in
  144** — today's index does not carry dated pre-claim reporting for
  Nova's claims, so the arm is claim-only plus noise and the lever cannot
  be measured leak-free this way. At live mint time the evidence is the
  digest itself, which is already in the samples' context. Reference
  class: the model produced a base rate for every claim (mean 0.47), the
  samples then sat within 0.05 of it, and the base rate alone scores
  0.3363 — the outside view anchors the estimate on an uninformative
  number. Rule was ≥60 % closer; it reached 41 %.

  **Noise floor, measured and worth keeping:** 143 pairs with identical
  context (the evidence arm where no evidence was found, against
  claim-only) differ by a mean |0.115| per claim at k=3, T=0.7. A paired
  count near 72/144 is a coin flip, and Brier differences under ~0.02 are
  unreadable at this n. Any future estimator comparison needs either k≥8
  or a larger record.
- **Step 5 shipped as an instrument (`92b89ed`):** eval category
  `long-memory`, five LongMemEval-shaped tasks (knowledge update, temporal
  recall, multi-session aggregation, multi-valued recall, abstention) on
  fictional entities through the before/seed/after runner. The temporal
  task is expected to fail until chat retrieval reads the KG's history;
  it ships failing so the gap has a number. First live numbers below once
  the image is deployed and the category has run.

## Sources

- OpenForecaster / OpenForesight: https://openforecaster.github.io/scaling-data/ ; paper https://arxiv.org/pdf/2512.25070
- Calibration probing + RLCR: https://arxiv.org/html/2607.08046v1
- HINDCAST: https://arxiv.org/abs/2607.14051
- Verifying the Verifiers: https://arxiv.org/pdf/2506.13342
- Gemma 4 26B-A4B vs Qwen3.8-27B: https://llm-stats.com/models/compare/gemma-4-26b-a4b-it-vs-qwen3.8-27b ; https://ollama.com/library/gemma4/tags
- Zep vs Mem0 on LongMemEval: https://vectorize.io/articles/mem0-vs-zep ; MemGuard https://arxiv.org/pdf/2605.28009 ; Memanto https://arxiv.org/pdf/2604.22085
- Ollama 0.34: https://releasebot.io/updates/ollama
