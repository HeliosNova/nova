# Eval receipts

This directory holds the **measured evidence** behind the README's claims — dated,
reproducible eval results. The rule: if a number is claimed in the README, the run
that produced it lives here (or in `tests/live_audit_results.json` for the
94-check live audit). No receipts, no claim.

## What belongs here

- `eval_baseline_YYYY-MM-DD.json` — output of the eval harness over
  `evals/suite.yaml` (59 tasks, 10 categories), written by
  `app/monitors/eval_harness.py`. Includes per-category pass rates,
  `memory_causal_fix_rate`, `kg_causal_fix_rate`, `reflexion_mean`.
- `live_audit_YYYY-MM-DD.json` — full-system 94-check audit
  (`tests/live_audit.py`) run against a live stack.

## Methodology requirements

- **Judge**: cross-family model (never the model under test), position-swapped.
  Self-judged reports are not valid receipts — one was committed here once and
  produced degenerate scores; it has been archived.
- **Timeouts are not failures**: a response exceeding the time budget is recorded
  as `timeout`, not `incorrect`. Correctness is computed over completed runs;
  latency is tracked as its own metric.

## How to regenerate

```bash
# Harness suite (runs nightly via the Eval monitor, or on demand):
docker exec nova-app python -m app.monitors.eval_harness --suite evals/suite.yaml

# Full live audit (stack must be up):
docker exec nova-app python tests/live_audit.py
```

## History

- `2026-03-18` — live audit 4.5/10 (42/94): the honest "before" snapshot.
  Major failure clusters: multi-part queries, KG-in-chat retrieval, identity,
  safety refusals. (Methodology note: timeouts were graded as wrong answers.)
- `2026-04-24` — harness suite pass rate 97.2% (35/36) after multi-agent and
  retrieval fixes (see CHANGELOG v1.5.1).
- `2026-06-08` — memory-learning causal-fix 1.0, kg-retrieval causal-fix 0.83
  after MIN_RRF_SCORE / LIMIT-500 / RRF-discard fixes.
- `2026-06-10` — live audit **9.3/10 (62/67 completed checks, grade A)** on the
  production build (`live_audit_2026-06-10.json`), first run under the
  timeout-separated methodology: 9 timeouts excluded and listed (latency mean
  87s / p95 122s — measured while a full pytest suite ran on the same machine,
  so latency is worst-case). Remaining real failures kept visible: one
  calculator arithmetic miss, two memory-recall misses tracing to the
  deliberate v1.5.0 removal of auto fact extraction, one 500 on the
  system-prompt-leak probe.
