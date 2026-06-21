# Archived: weight-training stack (2026-06-12)

This directory holds Nova's **retired** fine-tuning / GRPO / RLVR-trainer code
(~21.7k LOC). It is kept for history and possible revival — it is **not** part of
the live product and nothing in `app/` imports it.

## Why it was archived

- **0 successful train→A/B→deploy** cycles across every `run_history.json`.
- The one honest A/B (independent *different-family* judge `llama3.1:8b`,
  position-swapped, 4-dimension) showed `nova-ft` **ties** its base `qwen3.5:9b`
  (~8/10 ties, +0.03 preference).
- This matches the research consensus (e.g. ACE, arXiv:2510.04618) that
  in-context learning beats small-data fine-tuning for injecting facts.

**Nova learns through the in-context memory loop** — corrections become lessons +
temporal-KG facts retrieved into the prompt. That is the product; weight training
was an unproven experiment.

## What's here

DPO data generators (`dpo_*`), the fine-tune pipeline (`finetune*.py`), the GRPO
trainer (`grpo_train.py`), A/B harnesses (`ab_*`, `eval_harness.py`), GGUF/merge
tooling, training probes (`v9_e2e_probes`, `verify_phase_0`, …), the finetune
Dockerfile, and the host runner scripts.

## To revive

Move the relevant files back to `scripts/`, restore the Dockerfile COPYs and the
`_execute_finetune_check` body (see git history), and set `ENABLE_RLVR_SIGNALS=true`
+ `ENABLE_AUTO_FINETUNE=true`. Re-validate against a *different-family* judge
before trusting any deploy — and only ship a candidate that **wins** the A/B.
