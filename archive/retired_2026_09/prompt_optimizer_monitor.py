"""Heartbeat monitor handler for check_type='prompt_analyzer'.

Runs the PromptOptimizerAnalyzer after the nightly eval harness to:
  1. Detect metric drift patterns in eval_history.jsonl
  2. Draft candidate mutations via LLM
  3. Write candidates to prompt_modules table (subject to safety caps)
  4. Check for post-promotion regressions and auto-rollback if needed

This module is imported by heartbeat.py when check_type == 'prompt_analyzer'.
It is NOT imported at module level in heartbeat.py — it is loaded lazily inside
the dispatch branch to avoid startup cost.

Default monitor seeding (in heartbeat.py _DEFAULT_MONITORS):
    name:             "Prompt Optimizer"
    check_type:       "prompt_analyzer"
    schedule_seconds: 86400   # daily, after Quality Eval Harness
    notify_condition: "on_change"
    enabled:          driven by ENABLE_PROMPT_SELF_MOD config
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_prompt_analyzer(cfg: dict[str, Any]) -> str:
    """Execute one prompt-analyzer cycle.

    Returns a human-readable status string for the heartbeat result.
    cfg is the check_config dict from the monitor row (currently unused but
    reserved for future per-monitor override of target_category / module).
    """
    from app.config import config
    from app.core.prompt_optimizer import PromptOptimizerAnalyzer, PromptModuleStore

    if not config.ENABLE_PROMPT_SELF_MOD:
        return "[Prompt optimizer disabled — set ENABLE_PROMPT_SELF_MOD=true to enable]"

    store = PromptModuleStore()
    analyzer = PromptOptimizerAnalyzer(store=store)

    # Step 1: Check for post-promotion regressions and rollback if needed
    rollback_reasons = await analyzer.check_and_rollback_if_needed()

    # Step 2: Analyze drift and propose candidates
    new_candidate_ids = await analyzer.analyze_and_propose()

    parts: list[str] = []
    if rollback_reasons:
        parts.append(f"ROLLBACK({len(rollback_reasons)}): " + "; ".join(rollback_reasons[:2]))
    if new_candidate_ids:
        parts.append(f"proposed candidates: {new_candidate_ids}")
    else:
        parts.append("no new candidates")

    # Step 3: Summarize active module versions
    active = store.get_active_versions()
    non_baseline = {k: v for k, v in active.items() if v > 1}
    if non_baseline:
        parts.append("promoted: " + ", ".join(f"{k}@v{v}" for k, v in non_baseline.items()))

    return "PromptOptimizer: " + " | ".join(parts)
