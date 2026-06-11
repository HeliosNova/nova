"""GRPO dataset builder — turns RLVR verifiable_signals into trainable groups.

GRPO (Group Relative Policy Optimization) trains a policy by:
  1. Sampling N rollouts per prompt (a "group")
  2. Scoring each rollout with a deterministic verifier
  3. Computing within-group advantages: A_i = (r_i - mean) / (std + eps)
  4. Updating the policy with the standard PPO loss + KL penalty to ref

Nova's `verifiable_signals` table accumulates (query, response, reward, type)
tuples during normal operation. When the same query appears multiple times
(e.g. evals re-running, quizzes repeating, similar user queries), we have a
natural group: N responses to the same prompt with N rewards.

This module turns that table into:
  * GRPOGroup(prompt, completions[], rewards[], advantages[])
  * a HF Dataset shape consumable by trl/Unsloth GRPOTrainer
  * a DPO-fallback dataset (chosen=highest, rejected=lowest within group)
    when group_size < 4 or std is degenerate.

A query is "groupable" when 2+ responses exist with non-zero reward variance.
Queries with only one response are dropped (no learning signal); queries
where all rewards are identical are dropped (no relative information).

Public surface:
    build_groups(min_group_size=2, signal_types=None) -> list[GRPOGroup]
    to_grpo_dataset(groups) -> dict (prompt, completions, rewards) lists
    to_dpo_pairs(groups) -> list[dict] (prompt, chosen, rejected)
    write_jsonl(items, path) -> int
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.core import rlvr
from app.database import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group dataclass
# ---------------------------------------------------------------------------

@dataclass
class GRPOGroup:
    prompt: str
    signal_type: str
    completions: list[str]
    rewards: list[float]
    advantages: list[float] = field(default_factory=list)
    signal_ids: list[int] = field(default_factory=list)

    def is_trainable(self, min_size: int = 2, min_std: float = 0.05) -> bool:
        """Trainable when group has enough rollouts and reward variance.

        Below min_size: not enough relative information.
        Below min_std: all rewards are essentially equal — no advantage signal.
        """
        if len(self.rewards) < min_size:
            return False
        n = len(self.rewards)
        mean = sum(self.rewards) / n
        var = sum((r - mean) ** 2 for r in self.rewards) / n
        return var ** 0.5 >= min_std

    def compute_advantages(self) -> None:
        """Standardize rewards: A_i = (r_i - mean) / (std + 1e-8).

        Standard GRPO normalization — keeps the policy gradient bounded
        regardless of absolute reward scale.
        """
        if not self.rewards:
            self.advantages = []
            return
        n = len(self.rewards)
        mean = sum(self.rewards) / n
        var = sum((r - mean) ** 2 for r in self.rewards) / n
        std = var ** 0.5
        if std < 1e-8:
            self.advantages = [0.0] * n
            return
        self.advantages = [(r - mean) / (std + 1e-8) for r in self.rewards]


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_TRAIL_RE = re.compile(r"[.!?,:;\s]+$")


def _normalize_query(q: str) -> str:
    """Normalize a query so trivially different phrasings group together.

    Lowercase, collapse whitespace, strip trailing punctuation. NOT semantic
    normalization — just exact-match defenses. Two queries that mean the same
    thing but differ in wording will land in separate groups, which is correct
    behavior for GRPO (the model has different completion contexts for them).
    """
    if not q:
        return ""
    q = _WS_RE.sub(" ", q.strip().lower())
    q = _PUNCT_TRAIL_RE.sub("", q)
    return q


# ---------------------------------------------------------------------------
# Group builder
# ---------------------------------------------------------------------------

def build_groups(
    *,
    min_group_size: int = 2,
    signal_types: list[str] | None = None,
    only_unconsumed: bool = True,
    limit: int = 5000,
) -> list[GRPOGroup]:
    """Read verifiable_signals from DB and group by normalized query.

    Groups are typed: a single query may produce one group per signal_type
    (a query with both `tool_correct` and `claim_grounded` signals yields two
    groups, since the rewards aren't comparable across types).
    """
    types = list(signal_types) if signal_types else list(rlvr.SIGNAL_TYPES)
    signals = []
    for stype in types:
        chunk = rlvr.query_signals(
            stype, only_unconsumed=only_unconsumed, limit=limit,
        )
        signals.extend(chunk)
    if not signals:
        return []

    grouped: dict[tuple[str, str], GRPOGroup] = {}
    for s in signals:
        key = (_normalize_query(s.query), s.signal_type)
        if not key[0]:
            continue
        g = grouped.get(key)
        if g is None:
            g = GRPOGroup(
                prompt=s.query,
                signal_type=s.signal_type,
                completions=[],
                rewards=[],
                signal_ids=[],
            )
            grouped[key] = g
        # Skip empty completions — can't train on them
        if not s.response or not s.response.strip():
            continue
        g.completions.append(s.response)
        g.rewards.append(float(s.signal_value))
        g.signal_ids.append(s.id)

    out: list[GRPOGroup] = []
    for g in grouped.values():
        if len(g.completions) < min_group_size:
            continue
        g.compute_advantages()
        out.append(g)

    out.sort(key=lambda g: -len(g.completions))
    logger.info(
        "[grpo_dataset] built %d groups from %d signals (types=%s)",
        len(out), len(signals), sorted(set(s.signal_type for s in signals)),
    )
    return out


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

def to_grpo_dataset(
    groups: Iterable[GRPOGroup],
    *,
    require_trainable: bool = True,
) -> dict:
    """Build a parallel-list dict suitable for HF Dataset.from_dict().

    Output shape:
        {
          "prompt":      [str, ...],          # one per (group, completion)
          "completion":  [str, ...],
          "reward":      [float, ...],
          "advantage":   [float, ...],
          "group_id":    [int, ...],
          "signal_type": [str, ...],
        }

    Each row is one (prompt, completion, reward) triple. Group IDs allow the
    trainer to recover which rows belong together. trl GRPOTrainer doesn't
    require group_id (it samples groups itself), but the field is useful for
    offline GRPO variants that pre-compute advantages.
    """
    flat: dict[str, list] = {
        "prompt": [], "completion": [], "reward": [],
        "advantage": [], "group_id": [], "signal_type": [],
    }
    gid = 0
    for g in groups:
        if require_trainable and not g.is_trainable():
            continue
        for comp, rew, adv in zip(g.completions, g.rewards, g.advantages):
            flat["prompt"].append(g.prompt)
            flat["completion"].append(comp)
            flat["reward"].append(float(rew))
            flat["advantage"].append(float(adv))
            flat["group_id"].append(gid)
            flat["signal_type"].append(g.signal_type)
        gid += 1
    return flat


def to_dpo_pairs(
    groups: Iterable[GRPOGroup],
    *,
    require_trainable: bool = True,
    min_reward_gap: float = 0.5,
) -> list[dict]:
    """Within each group pick (highest, lowest) and emit a DPO pair.

    Used as a fallback when trl GRPOTrainer isn't available, or when groups
    are too small for proper relative advantage computation. The pair's
    reward gap must clear `min_reward_gap` so we don't train on noise.

    Output shape:
        [{"prompt": str, "chosen": str, "rejected": str,
          "chosen_reward": float, "rejected_reward": float,
          "signal_type": str, "source": "grpo_dataset"}, ...]
    """
    out: list[dict] = []
    for g in groups:
        if require_trainable and not g.is_trainable():
            continue
        if len(g.completions) < 2:
            continue
        order = sorted(
            range(len(g.completions)),
            key=lambda i: g.rewards[i],
            reverse=True,
        )
        best_i, worst_i = order[0], order[-1]
        gap = g.rewards[best_i] - g.rewards[worst_i]
        if gap < min_reward_gap:
            continue
        # Skip if best and worst completions are identical
        if g.completions[best_i].strip() == g.completions[worst_i].strip():
            continue
        out.append({
            "prompt": g.prompt,
            "chosen": g.completions[best_i],
            "rejected": g.completions[worst_i],
            "chosen_reward": float(g.rewards[best_i]),
            "rejected_reward": float(g.rewards[worst_i]),
            "signal_type": g.signal_type,
            "source": "grpo_dataset",
        })
    return out


def write_jsonl(items: Iterable[dict], path: str) -> int:
    """Write a list of dicts to JSONL. Returns count written."""
    n = 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
            n += 1
    logger.info("[grpo_dataset] wrote %d items to %s", n, path)
    return n


# ---------------------------------------------------------------------------
# Stats helpers (used by training scripts to decide what's runnable)
# ---------------------------------------------------------------------------

def stats(groups: Iterable[GRPOGroup]) -> dict:
    """Compute summary stats over a list of groups."""
    glist = list(groups)
    n_groups = len(glist)
    n_trainable = sum(1 for g in glist if g.is_trainable())
    completions = sum(len(g.completions) for g in glist)
    by_type: dict[str, int] = {}
    by_size: dict[int, int] = {}
    for g in glist:
        by_type[g.signal_type] = by_type.get(g.signal_type, 0) + 1
        size_bucket = min(len(g.completions), 8)
        by_size[size_bucket] = by_size.get(size_bucket, 0) + 1
    return {
        "n_groups": n_groups,
        "n_trainable": n_trainable,
        "n_completions": completions,
        "by_signal_type": by_type,
        "by_size_bucket": by_size,
    }
