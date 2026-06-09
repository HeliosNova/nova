"""TIES-style LoRA continual-merge for Nova fine-tunes.

When `ENABLE_LORA_CONTINUAL_MERGE=true`, after each new DPO/SimPO LoRA is
trained we merge it with the previously-deployed adapter rather than replacing
it. This keeps weight memory roughly constant across iterations, mitigating
catastrophic forgetting that would otherwise erode prior corrections each
weekly cycle.

Algorithm follows the TIES paper (arxiv 2306.01708):
    1. Trim: zero out small-magnitude task vectors per-tensor (top-K by abs value)
    2. Elect sign: per-parameter, pick the sign with greater total magnitude across adapters
    3. Disjoint merge: average only the values that agree with the elected sign

Falls back gracefully:
- If no prior adapter exists (first run): use the new adapter as-is
- If torch/safetensors unavailable: skip merge with a warning
- If shapes don't match (rank/target_modules changed): skip merge with a warning

USAGE (called from scripts/finetune.py after train()):
    new_adapter = train(...)
    if config_flag_enabled:
        merged = ties_merge(new_adapter, prior_adapter, alpha=0.5)
        return merged or new_adapter

Source-of-truth pointer: written to /data/finetune/active_continual_adapter.json
on success so the next run knows which adapter to use as "prior".
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


_POINTER_FILE = "/data/finetune/active_continual_adapter.json"


def _load_lora_state(adapter_dir: str):
    """Load LoRA weights from a saved adapter directory. Returns dict[name -> tensor]."""
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError(f"safetensors required for LoRA merge: {e}")

    candidates = [
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    for cand in candidates:
        path = os.path.join(adapter_dir, cand)
        if os.path.exists(path):
            if cand.endswith(".safetensors"):
                return load_file(path)
            else:
                import torch
                return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _save_lora_state(state: dict, adapter_dir: str) -> None:
    """Write merged weights back as adapter_model.safetensors."""
    from safetensors.torch import save_file
    out_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    save_file(state, out_path)


def ties_merge(
    new_adapter_dir: str,
    prior_adapter_dir: str | None,
    *,
    alpha: float = 0.5,
    trim_threshold: float = 0.20,
) -> str | None:
    """Merge `new_adapter` with `prior_adapter` using TIES (Trim, Elect-sign, Disjoint).

    `alpha` weights how much of the new adapter survives (1.0 = pure new, 0.0 = pure prior).
    `trim_threshold` controls per-tensor sparsification: any value with |x| < threshold *
    max(|tensor|) is zeroed before merge.

    Returns path to a NEW directory (sibling of new_adapter_dir, suffixed `_merged`) on
    success, or None if merge couldn't proceed (in which case the caller should use
    the unmerged new adapter).
    """
    if not prior_adapter_dir or not os.path.isdir(prior_adapter_dir):
        logger.info("[LoRA-merge] no prior adapter — using new adapter as the seed")
        return None

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.warning("[LoRA-merge] torch unavailable — skipping merge")
        return None

    try:
        new_state = _load_lora_state(new_adapter_dir)
        prior_state = _load_lora_state(prior_adapter_dir)
    except Exception as e:
        logger.warning("[LoRA-merge] state load failed: %s — skipping merge", e)
        return None

    # Sanity: keys must match. If they don't (rank or target_modules changed), abort.
    if set(new_state.keys()) != set(prior_state.keys()):
        new_only = set(new_state) - set(prior_state)
        prior_only = set(prior_state) - set(new_state)
        logger.warning(
            "[LoRA-merge] adapter shape mismatch — new_only=%d prior_only=%d. "
            "Skipping merge (rank or target_modules changed?).",
            len(new_only), len(prior_only),
        )
        return None

    import torch
    merged_state: dict = {}

    for key in new_state.keys():
        a_new = new_state[key].float()
        a_prior = prior_state[key].float()
        if a_new.shape != a_prior.shape:
            logger.warning(
                "[LoRA-merge] tensor shape mismatch on %s — skipping this tensor",
                key,
            )
            merged_state[key] = a_new
            continue

        # 1. Task vectors (delta from "no adapter" = the LoRA update itself)
        # 2. Trim: keep only top (1 - trim_threshold) by magnitude
        new_max = a_new.abs().max().item() or 1e-9
        prior_max = a_prior.abs().max().item() or 1e-9
        new_mask = (a_new.abs() >= trim_threshold * new_max).float()
        prior_mask = (a_prior.abs() >= trim_threshold * prior_max).float()
        new_trim = a_new * new_mask
        prior_trim = a_prior * prior_mask

        # 3. Elect sign: combined magnitude per element decides sign
        sign = torch.sign(alpha * new_trim + (1.0 - alpha) * prior_trim)

        # 4. Disjoint merge: include each value only if its sign matches
        new_keep = (torch.sign(new_trim) == sign).float() * new_trim
        prior_keep = (torch.sign(prior_trim) == sign).float() * prior_trim

        merged = alpha * new_keep + (1.0 - alpha) * prior_keep
        # Restore dtype to match new_adapter
        merged_state[key] = merged.to(new_state[key].dtype)

    # Output directory: <new_adapter_dir>_merged. Copy non-weight files (config,
    # tokenizer, etc.) so HuggingFace loaders find everything.
    out_dir = new_adapter_dir.rstrip("/\\") + "_merged"
    os.makedirs(out_dir, exist_ok=True)

    # Copy adapter_config.json + tokenizer files from new_adapter_dir
    import shutil
    for fname in os.listdir(new_adapter_dir):
        if fname.startswith("adapter_model"):
            continue
        src = os.path.join(new_adapter_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    _save_lora_state(merged_state, out_dir)

    # Persist pointer so next run uses this as "prior"
    try:
        os.makedirs(os.path.dirname(_POINTER_FILE), exist_ok=True)
        with open(_POINTER_FILE, "w") as f:
            json.dump({
                "path": out_dir,
                "merged_from_new": new_adapter_dir,
                "merged_from_prior": prior_adapter_dir,
                "alpha": alpha,
                "trim_threshold": trim_threshold,
            }, f, indent=2)
    except Exception as e:
        logger.warning("[LoRA-merge] pointer write failed: %s", e)

    logger.info("[LoRA-merge] merged adapter written to %s", out_dir)
    return out_dir


def get_prior_adapter_path() -> str | None:
    """Return the path to the last continually-merged adapter, if one exists."""
    try:
        if not os.path.exists(_POINTER_FILE):
            return None
        with open(_POINTER_FILE) as f:
            data = json.load(f)
        path = data.get("path")
        if path and os.path.isdir(path):
            return path
    except Exception as e:
        logger.warning("[LoRA-merge] pointer read failed: %s", e)
    return None
