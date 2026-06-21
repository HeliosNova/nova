"""Backfill run_history.json with the v16 train that landed on disk
but never wrote a metadata row.

Reads adapter/training_meta.json (ground truth from the actual train),
merges with whatever's already in run_history.json, and dedups by
ft_model + completed_at.

Run on host:
    python scripts/backfill_run_history.py --output-dir C:/data/finetune
And inside container:
    docker exec nova-app python /app/scripts/backfill_run_history.py --output-dir /data/finetune
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: Path) -> object:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="finetune output dir (contains adapter/, run_history.json)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if not out.exists():
        print(f"[err] {out} does not exist", file=sys.stderr)
        return 1

    history_path = out / "run_history.json"
    history = load_json(history_path) or []
    if not isinstance(history, list):
        print(f"[err] {history_path} is not a list", file=sys.stderr)
        return 1

    meta_path = out / "adapter" / "training_meta.json"
    meta = load_json(meta_path)
    if not isinstance(meta, dict):
        print(f"[info] no adapter/training_meta.json — nothing to backfill")
        return 0

    completed_at = meta.get("trained_at")
    training_pairs = meta.get("training_pairs", 0)
    if not completed_at or not training_pairs:
        print(f"[info] training_meta.json missing trained_at/training_pairs — nothing to backfill")
        return 0

    # Skip if already recorded
    for run in history:
        if run.get("completed_at") == completed_at and run.get("training_pairs") == training_pairs:
            print(f"[info] v16 success already in history (completed_at={completed_at}); nothing to do")
            return 0

    # Build the v16 success record
    backfill = {
        "started_at": completed_at,
        "data_path": str(out.parent / "training_data.jsonl"),
        "output_dir": str(out),
        "base_model": meta.get("model", "qwen3.5:9b"),
        "ft_model": "nova-ft-v16-q8",
        "deployed_model": "nova-ft",
        "status": "deployed",
        "total_pairs": training_pairs,
        "new_pairs": training_pairs,  # unknown delta, set conservatively to total
        "training_pairs": training_pairs,
        "holdout_queries": 0,
        "adapter_dir": str(out / "adapter"),
        "final_loss": meta.get("final_loss"),
        "epochs": meta.get("epochs"),
        "lora_rank": meta.get("lora_rank"),
        "completed_at": completed_at,
        "backfilled": True,
        "reason": "Backfilled from adapter/training_meta.json — original run did not write to history.",
    }
    history.append(backfill)
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[ok] appended v16 backfill to {history_path}")
    print(f"     training_pairs={training_pairs} completed_at={completed_at}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
