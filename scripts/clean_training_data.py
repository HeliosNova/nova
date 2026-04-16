"""Clean training data by removing low-quality DPO pairs.

Removes:
  - Pairs where chosen < 30 chars
  - Pairs where rejected < 30 chars
  - Pairs where chosen length < 20% of rejected length (extreme asymmetry)

USAGE:
    python scripts/clean_training_data.py --data training_data.jsonl
    python scripts/clean_training_data.py --data training_data.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def clean_training_data(data_path: str, *, dry_run: bool = False) -> dict:
    """Clean a JSONL training file and return removal statistics."""
    path = Path(data_path)
    if not path.exists():
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    # Read all lines
    entries: list[dict] = []
    bad_json = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                bad_json += 1

    total = len(entries)
    print(f"Loaded {total} pairs from {data_path} ({bad_json} bad JSON lines skipped)")

    # Apply filters
    kept: list[dict] = []
    removed_short_chosen = 0
    removed_short_rejected = 0
    removed_asymmetric = 0

    for entry in entries:
        chosen = entry.get("chosen", "")
        rejected = entry.get("rejected", "")
        query = entry.get("query", "")

        # Check chosen < 30 chars
        if len(chosen) < 30:
            removed_short_chosen += 1
            continue

        # Check rejected < 30 chars
        if len(rejected) < 30:
            removed_short_rejected += 1
            continue

        # Check extreme asymmetry: chosen < 20% of rejected
        if len(rejected) > 0 and len(chosen) < 0.20 * len(rejected):
            removed_asymmetric += 1
            continue

        kept.append(entry)

    total_removed = total - len(kept)

    print(f"\n{'='*60}")
    print(f"Cleaning Results")
    print(f"{'='*60}")
    print(f"Original pairs:           {total}")
    print(f"Removed (chosen < 30):    {removed_short_chosen}")
    print(f"Removed (rejected < 30):  {removed_short_rejected}")
    print(f"Removed (asymmetric):     {removed_asymmetric}")
    print(f"Total removed:            {total_removed}")
    print(f"Remaining pairs:          {len(kept)}")
    print(f"{'='*60}")

    if dry_run:
        print("\n[DRY RUN] No files modified.")
    elif total_removed > 0:
        # Backup original
        backup_path = path.with_suffix(f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(path, backup_path)
        print(f"\nBackup saved to: {backup_path}")

        # Write cleaned data
        with open(path, "w", encoding="utf-8") as f:
            for entry in kept:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Cleaned data written to: {path}")
    else:
        print("\nNo pairs removed. Data is clean.")

    return {
        "total": total,
        "removed_short_chosen": removed_short_chosen,
        "removed_short_rejected": removed_short_rejected,
        "removed_asymmetric": removed_asymmetric,
        "total_removed": total_removed,
        "kept": len(kept),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean DPO training data")
    parser.add_argument("--data", required=True, help="Path to training_data.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without modifying files")
    args = parser.parse_args()

    clean_training_data(args.data, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
