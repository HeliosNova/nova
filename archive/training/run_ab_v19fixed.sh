#!/usr/bin/env bash
# A/B eval: v16 (nova-ft, 9B production) vs v19-fixed (27B, metadata-corrected).
# Judge: qwen3.6:27b base. Same 10 holdout queries (seed=42).
set -e

PY=/home/sysadmin/finetune_env/bin/python
Q="/mnt/f/Helios Project/nova_/finetune/ab_3way/holdout_queries.json"
OUT="/mnt/f/Helios Project/nova_/finetune/ab_3way/v16_vs_v19fixed.json"
SCRIPT="/mnt/f/Helios Project/nova_/scripts/eval_harness.py"

cd "/mnt/f/Helios Project/nova_"
echo "=== A/B: v16 (nova-ft) vs v19-fixed ==="
"$PY" "$SCRIPT" --base nova-ft --candidate nova-ft-v19-fixed --judge qwen3.6:27b \
    --queries "$Q" --output "$OUT" 2>&1 | tail -40
echo "=== DONE ==="
