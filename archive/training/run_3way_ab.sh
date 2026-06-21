#!/usr/bin/env bash
# 3-way A/B eval: v16 (production 9B) vs v18 (old 27B) vs v19 (fresh 27B)
# All 3 pairwise runs use the same holdout queries (sampled with seed=42) and qwen3.6:27b as judge.
set -e

PY=/home/sysadmin/finetune_env/bin/python
Q="/mnt/f/Helios Project/nova_/finetune/ab_3way/holdout_queries.json"
OUT="/mnt/f/Helios Project/nova_/finetune/ab_3way"
SCRIPT="/mnt/f/Helios Project/nova_/scripts/eval_harness.py"

mkdir -p "$OUT"
cd "/mnt/f/Helios Project/nova_"

echo "=== 1/3: v16 (nova-ft 9B) vs v19 (27B fresh) ==="
"$PY" "$SCRIPT" --base nova-ft --candidate nova-ft-v19-q8 --judge qwen3.6:27b \
    --queries "$Q" --output "$OUT/v16_vs_v19.json" 2>&1 | tail -30

echo ""
echo "=== 2/3: v16 vs v18 (27B old) ==="
"$PY" "$SCRIPT" --base nova-ft --candidate nova-ft-v18-q8 --judge qwen3.6:27b \
    --queries "$Q" --output "$OUT/v16_vs_v18.json" 2>&1 | tail -30

echo ""
echo "=== 3/3: v18 vs v19 ==="
"$PY" "$SCRIPT" --base nova-ft-v18-q8 --candidate nova-ft-v19-q8 --judge qwen3.6:27b \
    --queries "$Q" --output "$OUT/v18_vs_v19.json" 2>&1 | tail -30

echo ""
echo "=== ALL DONE ==="
