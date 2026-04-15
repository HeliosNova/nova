#!/bin/bash
LOG="/c/Users/sysadmin/Desktop/Helios Project/nova_/finetune_progress.log"
while true; do
    # Check if python training is still running
    if ! tasklist 2>/dev/null | grep -q "python.exe.*19020"; then
        echo "$(date): Training process FINISHED or CRASHED"
        tail -5 "$LOG"
        nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null
        break
    fi
    # Report progress
    LAST=$(grep "Step " "$LOG" | tail -1)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
    echo "$(date): $LAST | GPU: $GPU"
    sleep 1800  # 30 min
done
