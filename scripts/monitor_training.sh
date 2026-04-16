#!/bin/bash
LOG="/c/Users/sysadmin/Desktop/Helios Project/nova_/finetune_progress.log"
while true; do
    # Check if ANY python process is using >2GB RAM (training)
    TRAINING_ALIVE=$(tasklist 2>/dev/null | grep "python.exe" | awk '{gsub(/,/,"",$5); if($5+0 > 2000000) print $2}')
    if [ -z "$TRAINING_ALIVE" ]; then
        echo "$(date): Training process FINISHED or CRASHED"
        tail -5 "$LOG"
        nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null
        break
    fi
    LAST=$(grep "Step " "$LOG" 2>/dev/null | tail -1)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
    echo "$(date): $LAST | GPU: $GPU"
    sleep 1800
done
