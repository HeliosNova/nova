"""Training progress monitor — checks every 15 minutes, prints status."""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "finetune_progress.log"
INTERVAL = 900  # 15 minutes


import re

_STEP_RE = re.compile(r"Step (\d+)/(\d+)\s*\((\d+)%\)\s*\|\s*loss=([\d.eE+-]+)\s*\|\s*ETA:\s*(\d+)m")


def get_last_step():
    """Parse the last training step from the progress log.

    Total step count is parsed from the log itself (not hard-coded), so the
    monitor works across training runs with different dataset sizes / epochs.
    """
    if not LOG.exists():
        return None
    last_match = None
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                last_match = m
    if not last_match:
        return None
    step, total, pct, loss_s, eta_m = last_match.groups()
    try:
        loss = float(loss_s)
    except ValueError:
        loss = float("nan")
    step_i, total_i, eta_i = int(step), int(total), int(eta_m)
    return {
        "step": step_i,
        "total": total_i,
        "pct": round(step_i / total_i * 100, 1) if total_i else 0.0,
        "loss": loss,
        "eta_min": eta_i,
        "eta_hr": round(eta_i / 60, 1),
    }


def get_gpu():
    """Get GPU memory usage."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            return {
                "used_mb": int(parts[0]),
                "free_mb": int(parts[1]),
                "util_pct": int(parts[2]),
                "temp_c": int(parts[3]),
            }
    except Exception:
        pass
    return None


def main():
    print("=" * 60)
    print("  Nova Training Monitor (every 15 min)")
    print("=" * 60)
    print()

    prev_step = 0
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step = get_last_step()
        gpu = get_gpu()

        if step is None:
            print(f"[{now}] No training progress found — training may have finished or not started.")
            # Check if training process is still running
            if gpu and gpu["util_pct"] < 10:
                print(f"  GPU idle ({gpu['util_pct']}% util) — training likely finished!")
                break
        else:
            speed = ""
            if prev_step and step.get("step", 0) > prev_step:
                steps_done = step["step"] - prev_step
                speed = f" | {steps_done} steps in 15min ({steps_done * 4:.0f}/hr)"
            prev_step = step.get("step", 0)

            print(f"[{now}] Step {step.get('step','?')}/{step.get('total','?')} "
                  f"({step.get('pct','?')}%) | loss={step.get('loss','?'):.6f} | "
                  f"ETA: {step.get('eta_hr','?')}h{speed}")

            if gpu:
                print(f"  GPU: {gpu['used_mb']}MB used, {gpu['temp_c']}°C, {gpu['util_pct']}% util")

            # Check if done
            if step.get("step", 0) >= step.get("total", 999):
                print(f"\n  TRAINING COMPLETE at step {step['step']}!")
                break

        print()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
