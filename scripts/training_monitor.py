"""Training progress monitor — checks every 15 minutes, prints status."""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "finetune_progress.log"
INTERVAL = 900  # 15 minutes


def get_last_step():
    """Parse the last training step from the progress log."""
    if not LOG.exists():
        return None
    with open(LOG, encoding="utf-8") as f:
        lines = [l.strip() for l in f if "Step " in l and "/534" in l]
    if not lines:
        return None
    last = lines[-1]
    # Parse: "2026-04-02 06:07:16 [INFO] Step 126/534 (24%) | loss=0.001 | ETA: 815m"
    parts = {}
    try:
        parts["raw"] = last
        step_part = last.split("Step ")[1].split("/")[0]
        parts["step"] = int(step_part)
        parts["total"] = 534
        parts["pct"] = round(parts["step"] / parts["total"] * 100, 1)
        if "loss=" in last:
            parts["loss"] = float(last.split("loss=")[1].split(" |")[0].split("\n")[0])
        if "ETA:" in last:
            parts["eta_min"] = int(last.split("ETA: ")[1].split("m")[0])
            parts["eta_hr"] = round(parts["eta_min"] / 60, 1)
    except (IndexError, ValueError):
        pass
    return parts


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
