#!/usr/bin/env python3
"""One-click fine-tuning pipeline for Nova.

Handles the full cycle:
1. Copy training data from Docker volume
2. Stop Ollama (free GPU VRAM)
3. DPO train with Unsloth on Qwen3.5-9B
4. Convert to GGUF (BF16 → Q4_K_M)
5. Register with Ollama
6. Restart Ollama + Nova
7. Update .env to use new model

Usage:
    python scripts/finetune_oneclick.py
    python scripts/finetune_oneclick.py --check     # Check readiness only
    python scripts/finetune_oneclick.py --skip-eval  # Skip A/B evaluation
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Paths
NOVA_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", "/data"))  # Container path for docker cp
FINETUNE_DIR = DATA_DIR / "finetune"
ADAPTER_DIR = FINETUNE_DIR / "adapter"
MERGED_DIR = FINETUNE_DIR / "merged"
GGUF_DIR = FINETUNE_DIR / "gguf"
TRAINING_DATA = DATA_DIR / "training_data.jsonl"
LOCAL_TRAINING_DATA = NOVA_DIR / "training_data.jsonl"
ENV_FILE = NOVA_DIR / ".env"
LLAMA_CPP_DIR = Path.home() / ".unsloth" / "llama.cpp"
CONVERTER = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZER = LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe"
RUN_HISTORY = NOVA_DIR / "finetune_output" / "run_history.json"

# Defaults
BASE_MODEL = os.getenv("FINETUNE_BASE_MODEL", "Qwen/Qwen3.5-9B")
MIN_PAIRS = int(os.getenv("FINETUNE_MIN_NEW_PAIRS", "15"))
EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "3"))
OLLAMA_CONTAINER = os.getenv("OLLAMA_CONTAINER", "nova-ollama")
MODEL_NAME = None  # Auto-detected by detect_next_version()


def detect_next_version() -> str:
    """Auto-detect the next nova-ft version by checking Ollama models."""
    global MODEL_NAME
    result = subprocess.run(
        ["docker", "exec", OLLAMA_CONTAINER, "ollama", "list"],
        capture_output=True, text=True, timeout=30,
    )
    max_ver = 0
    import re as _re
    # Match nova-ft-v3, nova-ft-v2, nova-ft-v8-q8, nova-ft-v8-fp16, etc.
    # Extract just the leading version digits after 'nova-ft-v', ignoring any
    # quantization suffix like -q8 / -fp16 / -q4km.
    _ver_re = _re.compile(r"^nova-ft-v(\d+)(?:[-:].*)?$")
    for line in result.stdout.splitlines():
        for token in line.split():
            m = _ver_re.match(token)
            if m:
                try:
                    max_ver = max(max_ver, int(m.group(1)))
                except ValueError:
                    pass
            elif token == "nova-ft:latest" or token.startswith("nova-ft:"):
                max_ver = max(max_ver, 1)
    MODEL_NAME = f"nova-ft-v{max_ver + 1}"
    print(f"  Next model version: {MODEL_NAME}")
    return MODEL_NAME


def run(cmd, **kwargs):
    """Run a command and return output."""
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    ignore_errors = kwargs.pop("ignore_errors", False)
    timeout = kwargs.pop("timeout", 600)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 and not ignore_errors:
        print(f"  ERROR: {result.stderr[:500]}")
    return result


def docker_compose(*args):
    """Run docker compose command."""
    return run(["docker", "compose", "-f", str(NOVA_DIR / "docker-compose.yml")] + list(args), timeout=120)


def ollama_exec(*args):
    """Run command in Ollama container."""
    return run(["docker", "exec", OLLAMA_CONTAINER] + list(args), timeout=300)


def check_readiness():
    """Check if we have enough training data."""
    # Copy training data from Docker volume
    result = run(["docker", "cp", f"nova-app:{DATA_DIR}/training_data.jsonl", str(LOCAL_TRAINING_DATA)], ignore_errors=True)

    if not LOCAL_TRAINING_DATA.exists():
        print("No training data found.")
        return False, 0

    with open(LOCAL_TRAINING_DATA, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    valid = 0
    for line in lines:
        try:
            d = json.loads(line)
            if d.get("query") and d.get("chosen"):
                valid += 1
        except json.JSONDecodeError:
            continue

    print(f"Training pairs: {valid} (need {MIN_PAIRS})")
    return valid >= MIN_PAIRS, valid


def step_1_copy_data():
    """Copy training data from Docker volume to host."""
    print("\n=== Step 1: Copy training data ===")
    run(["docker", "cp", f"nova-app:{DATA_DIR}/training_data.jsonl", str(LOCAL_TRAINING_DATA)])
    with open(LOCAL_TRAINING_DATA, encoding="utf-8") as f:
        count = sum(1 for l in f if l.strip())
    print(f"  {count} training pairs copied")
    return count


def step_2_stop_ollama():
    """Stop Ollama to free GPU VRAM."""
    print("\n=== Step 2: Stop Ollama ===")
    docker_compose("stop", "ollama")
    time.sleep(5)
    print("  Ollama stopped, GPU VRAM free")


def step_3_train():
    """Run DPO training."""
    print("\n=== Step 3: DPO Training ===")
    # Find the finetune venv Python
    venv_python = NOVA_DIR / "finetune_env" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = NOVA_DIR / "finetune_env" / "bin" / "python"
    if not venv_python.exists():
        print("  ERROR: finetune_env not found. Run: pip install unsloth")
        return False

    result = run([
        str(venv_python), str(NOVA_DIR / "scripts" / "finetune.py"),
        "--data", str(LOCAL_TRAINING_DATA),
        "--model", BASE_MODEL,
        "--epochs", str(EPOCHS),
    ], timeout=172800)  # 48 hours — training takes ~26h on RTX 3090

    if result.returncode != 0:
        print("  Training failed!")
        return False

    print("  Training complete!")
    return True


def step_4_convert_gguf():
    """Convert trained model to GGUF Q4_K_M."""
    print("\n=== Step 4: Convert to GGUF ===")

    if not MERGED_DIR.exists():
        print(f"  ERROR: Merged model not found at {MERGED_DIR}")
        return False

    bf16_path = FINETUNE_DIR / f"{MODEL_NAME}-bf16.gguf"
    q4km_path = FINETUNE_DIR / f"{MODEL_NAME}-q4km.gguf"

    # Find converter
    venv_python = NOVA_DIR / "finetune_env" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = NOVA_DIR / "finetune_env" / "bin" / "python"

    # Step 4a: Convert HF → BF16 GGUF
    print("  Converting to BF16 GGUF...")
    result = run([
        str(venv_python), str(CONVERTER),
        str(MERGED_DIR),
        "--outtype", "bf16",
        "--outfile", str(bf16_path),
    ], timeout=600)

    if result.returncode != 0 or not bf16_path.exists():
        print("  BF16 conversion failed!")
        return False

    print(f"  BF16 GGUF: {bf16_path} ({bf16_path.stat().st_size / 1e9:.1f} GB)")

    # Step 4b: Quantize BF16 → Q4_K_M
    print("  Quantizing to Q4_K_M...")
    if not QUANTIZER.exists():
        # Try alternative path
        alt = LLAMA_CPP_DIR / "build" / "tools" / "quantize" / "Release" / "llama-quantize.exe"
        if alt.exists():
            quantizer = alt
        else:
            print(f"  ERROR: llama-quantize not found at {QUANTIZER}")
            return False
    else:
        quantizer = QUANTIZER

    result = run([str(quantizer), str(bf16_path), str(q4km_path), "Q4_K_M"], timeout=600)

    if result.returncode != 0 or not q4km_path.exists():
        print("  Quantization failed!")
        return False

    print(f"  Q4_K_M GGUF: {q4km_path} ({q4km_path.stat().st_size / 1e9:.1f} GB)")

    # Clean up BF16 (huge file)
    if bf16_path.exists() and q4km_path.exists():
        bf16_path.unlink()
        print("  Cleaned up BF16 intermediate file")

    return True


def step_5_register_ollama():
    """Register the GGUF model with Ollama."""
    print("\n=== Step 5: Register with Ollama ===")

    q4km_path = FINETUNE_DIR / f"{MODEL_NAME}-q4km.gguf"
    if not q4km_path.exists():
        print(f"  ERROR: GGUF not found at {q4km_path}")
        return False

    # Start Ollama
    docker_compose("start", "ollama")
    time.sleep(15)

    # Copy GGUF to container
    run(["docker", "cp", str(q4km_path), f"{OLLAMA_CONTAINER}:/root/.ollama/{MODEL_NAME}.gguf"], timeout=300)

    # Create Modelfile and register
    modelfile = (
        f"FROM /root/.ollama/{MODEL_NAME}.gguf\n"
        "TEMPLATE {{ .Prompt }}\n"
        "RENDERER qwen3.5\n"
        "PARSER qwen3.5\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_predict 4000\n"
        "PARAMETER num_ctx 32768\n"
    )

    # Write modelfile inside container
    ollama_exec("sh", "-c", f"echo '{modelfile}' > /tmp/Modelfile.nova")
    result = ollama_exec("ollama", "create", MODEL_NAME, "-f", "/tmp/Modelfile.nova")

    if result.returncode != 0:
        print("  Registration failed!")
        return False

    # Verify
    result = ollama_exec("ollama", "list")
    if MODEL_NAME in result.stdout:
        print(f"  Model '{MODEL_NAME}' registered successfully!")
        return True

    print("  Model not found after registration")
    return False


def step_6_update_env():
    """Update .env to use the new model."""
    print("\n=== Step 6: Update .env ===")
    import re as _re
    env_content = ENV_FILE.read_text()
    if f"LLM_MODEL={MODEL_NAME}" not in env_content:
        # Replace any existing LLM_MODEL line (nova-ft-vN, qwen3.5:*, etc.)
        env_content = _re.sub(
            r"^LLM_MODEL=.*$",
            f"LLM_MODEL={MODEL_NAME}",
            env_content,
            flags=_re.MULTILINE,
        )
        ENV_FILE.write_text(env_content)
        print(f"  Updated .env: LLM_MODEL={MODEL_NAME}")
    else:
        print(f"  .env already uses {MODEL_NAME}")


def step_7_restart():
    """Restart Nova to use the new model."""
    print("\n=== Step 7: Restart Nova ===")
    docker_compose("up", "-d", "--force-recreate", "nova")
    time.sleep(15)
    print("  Nova restarted with new model")


def save_run(status, pairs, **extra):
    """Save run metadata."""
    RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if RUN_HISTORY.exists():
        try:
            history = json.loads(RUN_HISTORY.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    history.append({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "base_model": BASE_MODEL,
        "ft_model": MODEL_NAME,
        "total_pairs": pairs,
        "epochs": EPOCHS,
        **extra,
    })

    RUN_HISTORY.write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Nova One-Click Fine-Tuning")
    parser.add_argument("--check", action="store_true", help="Check readiness only")
    parser.add_argument("--skip-eval", action="store_true", help="Skip A/B evaluation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing adapter)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Nova One-Click Fine-Tuning Pipeline")
    print("=" * 60)

    # Auto-detect next version
    detect_next_version()

    # Check readiness
    ready, pairs = check_readiness()
    if args.check:
        print(f"\nReady: {ready} ({pairs} pairs, need {MIN_PAIRS})")
        return

    if not ready:
        print(f"\nNot enough training data ({pairs} pairs, need {MIN_PAIRS})")
        print("Generate more DPO pairs by correcting Nova during conversations,")
        print("or run: python scripts/training_curriculum.py")
        return

    print(f"\nProceeding with {pairs} training pairs...")

    # Run pipeline
    try:
        step_1_copy_data()
        step_2_stop_ollama()

        if not args.skip_train:
            if not step_3_train():
                save_run("train_failed", pairs)
                docker_compose("start", "ollama")
                return
        else:
            print("\n=== Step 3: Skipped (using existing adapter) ===")

        if not step_4_convert_gguf():
            save_run("convert_failed", pairs)
            docker_compose("start", "ollama")
            return

        if not step_5_register_ollama():
            save_run("register_failed", pairs)
            return

        step_6_update_env()
        step_7_restart()

        save_run("deployed", pairs)
        print("\n" + "=" * 60)
        print(f"  Fine-tuning complete! Model '{MODEL_NAME}' deployed.")
        print("=" * 60)

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        save_run("error", pairs, error=str(e))
        # Try to restart Ollama
        docker_compose("start", "ollama")
        raise


if __name__ == "__main__":
    main()
