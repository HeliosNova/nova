"""Generate synthetic rejected examples for DPO training.

For lessons with no wrong_answer, calls the BASE model (not fine-tuned) to
generate an unaligned response, then pairs it with the lesson's correct_answer
to create DPO training pairs.

USAGE:
    # Run inside Docker container (direct DB access):
    python scripts/generate_synthetic_pairs.py

    # Run from host with explicit paths:
    python scripts/generate_synthetic_pairs.py \
        --db /path/to/nova.db \
        --output /path/to/training_data.jsonl \
        --ollama-url http://localhost:11434 \
        --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "/data/nova.db"
DEFAULT_OUTPUT_PATH = "/data/training_data.jsonl"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"  # BASE model, not fine-tuned


def get_lessons_without_wrong_answer(db_path: str) -> list[dict]:
    """Get all lessons where wrong_answer is NULL or empty."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, topic, correct_answer, lesson_text, context "
        "FROM lessons WHERE wrong_answer IS NULL OR wrong_answer = ''"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def generate_rejected_response(
    topic: str,
    ollama_url: str,
    model: str,
    timeout: int = 60,
) -> str | None:
    """Call Ollama base model to generate a response for the topic.

    Uses a neutral prompt so the base model gives its unaligned answer,
    which serves as the 'rejected' example in DPO.
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "prompt": topic,
        "stream": False,
        "think": False,  # Disable thinking mode for Qwen3.5 — otherwise response is empty
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response = data.get("response", "").strip()
            if response:
                return response
    except urllib.error.URLError as e:
        logger.warning("Ollama request failed for '%s': %s", topic[:40], e)
    except Exception as e:
        logger.warning("Unexpected error for '%s': %s", topic[:40], e)

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic rejected DPO pairs")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to nova.db")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to training_data.jsonl (append)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name (NOT fine-tuned)")
    parser.add_argument("--limit", type=int, default=0, help="Max lessons to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show lessons but don't generate")
    args = parser.parse_args()

    # Check DB exists
    if not Path(args.db).exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    lessons = get_lessons_without_wrong_answer(args.db)
    logger.info("Found %d lessons with empty wrong_answer", len(lessons))

    if args.limit > 0:
        lessons = lessons[:args.limit]
        logger.info("Limited to %d lessons", len(lessons))

    if args.dry_run:
        for l in lessons:
            print(f"  id={l['id']} topic={l['topic'][:60]}")
        print(f"\n[DRY RUN] Would generate {len(lessons)} synthetic pairs")
        return

    # Check Ollama is reachable
    import urllib.request
    try:
        with urllib.request.urlopen(f"{args.ollama_url}/api/tags", timeout=10):
            pass
    except Exception as e:
        logger.error("Cannot reach Ollama at %s: %s", args.ollama_url, e)
        sys.exit(1)

    generated = 0
    skipped = 0
    failed = 0

    # Load existing queries to avoid duplicates
    existing_queries = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    existing_queries.add(entry.get("query", "").strip().lower())
                except json.JSONDecodeError:
                    continue

    # Open output file in append mode
    out_f = open(output_path, "a", encoding="utf-8")

    for i, lesson in enumerate(lessons, 1):
        topic = lesson["topic"]
        correct = lesson["correct_answer"]

        # Skip if this topic already exists as a query
        if topic.strip().lower() in existing_queries:
            skipped += 1
            continue

        # Skip if correct_answer is too short to be useful
        if len(correct) < 30:
            skipped += 1
            continue

        logger.info("[%d/%d] Generating rejected for: %s", i, len(lessons), topic[:60])

        rejected = generate_rejected_response(
            topic=topic,
            ollama_url=args.ollama_url,
            model=args.model,
        )

        if not rejected:
            failed += 1
            continue

        # Skip if rejected is too short
        if len(rejected) < 30:
            logger.warning("Rejected too short (%d chars), skipping: %s", len(rejected), topic[:40])
            failed += 1
            continue

        # Skip if rejected is nearly identical to chosen (Jaccard > 0.8)
        chosen_words = set(correct.lower().split())
        rejected_words = set(rejected.lower().split())
        if chosen_words and rejected_words:
            overlap = len(chosen_words & rejected_words)
            union = len(chosen_words | rejected_words)
            if union > 0 and overlap / union > 0.8:
                logger.info("Skipping near-duplicate pair for: %s", topic[:40])
                skipped += 1
                continue

        # Write the pair
        pair = {
            "query": topic,
            "chosen": correct,
            "rejected": rejected,
            "timestamp": datetime.now().isoformat(),
            "source": "synthetic_base_model",
        }
        out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        out_f.flush()
        generated += 1

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)

    out_f.close()

    print(f"\n{'='*60}")
    print(f"Synthetic Pair Generation Results")
    print(f"{'='*60}")
    print(f"Total lessons processed:  {len(lessons)}")
    print(f"Pairs generated:          {generated}")
    print(f"Skipped (duplicate/short): {skipped}")
    print(f"Failed (LLM error):       {failed}")
    print(f"Output file:              {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
