#!/usr/bin/env python3
"""Export all training signal from Nova's DB as JSONL.

Extracts 7 signal types for DPO/ORPO/SFT fine-tuning:
  1. correction_dpo   — User corrections: bad → good response (DPO)
  2. reflexion_pos     — High-quality responses (score ≥ 0.8) as SFT positives
  3. reflexion_neg     — Low-quality responses (score < 0.4) as DPO negatives
  4. skill_procedure   — Successful multi-step tool chains as instruction-following
  5. reasoning_trace   — Reflexion critiques with reasoning as chain-of-thought
  6. lesson_knowledge  — Learned lessons as knowledge distillation
  7. tool_chain        — Multi-round tool sequences from conversation history

Usage:
    python scripts/export_training_signal.py --db data/nova.db --output training_export.jsonl
    python scripts/export_training_signal.py --db data/nova.db --output training_export.jsonl --types correction_dpo,reflexion_pos
    python scripts/export_training_signal.py --db data/nova.db --stats  # Show counts only

Format:
    Each line is a JSON object with:
    {
        "signal_type": "correction_dpo",       # one of the 7 types
        "prompt": "user query",                # the input/question
        "chosen": "good response",             # preferred output (DPO/SFT)
        "rejected": "bad response",            # dispreferred output (DPO only)
        "metadata": {
            "source_table": "lessons",
            "source_id": 42,
            "confidence": 0.9,
            "quality_score": 0.85,
            "tools_used": ["web_search"],
            "timestamp": "2026-04-15T10:30:00"
        }
    }

Recommended training recipes:
    - DPO:  correction_dpo + reflexion_neg (chosen/rejected pairs)
    - ORPO: correction_dpo + reflexion_pos (preference + positive signal)
    - SFT:  reflexion_pos + skill_procedure + reasoning_trace (supervised)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SafeDB


def _export_correction_dpo(db: SafeDB, training_data_path: str | None) -> list[dict]:
    """Export correction DPO pairs from training_data.jsonl + lessons table."""
    records = []

    # Source 1: Existing JSONL training data
    if training_data_path:
        p = Path(training_data_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        records.append({
                            "signal_type": "correction_dpo",
                            "prompt": entry.get("query", ""),
                            "chosen": entry.get("chosen", ""),
                            "rejected": entry.get("rejected", ""),
                            "metadata": {
                                "source_table": "training_data_jsonl",
                                "confidence": entry.get("confidence", 1.0),
                                "timestamp": entry.get("timestamp", ""),
                                "channel": entry.get("channel", "api"),
                            },
                        })
                    except json.JSONDecodeError:
                        continue

    # Source 2: Lessons with wrong_answer (implicit DPO pairs)
    rows = db.fetchall(
        "SELECT id, topic, wrong_answer, correct_answer, lesson_text, confidence, created_at "
        "FROM lessons WHERE wrong_answer IS NOT NULL AND wrong_answer != '' "
        "ORDER BY confidence DESC"
    )
    for row in rows:
        prompt = row["topic"]
        rejected = row["wrong_answer"]
        chosen = row["correct_answer"]
        if row["lesson_text"]:
            chosen = f"{chosen}\n\n(Note: {row['lesson_text']})"
        records.append({
            "signal_type": "correction_dpo",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "source_table": "lessons",
                "source_id": row["id"],
                "confidence": row["confidence"],
                "timestamp": row["created_at"] or "",
            },
        })

    return records


def _export_reflexion_positive(db: SafeDB) -> list[dict]:
    """Export high-quality responses as SFT positive examples."""
    rows = db.fetchall(
        "SELECT id, task_summary, reflection, quality_score, tools_used, revision_count, created_at "
        "FROM reflexions WHERE outcome = 'success' AND quality_score >= 0.8 "
        "ORDER BY quality_score DESC LIMIT 500"
    )
    records = []
    for row in rows:
        records.append({
            "signal_type": "reflexion_pos",
            "prompt": row["task_summary"],
            "chosen": row["reflection"],
            "rejected": "",
            "metadata": {
                "source_table": "reflexions",
                "source_id": row["id"],
                "quality_score": row["quality_score"],
                "tools_used": row["tools_used"].split(",") if row["tools_used"] else [],
                "revision_count": row["revision_count"],
                "timestamp": row["created_at"] or "",
            },
        })
    return records


def _export_reflexion_negative(db: SafeDB) -> list[dict]:
    """Export low-quality responses as DPO negative examples."""
    rows = db.fetchall(
        "SELECT id, task_summary, reflection, quality_score, tools_used, revision_count, created_at "
        "FROM reflexions WHERE outcome = 'failure' AND quality_score < 0.4 "
        "ORDER BY quality_score ASC LIMIT 500"
    )
    records = []
    for row in rows:
        records.append({
            "signal_type": "reflexion_neg",
            "prompt": row["task_summary"],
            "chosen": "",  # Will need to be paired with a positive response
            "rejected": row["reflection"],
            "metadata": {
                "source_table": "reflexions",
                "source_id": row["id"],
                "quality_score": row["quality_score"],
                "tools_used": row["tools_used"].split(",") if row["tools_used"] else [],
                "revision_count": row["revision_count"],
                "timestamp": row["created_at"] or "",
            },
        })
    return records


def _export_skill_procedures(db: SafeDB) -> list[dict]:
    """Export successful skill chains as instruction-following examples."""
    # 'source' column may not exist in older schemas — use a safe query
    try:
        rows = db.fetchall(
            "SELECT id, name, trigger_pattern, steps, answer_template, success_rate, "
            "times_used, source, created_at "
            "FROM skills WHERE enabled = 1 AND success_rate >= 0.7 AND times_used >= 2 "
            "ORDER BY success_rate DESC"
        )
    except Exception:
        rows = db.fetchall(
            "SELECT id, name, trigger_pattern, steps, answer_template, success_rate, "
            "times_used, created_at "
            "FROM skills WHERE enabled = 1 AND success_rate >= 0.7 AND times_used >= 2 "
            "ORDER BY success_rate DESC"
        )
    records = []
    for row in rows:
        try:
            steps = json.loads(row["steps"]) if row["steps"] else []
        except json.JSONDecodeError:
            steps = []

        # Format steps as a readable procedure
        step_lines = []
        for i, step in enumerate(steps, 1):
            tool = step.get("tool", "unknown")
            args = step.get("args_template", {})
            step_lines.append(f"Step {i}: Use {tool} with {json.dumps(args)}")

        procedure = "\n".join(step_lines)
        if row["answer_template"]:
            procedure += f"\nAnswer: {row['answer_template']}"

        records.append({
            "signal_type": "skill_procedure",
            "prompt": f"When asked about: {row['name']} (trigger: {row['trigger_pattern']})",
            "chosen": procedure,
            "rejected": "",
            "metadata": {
                "source_table": "skills",
                "source_id": row["id"],
                "success_rate": row["success_rate"],
                "times_used": row["times_used"],
                "source": row["source"] if "source" in row.keys() else "unknown",
                "timestamp": row["created_at"] or "",
            },
        })
    return records


def _export_reasoning_traces(db: SafeDB) -> list[dict]:
    """Export reflexion critiques as reasoning/chain-of-thought examples."""
    rows = db.fetchall(
        "SELECT id, task_summary, reflection, quality_score, tools_used, revision_count, created_at "
        "FROM reflexions WHERE reflection IS NOT NULL AND length(reflection) > 100 "
        "ORDER BY quality_score DESC LIMIT 300"
    )
    records = []
    for row in rows:
        # Build a reasoning trace from the critique
        reasoning = (
            f"Query: {row['task_summary']}\n"
            f"Tools used: {row['tools_used'] or 'none'}\n"
            f"Rounds needed: {row['revision_count']}\n"
            f"Quality assessment: {row['quality_score']:.2f}/1.0\n"
            f"Analysis: {row['reflection']}"
        )
        records.append({
            "signal_type": "reasoning_trace",
            "prompt": row["task_summary"],
            "chosen": reasoning,
            "rejected": "",
            "metadata": {
                "source_table": "reflexions",
                "source_id": row["id"],
                "quality_score": row["quality_score"],
                "tools_used": row["tools_used"].split(",") if row["tools_used"] else [],
                "revision_count": row["revision_count"],
                "timestamp": row["created_at"] or "",
            },
        })
    return records


def _export_lesson_knowledge(db: SafeDB) -> list[dict]:
    """Export learned lessons as knowledge distillation examples."""
    rows = db.fetchall(
        "SELECT id, topic, correct_answer, lesson_text, confidence, times_helpful, created_at "
        "FROM lessons WHERE confidence >= 0.6 "
        "ORDER BY confidence DESC, times_helpful DESC"
    )
    records = []
    for row in rows:
        answer = row["correct_answer"]
        if row["lesson_text"]:
            answer += f"\n\nLearned insight: {row['lesson_text']}"
        records.append({
            "signal_type": "lesson_knowledge",
            "prompt": row["topic"],
            "chosen": answer,
            "rejected": "",
            "metadata": {
                "source_table": "lessons",
                "source_id": row["id"],
                "confidence": row["confidence"],
                "times_helpful": row["times_helpful"],
                "timestamp": row["created_at"] or "",
            },
        })
    return records


def _export_tool_chains(db: SafeDB) -> list[dict]:
    """Export multi-round tool sequences from conversation history."""
    # Find conversations with tool messages (multi-step tool chains)
    conv_rows = db.fetchall(
        "SELECT DISTINCT conversation_id FROM messages WHERE role = 'tool' "
        "GROUP BY conversation_id HAVING count(*) >= 2 "
        "ORDER BY max(created_at) DESC LIMIT 100"
    )
    records = []
    for conv_row in conv_rows:
        conv_id = conv_row["conversation_id"]
        messages = db.fetchall(
            "SELECT role, content, tool_name, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        )
        if len(messages) < 3:
            continue

        # Extract the query (first user message) and tool chain
        query = ""
        tool_steps = []
        final_answer = ""
        for msg in messages:
            if msg["role"] == "user" and not query:
                query = msg["content"]
            elif msg["role"] == "tool":
                tool_steps.append({
                    "tool": msg["tool_name"] or "unknown",
                    "result": (msg["content"] or "")[:200],
                })
            elif msg["role"] == "assistant" and tool_steps:
                final_answer = msg["content"] or ""

        if not query or not tool_steps or not final_answer:
            continue

        chain_description = "\n".join(
            f"  {i}. {s['tool']}: {s['result'][:100]}..."
            for i, s in enumerate(tool_steps, 1)
        )
        chosen = f"Tool chain:\n{chain_description}\n\nFinal answer:\n{final_answer[:500]}"

        records.append({
            "signal_type": "tool_chain",
            "prompt": query[:500],
            "chosen": chosen,
            "rejected": "",
            "metadata": {
                "source_table": "messages",
                "conversation_id": conv_id,
                "tool_count": len(tool_steps),
                "tools_used": list({s["tool"] for s in tool_steps}),
                "timestamp": messages[-1]["created_at"] or "",
            },
        })
    return records


EXPORTERS = {
    "correction_dpo": _export_correction_dpo,
    "reflexion_pos": _export_reflexion_positive,
    "reflexion_neg": _export_reflexion_negative,
    "skill_procedure": _export_skill_procedures,
    "reasoning_trace": _export_reasoning_traces,
    "lesson_knowledge": _export_lesson_knowledge,
    "tool_chain": _export_tool_chains,
}


def main():
    parser = argparse.ArgumentParser(description="Export Nova training signal as JSONL")
    parser.add_argument("--db", default="/data/nova.db", help="Path to Nova SQLite database")
    parser.add_argument("--output", "-o", default="training_export.jsonl", help="Output JSONL path")
    parser.add_argument("--training-data", default=None,
                        help="Path to existing training_data.jsonl (default: from config)")
    parser.add_argument("--types", default=None,
                        help="Comma-separated signal types to export (default: all)")
    parser.add_argument("--stats", action="store_true", help="Show counts only, don't write")
    args = parser.parse_args()

    db = SafeDB(args.db)

    # Determine training data path
    training_data_path = args.training_data
    if training_data_path is None:
        try:
            from app.config import config
            training_data_path = config.TRAINING_DATA_PATH
        except Exception:
            training_data_path = "/data/training_data.jsonl"

    # Select signal types
    if args.types:
        selected = [t.strip() for t in args.types.split(",")]
        invalid = [t for t in selected if t not in EXPORTERS]
        if invalid:
            print(f"Error: unknown signal types: {invalid}")
            print(f"Available: {list(EXPORTERS.keys())}")
            sys.exit(1)
    else:
        selected = list(EXPORTERS.keys())

    # Export each signal type
    all_records = []
    print(f"Exporting training signal from {args.db}")
    print(f"Signal types: {', '.join(selected)}")
    print()

    for signal_type in selected:
        exporter = EXPORTERS[signal_type]
        if signal_type == "correction_dpo":
            records = exporter(db, training_data_path)
        else:
            records = exporter(db)
        all_records.extend(records)
        print(f"  {signal_type}: {len(records)} records")

    print(f"\nTotal: {len(all_records)} records")

    if args.stats:
        # Show breakdown
        by_type = {}
        for r in all_records:
            t = r["signal_type"]
            by_type[t] = by_type.get(t, 0) + 1

        print("\nBreakdown:")
        for t, count in sorted(by_type.items()):
            print(f"  {t}: {count}")

        # Training recipe recommendations
        dpo_count = by_type.get("correction_dpo", 0) + by_type.get("reflexion_neg", 0)
        sft_count = by_type.get("reflexion_pos", 0) + by_type.get("skill_procedure", 0) + by_type.get("reasoning_trace", 0)
        print(f"\nRecommended recipes:")
        print(f"  DPO/ORPO: {dpo_count} pairs (correction_dpo + reflexion_neg)")
        print(f"  SFT: {sft_count} examples (reflexion_pos + skill_procedure + reasoning_trace)")
        return

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWritten to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show recipe recommendations
    by_type = {}
    for r in all_records:
        t = r["signal_type"]
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\nUsage with training frameworks:")
    print(f"  Unsloth DPO:  Filter signal_type in [correction_dpo, reflexion_neg]")
    print(f"  Unsloth SFT:  Filter signal_type in [reflexion_pos, skill_procedure, reasoning_trace]")
    print(f"  Axolotl DPO:  Use prompt/chosen/rejected fields directly")
    print(f"  See README at scripts/TRAINING_DATA_README.md for full guide")


if __name__ == "__main__":
    main()
