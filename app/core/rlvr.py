"""RLVR — Reinforcement Learning from Verifiable Rewards (signal collection).

Stores ground-truth-style signals during normal Nova operation so the next
GRPO/RLVR fine-tune cycle has reward data without re-grading the trajectory.

Each signal has a type (`tool_correct`, `json_valid`, `math_correct`,
`claim_grounded`, `quiz_correct`, `code_passes_tests`) and a value in [0,1].
The signal is verifiable in the strict RLVR sense: it's derived from
deterministic checks (parser, executor, schema validator, retrieval lookup)
not LLM-graded judgment.

Public surface:
    record_signal(signal_type, value, *, query, response, evidence, conversation_id)
    query_signals(signal_type=None, since_iso=None, limit=1000, only_unconsumed=True)
    aggregate(since_iso=None) -> dict[str, dict]   # type -> {n, mean, p25, p75}
    mark_consumed(ids)                              # called by GRPO trainer
    export_grpo_jsonl(out_path, *, signal_types=None, min_value=None, limit=5000)

The exporter writes one JSON object per line:
    {"query": ..., "response": ..., "reward": <value>, "signal_type": ..., "id": ...}

A trainer can then map reward to advantage and run GRPO/DPO/SimPO on this.
Recording is fire-and-forget; failures log at debug level so production
chat is never blocked by signal-store hiccups.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

from app.database import get_db

logger = logging.getLogger(__name__)


# Hard allow-list — accepting arbitrary signal_type strings would let
# upstream code accidentally fragment the table with typos (`tool_correct` vs
# `tool-correct` vs `toolCorrect`).
SIGNAL_TYPES = frozenset({
    "tool_correct",       # tool dispatch returned without ToolError
    "json_valid",         # LLM JSON output parsed without fallback
    "math_correct",       # calculator/symbolic eval matched final answer
    "claim_grounded",     # claim_validator.validate_claims returned no strips
    "quiz_correct",       # heartbeat lesson quiz answered correctly
    "code_passes_tests",  # generated code execution exit-coded 0
    "schema_match",       # tool_use args validated against tool schema
})


@dataclass
class Signal:
    id: int
    signal_type: str
    signal_value: float
    query: str
    response: str
    evidence: str
    conversation_id: str | None
    consumed_for_training: bool
    created_at: str


def record_signal(
    signal_type: str,
    value: float,
    *,
    query: str = "",
    response: str = "",
    evidence: str = "",
    conversation_id: str | None = None,
) -> bool:
    """Store one verifiable reward signal. Returns True on insert.

    Bad inputs (unknown type, NaN/Inf value) are dropped silently — recording
    is a hot-path hook, not an exception source.
    """
    if signal_type not in SIGNAL_TYPES:
        logger.debug("[RLVR] unknown signal_type=%r — dropping", signal_type)
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v or v in (float("inf"), float("-inf")):
        return False
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    try:
        db = get_db()
        db.execute(
            "INSERT INTO verifiable_signals "
            "(conversation_id, query, response, signal_type, signal_value, evidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                (query or "")[:2000],
                (response or "")[:4000],
                signal_type,
                v,
                (evidence or "")[:2000],
            ),
        )
        return True
    except Exception as e:
        logger.debug("[RLVR] record_signal failed: %s", e)
        return False


def query_signals(
    signal_type: str | None = None,
    *,
    since_iso: str | None = None,
    limit: int = 1000,
    only_unconsumed: bool = True,
) -> list[Signal]:
    """Fetch signals for trainer consumption."""
    clauses: list[str] = []
    params: list[Any] = []
    if signal_type:
        if signal_type not in SIGNAL_TYPES:
            return []
        clauses.append("signal_type = ?")
        params.append(signal_type)
    if since_iso:
        clauses.append("created_at >= ?")
        params.append(since_iso)
    if only_unconsumed:
        clauses.append("consumed_for_training = 0")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT id, signal_type, signal_value, query, response, evidence, "
        "conversation_id, consumed_for_training, created_at "
        f"FROM verifiable_signals{where} ORDER BY id DESC LIMIT ?"
    )
    params.append(int(limit))
    try:
        rows = get_db().fetchall(sql, tuple(params))
    except Exception as e:
        logger.warning("[RLVR] query_signals failed: %s", e)
        return []
    return [
        Signal(
            id=row["id"],
            signal_type=row["signal_type"],
            signal_value=float(row["signal_value"]),
            query=row["query"] or "",
            response=row["response"] or "",
            evidence=row["evidence"] or "",
            conversation_id=row["conversation_id"],
            consumed_for_training=bool(row["consumed_for_training"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def aggregate(since_iso: str | None = None) -> dict[str, dict[str, float]]:
    """Per-signal-type roll-up: count, mean, P25, P75. Useful for dashboards
    and for the eval-harness regression flags."""
    clauses: list[str] = []
    params: list[Any] = []
    if since_iso:
        clauses.append("created_at >= ?")
        params.append(since_iso)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        rows = get_db().fetchall(
            f"SELECT signal_type, signal_value FROM verifiable_signals{where}",
            tuple(params),
        )
    except Exception as e:
        logger.warning("[RLVR] aggregate failed: %s", e)
        return {}

    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["signal_type"], []).append(float(row["signal_value"]))

    out: dict[str, dict[str, float]] = {}
    for stype, vals in buckets.items():
        if not vals:
            continue
        vs = sorted(vals)
        n = len(vs)
        mean = sum(vs) / n
        p25 = vs[max(0, int(n * 0.25))]
        p75 = vs[min(n - 1, int(n * 0.75))]
        out[stype] = {
            "n": float(n),
            "mean": mean,
            "p25": p25,
            "p75": p75,
        }
    return out


def mark_consumed(ids: Iterable[int]) -> int:
    """Mark signals as consumed by a GRPO/DPO trainer pass. Returns rowcount."""
    id_list = [int(i) for i in ids]
    if not id_list:
        return 0
    placeholders = ",".join("?" for _ in id_list)
    try:
        cursor = get_db().execute(
            f"UPDATE verifiable_signals SET consumed_for_training = 1 "
            f"WHERE id IN ({placeholders})",
            tuple(id_list),
        )
        return cursor.rowcount or 0
    except Exception as e:
        logger.warning("[RLVR] mark_consumed failed: %s", e)
        return 0


def export_grpo_jsonl(
    out_path: str,
    *,
    signal_types: list[str] | None = None,
    min_value: float | None = None,
    limit: int = 5000,
) -> int:
    """Write a JSONL file consumable by an RLVR/GRPO trainer. Returns row count.

    Each line:
        {"query":..., "response":..., "reward":..., "signal_type":..., "id":...}

    Trainer responsibility: bucket by signal_type, normalize rewards, sample
    pairs for DPO / advantages for GRPO.
    """
    types = list(signal_types) if signal_types else list(SIGNAL_TYPES)
    placeholders = ",".join("?" for _ in types)
    clauses = [f"signal_type IN ({placeholders})", "consumed_for_training = 0"]
    params: list[Any] = list(types)
    if min_value is not None:
        clauses.append("signal_value >= ?")
        params.append(float(min_value))
    sql = (
        "SELECT id, signal_type, signal_value, query, response, evidence "
        f"FROM verifiable_signals WHERE {' AND '.join(clauses)} "
        "ORDER BY id DESC LIMIT ?"
    )
    params.append(int(limit))
    try:
        rows = get_db().fetchall(sql, tuple(params))
    except Exception as e:
        logger.warning("[RLVR] export_grpo_jsonl read failed: %s", e)
        return 0

    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            obj = {
                "id": row["id"],
                "signal_type": row["signal_type"],
                "reward": float(row["signal_value"]),
                "query": row["query"] or "",
                "response": row["response"] or "",
                "evidence": row["evidence"] or "",
            }
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")
            written += 1
    logger.info("[RLVR] exported %d signals to %s", written, out_path)
    return written
