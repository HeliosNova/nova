"""Persistent agent workspace — scratchpads survive across sessions.

Each `AgentLoop.solve()` run produces findings (extracted facts) and
a final answer that today disappear when the run ends. If the user
asks a similar question 30 minutes or 30 days later, the loop starts
from zero, re-derives everything, re-spends tokens.

This module persists the workspace keyed by a *query signature* —
a normalized fingerprint of the query's substantive keywords. Future
runs whose signature collides with a stored workspace can:
  - Pre-load prior findings as the seed scratchpad (so the LLM sees
    "we already established X = 5")
  - See the prior answer (for comparison / refinement)
  - Track success/fail counts to know whether prior runs converged

Storage: `agent_workspace` table (created in migration 18).
Lookup is exact-signature only — fuzzy matching is left to the
caller (decompose+search the table) to keep this module simple.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9-]{2,}\b")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "of", "in", "on", "at", "to", "for", "with", "by", "as", "from", "this",
    "that", "what", "which", "when", "where", "how", "why", "you", "your",
    "we", "our", "they", "their", "i", "me", "my", "it", "its", "do", "does",
    "did", "have", "has", "had", "would", "could", "should", "will", "can",
    "just", "also", "than", "then", "more", "most", "less", "any", "all",
    "tell", "find", "show", "give", "make", "see", "look", "search", "get",
    "please", "thanks", "thank", "hey", "hi", "hello",
})


@dataclass
class WorkspaceEntry:
    id: int
    signature: str
    last_query: str
    findings: dict[str, str]
    last_answer: str
    run_count: int
    success_count: int
    fail_count: int
    failed_approaches: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.failed_approaches is None:
            self.failed_approaches = []


def query_signature(query: str, *, top_k: int = 6) -> str:
    """Build a stable signature from the substantive tokens of a query.

    Top-K most-distinctive tokens (length-sorted as a proxy for specificity),
    sorted alphabetically, joined by '|'. Two queries that ask for the same
    underlying facts in different phrasing collapse to the same signature.

    Returns empty string if the query has no substantive tokens.
    """
    if not query:
        return ""
    # Strip KAIROS WILL-MODULE framing if present — the boilerplate tokens
    # ("clarifying", "interpretation", "self-contained", etc.) would otherwise
    # contaminate the signature and prevent workspace retrieval across runs of
    # the same goal. Keep only the GOAL: text.
    if "=== END TASK ===" in query and "GOAL:" in query:
        _, _, after_marker = query.partition("=== END TASK ===")
        _, _, after_goal = after_marker.partition("GOAL:")
        if after_goal.strip():
            query = after_goal.strip()
    tokens = [
        t for t in _TOKEN_RE.findall(query.lower())
        if t not in _STOPWORDS and not t.isdigit()
    ]
    if not tokens:
        return ""
    # Dedupe preserving longest-first
    seen: set[str] = set()
    uniq: list[str] = []
    for t in sorted(tokens, key=len, reverse=True):
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= top_k:
            break
    return "|".join(sorted(uniq))


def _row_to_entry(row) -> WorkspaceEntry:
    findings: dict[str, str] = {}
    raw = row["findings_json"] if row["findings_json"] else ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                findings = {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            pass
    failed_approaches: list[str] = []
    try:
        fa_raw = row["failed_approaches_json"] if "failed_approaches_json" in row.keys() else None
    except Exception:
        fa_raw = None
    if fa_raw:
        try:
            parsed_fa = json.loads(fa_raw)
            if isinstance(parsed_fa, list):
                failed_approaches = [str(x) for x in parsed_fa[:10]]
        except (json.JSONDecodeError, TypeError):
            pass
    return WorkspaceEntry(
        id=int(row["id"]),
        signature=row["signature"] if "signature" in row.keys() else row["query_signature"],
        last_query=row["last_query"] or "",
        findings=findings,
        last_answer=row["last_answer"] or "",
        run_count=int(row["run_count"] or 0),
        success_count=int(row["success_count"] or 0),
        fail_count=int(row["fail_count"] or 0),
        failed_approaches=failed_approaches,
    )


def _ensure_failed_approaches_column(db) -> None:
    """Best-effort migration: add failed_approaches_json column if missing."""
    try:
        db.execute("ALTER TABLE agent_workspace ADD COLUMN failed_approaches_json TEXT")
    except Exception:
        pass  # Column already exists


def load_workspace(db, query: str) -> WorkspaceEntry | None:
    """Look up the persisted workspace for a query, if any."""
    sig = query_signature(query)
    if not sig:
        return None
    _ensure_failed_approaches_column(db)
    try:
        row = db.fetchone(
            "SELECT id, query_signature AS signature, last_query, findings_json, "
            "       last_answer, run_count, success_count, fail_count, "
            "       failed_approaches_json "
            "FROM agent_workspace WHERE query_signature = ? LIMIT 1",
            (sig,),
        )
    except Exception as e:
        logger.warning("[Workspace] load failed: %s", e)
        return None
    return _row_to_entry(row) if row else None


def save_workspace(
    db,
    *,
    query: str,
    findings: dict[str, str],
    answer: str,
    success: bool,
    failed_approaches: list[str] | None = None,
) -> int | None:
    """Upsert the workspace for this query. Returns row id or None."""
    sig = query_signature(query)
    if not sig:
        return None

    _ensure_failed_approaches_column(db)
    findings_json = json.dumps({k: str(v)[:600] for k, v in (findings or {}).items()})
    answer_clip = (answer or "")[:4000]
    fa_json = json.dumps([str(x)[:200] for x in (failed_approaches or [])][:10]) if failed_approaches else None

    try:
        existing = db.fetchone(
            "SELECT id, run_count, success_count, fail_count, findings_json "
            "FROM agent_workspace WHERE query_signature = ? LIMIT 1",
            (sig,),
        )
        if existing:
            # Merge: union prior + new findings, new overrides duplicates
            merged: dict[str, str] = {}
            try:
                prior_raw = existing["findings_json"] or ""
                if prior_raw:
                    prior = json.loads(prior_raw)
                    if isinstance(prior, dict):
                        merged.update({k: str(v) for k, v in prior.items()})
            except (json.JSONDecodeError, TypeError):
                pass
            merged.update({k: str(v)[:600] for k, v in (findings or {}).items()})
            # Cap merged size — prevent unbounded growth on repeated runs
            if len(merged) > 30:
                # Keep the 30 most-recent (relies on dict insertion order)
                merged = dict(list(merged.items())[-30:])
            findings_json = json.dumps(merged)

            new_runs = int(existing["run_count"] or 0) + 1
            new_success = int(existing["success_count"] or 0) + (1 if success else 0)
            new_fail = int(existing["fail_count"] or 0) + (0 if success else 1)
            # Update failed_approaches only if new ones provided. We don't merge
            # — the latest run's failed approaches are most relevant.
            if fa_json is not None:
                db.execute(
                    "UPDATE agent_workspace SET last_query=?, findings_json=?, "
                    "       last_answer=?, run_count=?, success_count=?, fail_count=?, "
                    "       failed_approaches_json=?, "
                    "       updated_at=CURRENT_TIMESTAMP "
                    "WHERE id=?",
                    (
                        query[:1000], findings_json, answer_clip,
                        new_runs, new_success, new_fail, fa_json, existing["id"],
                    ),
                )
            else:
                db.execute(
                    "UPDATE agent_workspace SET last_query=?, findings_json=?, "
                    "       last_answer=?, run_count=?, success_count=?, fail_count=?, "
                    "       updated_at=CURRENT_TIMESTAMP "
                    "WHERE id=?",
                    (
                        query[:1000], findings_json, answer_clip,
                        new_runs, new_success, new_fail, existing["id"],
                    ),
                )
            return int(existing["id"])

        cursor = db.execute(
            "INSERT INTO agent_workspace "
            "(query_signature, last_query, findings_json, last_answer, "
            " run_count, success_count, fail_count, failed_approaches_json) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (
                sig, query[:1000], findings_json, answer_clip,
                1 if success else 0, 0 if success else 1, fa_json,
            ),
        )
        return cursor.lastrowid
    except Exception as e:
        logger.warning("[Workspace] save failed: %s", e)
        return None


def hydrate_scratchpad(scratchpad, entry: WorkspaceEntry) -> int:
    """Inject prior findings into a fresh scratchpad. Returns count added.

    Existing keys on the scratchpad win (the active run is the source of
    truth). We only fill in what the active run hasn't established yet.
    """
    if not entry or not entry.findings:
        return 0
    added = 0
    for k, v in entry.findings.items():
        if k in scratchpad.findings:
            continue
        scratchpad.findings[f"prior:{k}"] = v
        added += 1
    return added
