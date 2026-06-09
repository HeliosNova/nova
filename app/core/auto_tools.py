"""Auto-tool synthesis — Nova writes its own Python tools from capability gaps.

When `capability_gaps` accumulates 3+ similar failures (e.g. "needed to compute
a moving average", "needed to fetch GitHub repo metadata"), this module asks
the LLM to draft a Python function that would close the gap. If the function
parses and looks reasonable, it goes into `custom_tools` and becomes
immediately callable by the next tool loop — no rebuild required, because
DynamicTool reads from the DB on each invocation.

This closes the loop the user has been demanding:
  failure observed → gap recorded → tool synthesized → tool runs → gap fixed

Called from heartbeat_loop.py via check_type='auto_tool'.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Stopwords for clustering — same flavor as goal_deriver
_CLUSTER_STOPWORDS = frozenset({
    "what", "where", "when", "which", "show", "tell", "find", "search",
    "calculate", "compute", "the", "a", "an", "and", "or", "is", "are",
    "was", "were", "of", "in", "on", "for", "with", "by", "from", "this",
    "that", "today", "yesterday", "current", "latest", "recent", "now",
    "please", "give", "make", "list", "get", "want", "need", "should",
    "would", "could", "have", "had", "has", "you", "your", "we", "our",
    "they", "their", "them",
})


@dataclass
class GapCluster:
    keyword: str
    count: int
    sample_queries: list[str]
    sample_reasons: list[str]


def _cluster_gaps(rows: list[dict]) -> list[GapCluster]:
    """Group capability_gap rows by their most distinctive keyword."""
    by_keyword: dict[str, list[dict]] = {}
    for r in rows:
        q = (r["query"] or "").lower()
        words = [
            w for w in re.findall(r"\b[a-z][a-z0-9_-]{4,}\b", q)
            if w not in _CLUSTER_STOPWORDS
        ]
        if not words:
            continue
        # Use the longest word in the query as the cluster key
        key = max(words, key=len)
        by_keyword.setdefault(key, []).append(r)

    clusters: list[GapCluster] = []
    for key, items in by_keyword.items():
        if len(items) < 3:
            continue
        clusters.append(GapCluster(
            keyword=key,
            count=len(items),
            sample_queries=[i["query"][:200] for i in items[:5]],
            sample_reasons=[(i.get("reason") or "")[:120] for i in items[:3] if i.get("reason")],
        ))
    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters


_TOOL_PROPOSAL_PROMPT = """You are designing a Python tool to close a recurring capability gap in an AI assistant.

The assistant failed {count} similar queries clustered around the keyword: '{keyword}'

SAMPLE FAILED QUERIES:
{queries}

SAMPLE FAILURE REASONS (if any):
{reasons}

Your job: design ONE Python function that would have answered these queries. Output STRICT JSON with these fields:
{{
  "name": "snake_case_tool_name",          // <50 chars, lowercase+underscores only
  "description": "What the tool does in one sentence",
  "parameters": [
    {{"name": "param1", "type": "str", "description": "what it is"}}
    // 0 to 4 params; types: str, int, float, bool
  ],
  "code": "def run(param1=...):\\n    ...\\n    return <result>"
}}

CONSTRAINTS:
- The function MUST be named `run`.
- Code MUST be valid Python — it will be parsed with ast.parse.
- Keep code under 80 lines. Use the standard library freely (math, json, re, datetime, statistics, urllib, http, socket, os if needed).
- DO NOT use placeholders like "# implement here" or hardcoded fake data — write the real logic.
- DO NOT call other custom tools or undefined functions; the function must run standalone.
- If the gap is about web fetching, use `urllib.request.urlopen` or `httpx`.
- If the gap is about computation, do the math directly.
- The function must return a STRING (the answer the assistant would give).

Output ONLY the JSON object — no preamble, no explanations, no code fences."""


async def _propose_tool(cluster: GapCluster) -> dict | None:
    """Ask the LLM to design a tool for one cluster. Returns tool dict or None."""
    from app.core.llm import invoke_nothink

    prompt = _TOOL_PROPOSAL_PROMPT.format(
        keyword=cluster.keyword,
        count=cluster.count,
        queries="\n".join(f"  - {q}" for q in cluster.sample_queries),
        reasons="\n".join(f"  - {r}" for r in cluster.sample_reasons) or "  (no recorded reasons)",
    )

    try:
        text = await invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            json_prefix="{",
            max_tokens=1500,
            temperature=0.3,
        )
    except Exception as e:
        logger.warning("[AutoTool] LLM call failed for '%s': %s", cluster.keyword, e)
        return None

    text = (text or "").strip()
    if not text:
        return None
    # Strip code fences if the model added them anyway
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        proposal = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("[AutoTool] Invalid JSON for '%s': %s", cluster.keyword, e)
        return None

    if not isinstance(proposal, dict):
        return None
    return proposal


def _validate_proposal(proposal: dict) -> tuple[bool, str]:
    """Return (ok, reason). Caller should reject if not ok."""
    name = proposal.get("name") or ""
    desc = proposal.get("description") or ""
    code = proposal.get("code") or ""

    if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]{2,49}$", name):
        return False, f"bad name: {name!r}"
    if not isinstance(desc, str) or len(desc) < 10:
        return False, "missing/short description"
    if not isinstance(code, str) or len(code) < 20 or len(code) > 5000:
        return False, f"bad code length: {len(code) if isinstance(code, str) else 'N/A'}"

    # Code must parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} on line {e.lineno}"

    # Code must define a `run` function
    has_run = any(
        isinstance(n, ast.FunctionDef) and n.name == "run"
        for n in tree.body
    )
    if not has_run:
        return False, "no top-level run() function"

    # Parameters must be a list
    params = proposal.get("parameters", [])
    if not isinstance(params, list):
        return False, "parameters must be a list"

    # AST-walk blocked-import gate. Runtime sandbox in access_tiers already
    # blocks dangerous imports at execution time, but a proposal containing
    # `import os` gets PERSISTED to the custom_tools registry as valid before
    # the runtime check fires. Registering them looks legitimate in the UI
    # and could be promoted/aged differently. Reject up-front when any top-
    # level import refers to a package on the current tier's block-list,
    # plus dynamic __import__() calls (`importlib` is already in the list,
    # but __import__ is a builtin that bypasses the import-statement check).
    try:
        from app.core.access_tiers import get_blocked_imports
        blocked = get_blocked_imports()
    except Exception:
        blocked = set()

    if blocked:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = (alias.name or "").split(".", 1)[0]
                    if top in blocked:
                        return False, f"blocked import: {alias.name!r}"
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".", 1)[0] if node.module else ""
                if top and top in blocked:
                    return False, f"blocked import: from {node.module!r}"
            elif isinstance(node, ast.Call):
                # __import__("os") — dynamic import bypass
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "__import__":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        target = (node.args[0].value or "")
                        if isinstance(target, str):
                            top = target.split(".", 1)[0]
                            if top in blocked:
                                return False, f"blocked __import__: {target!r}"

    return True, "ok"


def _build_smoke_args(
    proposal: dict, cluster: GapCluster
) -> dict | None:
    """Build a plausible kwargs dict for smoke-testing the proposed tool.

    Strategy: use the cluster keyword for any obvious 'query/text/topic' style
    string param; pick small ints (5) and floats (1.0) for numerics; True for
    bools. Returns None if the parameter list looks unparseable.
    """
    params = proposal.get("parameters", [])
    if not isinstance(params, list):
        return None

    args: dict = {}
    string_keys = ("query", "text", "input", "topic", "name", "keyword",
                   "term", "subject", "entity", "entity_name", "search",
                   "q", "description", "title")
    sample_query = (cluster.sample_queries[0] if cluster.sample_queries
                    else cluster.keyword)

    for p in params:
        if not isinstance(p, dict):
            continue
        pname = (p.get("name") or "").strip()
        ptype = (p.get("type") or "str").lower()
        if not pname:
            continue
        if ptype in ("int", "integer"):
            args[pname] = 5
        elif ptype in ("float", "number"):
            args[pname] = 1.0
        elif ptype in ("bool", "boolean"):
            args[pname] = True
        elif ptype in ("list", "array"):
            args[pname] = [cluster.keyword]
        else:
            # string-like: use cluster keyword for keyword-style params,
            # otherwise the sample query
            args[pname] = (
                cluster.keyword if any(k in pname.lower() for k in string_keys[:8])
                else sample_query
            )
    return args


# Markers in tool output that mean "tool ran but did nothing useful" — same
# pattern that surfaced "Could not find Japan population" earlier.
_USELESS_OUTPUT_MARKERS = (
    "could not find", "unable to retrieve", "no information available",
    "no data available", "no results", "not found", "could not retrieve",
    "no information found", "failed to fetch", "failed to retrieve",
    "no result", "n/a", "unable to find", "[error",
)


async def _smoke_test_tool(
    db, name: str, proposal: dict, cluster: GapCluster
) -> tuple[bool, str]:
    """Execute the proposed tool once with synthesized args. Reject useless ones.

    Returns (ok, reason). Tool is treated as useful if it:
      - returns a string
      - longer than 25 chars (rules out "ok" / "[]" placeholder returns)
      - doesn't match a "could not find" / "no data" marker
    """
    from app.core.custom_tools import CustomToolStore, DynamicTool

    args = _build_smoke_args(proposal, cluster)
    if args is None:
        return False, "could not synthesize smoke args"

    # Insert provisionally so DynamicTool can load it
    store = CustomToolStore(db)
    code = proposal.get("code", "")
    desc = proposal.get("description", "")
    params_json = json.dumps(proposal.get("parameters", []))
    try:
        # Use a temp name to avoid colliding with the eventual real insert
        tmp_name = f"_smoke_{name}"
        # Remove any prior smoke leftover
        try:
            db.execute("DELETE FROM custom_tools WHERE name = ?", (tmp_name,))
        except Exception:
            pass
        db.execute(
            "INSERT INTO custom_tools (name, description, parameters, code) "
            "VALUES (?, ?, ?, ?)",
            (tmp_name, desc[:500], params_json, code),
        )
        record = store.get_tool(tmp_name)
        if not record:
            return False, "smoke insert failed"

        tool = DynamicTool(record, store)
        try:
            result = await tool.execute(**args)
        except TypeError as e:
            return False, f"smoke args mismatch: {e}"
        except Exception as e:
            return False, f"smoke raised: {e}"
        finally:
            try:
                db.execute("DELETE FROM custom_tools WHERE name = ?", (tmp_name,))
            except Exception:
                pass

        if not result.success:
            return False, f"smoke result.success=False: {(result.error or '')[:80]}"
        out = (result.output or "").strip()
        if len(out) < 25:
            return False, f"smoke output too short ({len(out)} chars): {out!r}"
        out_low = out.lower()
        for marker in _USELESS_OUTPUT_MARKERS:
            if marker in out_low:
                return False, f"smoke output contains useless marker: {marker!r}"
        return True, "ok"
    except Exception as e:
        # Cleanup just in case
        try:
            db.execute("DELETE FROM custom_tools WHERE name LIKE '_smoke_%'")
        except Exception:
            pass
        return False, f"smoke gate raised: {e}"


async def synthesize_tools_from_gaps(db, *, max_per_run: int = 3) -> dict:
    """Mine capability_gaps, propose+validate tools, store them in custom_tools.

    Returns: {"clusters_seen": N, "tools_created": M, "rejected": [...], "created_names": [...]}
    """
    from app.core.custom_tools import CustomToolStore

    # Pull recent unreviewed gaps
    rows = db.fetchall(
        "SELECT id, query, reason FROM capability_gaps "
        "WHERE reviewed = 0 AND created_at > datetime('now', '-30 days') "
        "LIMIT 500"
    )
    if not rows:
        return {"clusters_seen": 0, "tools_created": 0, "rejected": [], "created_names": []}

    clusters = _cluster_gaps([dict(r) for r in rows])
    if not clusters:
        return {"clusters_seen": 0, "tools_created": 0, "rejected": [], "created_names": []}

    store = CustomToolStore(db)
    existing_names = {t.name for t in store.list_tools()} if hasattr(store, "list_tools") else set()

    created: list[str] = []
    rejected: list[dict] = []

    for cluster in clusters[:max_per_run * 2]:  # over-generate, cap creates below
        if len(created) >= max_per_run:
            break

        # Skip if a tool already exists with a similar name (the keyword)
        if any(cluster.keyword in n for n in existing_names):
            continue

        proposal = await _propose_tool(cluster)
        if not proposal:
            rejected.append({"cluster": cluster.keyword, "reason": "LLM returned no proposal"})
            continue

        ok, reason = _validate_proposal(proposal)
        if not ok:
            rejected.append({
                "cluster": cluster.keyword,
                "reason": reason,
                "name": proposal.get("name"),
            })
            continue

        # Smoke test the proposed tool BEFORE persisting it. Catches the
        # "Could not find Japan population" pattern: tool runs but returns
        # uselessly. Tools that fail the smoke gate never reach the registry.
        smoke_ok, smoke_reason = await _smoke_test_tool(
            db, proposal["name"], proposal, cluster
        )
        if not smoke_ok:
            rejected.append({
                "cluster": cluster.keyword,
                "reason": f"smoke: {smoke_reason}",
                "name": proposal.get("name"),
            })
            logger.info(
                "[AutoTool] '%s' rejected by smoke gate: %s",
                proposal["name"], smoke_reason,
            )
            continue

        # Try to create
        params_json = json.dumps(proposal.get("parameters", []))
        tid = store.create_tool(
            name=proposal["name"],
            description=proposal["description"],
            parameters=params_json,
            code=proposal["code"],
        )
        if tid > 0:
            created.append(proposal["name"])
            existing_names.add(proposal["name"])
            logger.info(
                "[AutoTool] created '%s' from cluster '%s' (%d gaps)",
                proposal["name"], cluster.keyword, cluster.count,
            )

            # Hot-register the new tool with the live ToolRegistry so the
            # next think() call can use it without an app restart. This is
            # the actual "self-rebuild loop": gap → tool → callable.
            try:
                from app.core.brain import get_services
                from app.core.custom_tools import DynamicTool
                svc = get_services()
                if svc and getattr(svc, "tool_registry", None):
                    record = store.get_tool(proposal["name"])
                    if record:
                        svc.tool_registry.register(DynamicTool(record, store))
                        logger.info(
                            "[AutoTool] hot-registered '%s' with live registry",
                            proposal["name"],
                        )
            except Exception as e:
                logger.warning("[AutoTool] live registration failed: %s", e)

            # Mark the gaps in this cluster as reviewed so we don't keep
            # synthesizing from them
            try:
                db.execute(
                    "UPDATE capability_gaps SET reviewed = 1 "
                    "WHERE id IN (SELECT id FROM capability_gaps "
                    "WHERE LOWER(query) LIKE ? AND reviewed = 0 LIMIT 20)",
                    (f"%{cluster.keyword}%",),
                )
            except Exception as e:
                logger.warning("[AutoTool] mark-reviewed failed: %s", e)
        else:
            rejected.append({
                "cluster": cluster.keyword,
                "reason": "store.create_tool returned -1 (dup/limit/length)",
                "name": proposal.get("name"),
            })

    return {
        "clusters_seen": len(clusters),
        "tools_created": len(created),
        "rejected": rejected,
        "created_names": created,
    }


def prune_unused_tools(db, *, min_age_days: int = 3,
                        min_uses_for_quality_check: int = 5,
                        max_failure_rate: float = 0.7) -> dict:
    """Disable auto-tools that aren't earning their keep.

    Two-axis pruning:
      1. UNUSED: older than `min_age_days` + times_used == 0 → disable.
      2. BAD: times_used >= `min_uses_for_quality_check` AND
              success_rate < (1 - max_failure_rate) → disable.

    Disabling (not deleting) preserves the row for audit and lets the
    auto-tool synthesizer skip the same name on its next pass via the
    name-substring dedup gate.
    """
    # Pass 1: unused
    unused = db.fetchall(
        f"""SELECT id, name FROM custom_tools
            WHERE enabled = 1
              AND times_used = 0
              AND datetime(created_at) < datetime('now', '-{int(min_age_days)} days')"""
    )
    # Pass 2: bad — low success rate after enough samples
    min_success = 1.0 - float(max_failure_rate)
    bad = db.fetchall(
        """SELECT id, name, times_used, success_rate FROM custom_tools
            WHERE enabled = 1
              AND times_used >= ?
              AND success_rate < ?""",
        (int(min_uses_for_quality_check), min_success),
    )
    all_rows = list(unused) + list(bad)
    if not all_rows:
        return {"disabled": 0, "names": [], "unused": 0, "bad": 0}
    ids = list({r["id"] for r in all_rows})
    names = [r["name"] for r in all_rows]
    placeholders = ",".join("?" for _ in ids)
    db.execute(f"UPDATE custom_tools SET enabled = 0 WHERE id IN ({placeholders})", tuple(ids))
    logger.info(
        "[AutoTool] pruned %d tools (unused=%d bad=%d): %s",
        len(ids), len(unused), len(bad), ", ".join(names[:5]),
    )
    return {
        "disabled": len(ids),
        "names": names,
        "unused": len(unused),
        "bad": len(bad),
    }


def get_auto_tool_health(db) -> dict:
    """Return aggregate health stats for the custom_tools table.

    Surfaces "are auto-generated tools actually getting used?" which is the
    closure metric for the auto-tool synthesizer pipeline.
    """
    try:
        row = db.fetchone(
            "SELECT "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) AS enabled, "
            "  SUM(CASE WHEN times_used > 0 THEN 1 ELSE 0 END) AS used, "
            "  AVG(times_used) AS avg_uses, "
            "  AVG(success_rate) AS avg_success "
            "FROM custom_tools"
        )
        if not row:
            return {"total": 0, "enabled": 0, "used": 0, "avg_uses": 0.0, "avg_success": 0.0}
        return {
            "total": int(row["total"] or 0),
            "enabled": int(row["enabled"] or 0),
            "used": int(row["used"] or 0),
            "avg_uses": float(row["avg_uses"] or 0.0),
            "avg_success": float(row["avg_success"] or 0.0),
        }
    except Exception as e:
        logger.warning("get_auto_tool_health failed: %s", e)
        return {"total": 0, "enabled": 0, "used": 0, "avg_uses": 0.0, "avg_success": 0.0}


async def synthesize_and_log(db) -> str:
    """Monitor-friendly wrapper. Returns a one-paragraph summary string."""
    try:
        result = await synthesize_tools_from_gaps(db)
    except Exception as e:
        logger.exception("synthesize_tools_from_gaps failed")
        return f"AUTO-TOOL ERROR: {e}"

    if result["tools_created"] == 0:
        if result["clusters_seen"] == 0:
            return "AUTO-TOOL | no recent capability_gap clusters to synthesize from"
        return (
            f"AUTO-TOOL | scanned {result['clusters_seen']} clusters, "
            f"created 0 tools (all rejected: "
            f"{', '.join(r['reason'][:40] for r in result['rejected'][:3])})"
        )

    summary = (
        f"AUTO-TOOL | created {result['tools_created']} new tool(s) from "
        f"{result['clusters_seen']} capability gaps:\n"
    )
    for n in result["created_names"]:
        summary += f"  - {n}\n"
    if result["rejected"]:
        summary += f"  ({len(result['rejected'])} rejected)"
    return summary.strip()
