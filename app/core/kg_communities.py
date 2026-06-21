"""KG community layer (GraphRAG-style global synthesis).

The monitor-fed knowledge graph is broad but shallow — ~1.4 facts per entity, a
pile of disconnected triples that can answer "what do you know about X" (local)
but not "what are the big themes across everything you've watched" (global).

This builds the actual entity graph from current kg_facts, detects communities
(densely-connected entity clusters) with networkx Louvain, and writes one LLM
summary per community. Global/thematic queries then retrieve those summaries
instead of scanning thousands of isolated facts. Per Microsoft GraphRAG
(arXiv:2404.16130): community summaries lift comprehensiveness on whole-corpus
questions 50-70% over flat retrieval.

Runs in the dream REM phase (nightly). Degrades to a no-op if networkx is
unavailable or the graph is too sparse to cluster.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_communities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    entities TEXT NOT NULL DEFAULT '[]',
    entity_count INTEGER DEFAULT 0,
    fact_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_kg_comm_valid ON kg_communities(valid_to);
"""

# Queries that want the global/thematic view rather than a specific fact.
_GLOBAL_QUERY_RE = re.compile(
    r"\b(themes?|trends?|big picture|overview|landscape|what'?s\s+happening|"
    r"across (?:all|the|your)|in general|broadly|summar(?:y|ize|ise)|"
    r"main (?:topics?|areas?|developments?)|key (?:topics?|themes?|developments?)|"
    r"what have you (?:learned|seen|noticed)|recent(?:ly)?\s+(?:across|in))\b",
    re.IGNORECASE,
)

_MIN_COMMUNITY_SIZE = 4      # entities; smaller clusters aren't a "theme"
_MAX_COMMUNITIES = 20        # cap LLM summary calls per dream cycle
_MAX_FACTS_PER_SUMMARY = 40  # context budget per community summary


def is_global_query(query: str) -> bool:
    """True if the query asks for a thematic/whole-corpus view."""
    return bool(query) and bool(_GLOBAL_QUERY_RE.search(query))


def ensure_schema(db) -> None:
    for stmt in _SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                db.execute(stmt)
            except Exception as e:  # pragma: no cover
                logger.debug("kg_communities schema stmt failed: %s", e)


def detect_communities(facts: list[dict], min_size: int = _MIN_COMMUNITY_SIZE) -> list[set[str]]:
    """Build an entity graph from facts and return communities >= min_size.

    Nodes are subjects/objects; an edge connects the endpoints of each fact,
    weighted by how often the pair co-occurs (capped). Uses networkx Louvain.
    Returns [] if networkx is missing or the graph is too small to cluster.
    """
    try:
        import networkx as nx
    except Exception:
        logger.info("[kg-communities] networkx unavailable — skipping")
        return []

    edge_w: dict[tuple[str, str], float] = defaultdict(float)
    for f in facts:
        s = (f.get("subject") or "").strip()
        o = (f.get("object") or "").strip()
        if not s or not o or s.lower() == o.lower():
            continue
        key = tuple(sorted((s, o), key=str.lower))
        edge_w[key] += float(f.get("confidence") or 0.8)

    if len(edge_w) < min_size:
        return []

    g = nx.Graph()
    for (a, b), w in edge_w.items():
        g.add_edge(a, b, weight=min(w, 5.0))
    if g.number_of_nodes() < min_size:
        return []

    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(g, weight="weight", seed=42)
    except Exception:
        # Older networkx: fall back to greedy modularity.
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(g, weight="weight")
        except Exception as e:
            logger.warning("[kg-communities] community detection failed: %s", e)
            return []

    out = [set(c) for c in communities if len(c) >= min_size]
    out.sort(key=len, reverse=True)
    return out


async def _summarize_community(entities: set[str], facts: list[dict]) -> dict | None:
    """LLM summary of one community: {title, summary}. None on failure."""
    from app.core import llm

    member = {e.lower() for e in entities}
    rel = [
        f"{f['subject']} {f['predicate'].replace('_', ' ')} {f['object']}"
        for f in facts
        if (f.get("subject", "").lower() in member or f.get("object", "").lower() in member)
    ][:_MAX_FACTS_PER_SUMMARY]
    if len(rel) < 3:
        return None

    prompt = (
        "These facts form one cluster of a knowledge graph built from news and "
        "research monitors. Write a SHORT thematic summary.\n\n"
        "Facts:\n" + "\n".join(f"- {r}" for r in rel) + "\n\n"
        'Respond with JSON: {"title": "3-6 word theme", "summary": "2-3 '
        'sentences on what this cluster is about and why it matters"}. '
        "No preamble."
    )
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_prefix="{", max_tokens=220, temperature=0.2,
        )
        obj = llm.extract_json_object(raw)
        if not obj or not obj.get("summary"):
            return None
        return {
            "title": str(obj.get("title") or "Untitled theme")[:120],
            "summary": str(obj["summary"])[:1000],
        }
    except Exception as e:
        logger.debug("[kg-communities] summary gen failed: %s", e)
        return None


async def build_and_store(db, *, max_communities: int = _MAX_COMMUNITIES) -> int:
    """Full pass: detect communities, summarize the largest, replace the stored
    set. Returns the number of community summaries written."""
    ensure_schema(db)
    rows = db.fetchall(
        "SELECT subject, predicate, object, confidence FROM kg_facts WHERE valid_to IS NULL"
    )
    facts = [dict(r) for r in rows]
    if len(facts) < _MIN_COMMUNITY_SIZE:
        return 0

    communities = detect_communities(facts)
    if not communities:
        logger.info("[kg-communities] no communities >= %d entities (graph too sparse)", _MIN_COMMUNITY_SIZE)
        return 0

    summaries = []
    for comm in communities[:max_communities]:
        s = await _summarize_community(comm, facts)
        if s:
            summaries.append((s, comm))

    if not summaries:
        return 0

    # Replace the previous generation atomically.
    with db.transaction() as tx:
        tx.execute("DELETE FROM kg_communities")
        for s, comm in summaries:
            tx.execute(
                "INSERT INTO kg_communities (title, summary, entities, entity_count, fact_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (s["title"], s["summary"], json.dumps(sorted(comm)[:50]), len(comm), 0),
            )
    logger.info("[kg-communities] wrote %d community summaries (from %d communities)",
                len(summaries), len(communities))
    return len(summaries)


def get_relevant_communities(db, query: str, limit: int = 3) -> list[dict]:
    """Retrieve community summaries relevant to a global query (keyword overlap
    on title+summary+entities). Cheap; no embedder call."""
    ensure_schema(db)
    rows = db.fetchall(
        "SELECT title, summary, entities, entity_count FROM kg_communities WHERE valid_to IS NULL"
    )
    if not rows:
        return []
    q_tokens = {t for t in re.findall(r"[a-z][a-z0-9]{2,}", (query or "").lower())}
    scored = []
    for r in rows:
        blob = (r["title"] + " " + r["summary"] + " " + (r["entities"] or "")).lower()
        toks = set(re.findall(r"[a-z][a-z0-9]{2,}", blob))
        overlap = len(q_tokens & toks)
        scored.append((overlap, dict(r)))
    scored.sort(key=lambda x: (-x[0], -x[1]["entity_count"]))
    # If the query has no overlap with any community, still return the largest
    # few — a bare "what are the main themes?" should surface the top clusters.
    top = [d for ov, d in scored if ov > 0][:limit]
    if not top:
        top = [d for _, d in scored[:limit]]
    return top


def format_for_prompt(communities: list[dict]) -> str:
    if not communities:
        return ""
    lines = ["## Knowledge themes (from your monitors)"]
    for c in communities:
        lines.append(f"- **{c['title']}** ({c['entity_count']} entities): {c['summary']}")
    return "\n".join(lines)
