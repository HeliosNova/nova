"""Personalized PageRank (PPR) over the kg_facts graph.

HippoRAG 2 style retrieval — when a query mentions entities present in the KG,
PPR finds facts about *related* entities reachable via subject/object edges,
not just facts whose text matches the query keywords. This captures multi-hop
reasoning (e.g. "tell me about Tim Cook" can surface facts about Apple, iOS,
and Steve Jobs even if the query never names them).

The KG is treated as an undirected graph: each (subject, predicate, object)
becomes an edge between the lowercase-normalized subject and object strings.
Edge weight is the fact's confidence (defaults to 0.5 when missing) so
high-confidence facts steer the walk more strongly.

The adjacency snapshot is cached for `_PPR_CACHE_TTL` seconds because rebuilding
from 5k+ facts is ~30 ms — short enough to refresh per-query when needed but
expensive enough to avoid doing in a hot path.

Public surface:
    extract_entities(query)  -> list[str]   - lowercase candidate entity strings
    compute_ppr(seeds, ...)  -> dict[str, float]  - entity -> ppr score
    score_facts(facts, ppr_scores) -> list[tuple[fact, score]]  - rank facts by entity overlap
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict

from app.config import config
from app.database import get_db

logger = logging.getLogger(__name__)


# Tunables (kept here rather than config until proven worth surfacing)
_PPR_CACHE_TTL = 300.0          # 5 min — rebuild adjacency on next call
_PPR_RESTART_PROB = 0.30        # alpha — probability of teleport to seeds
_PPR_MAX_ITER = 15              # iterations until convergence
_PPR_TOLERANCE = 1e-4           # L1 delta threshold for early stop
_PPR_MAX_NODES = 12000          # safety cap on graph size


_CACHE_LOCK = threading.Lock()
_ADJ_CACHE: dict[str, list[tuple[str, float]]] | None = None
_FACT_INDEX_CACHE: dict[tuple[str, str], list[int]] | None = None  # (subj,obj) -> fact ids
_CACHE_BUILT_AT: float = 0.0


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

# Match capitalized multi-word phrases (proper-noun candidates) and lowercase
# alpha tokens >=4 chars. Numbers and stopwords stripped downstream.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\b")
_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]{3,}\b")

_STOP_WORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "this", "that", "these", "those", "and", "but", "for", "with",
    "from", "into", "onto", "about", "between", "through", "after", "before",
    "above", "below", "during", "while", "would", "could", "should", "might",
    "have", "has", "had", "did", "does", "doing", "done", "are", "was", "were",
    "been", "being", "is", "am", "be", "any", "some", "many", "much", "more",
    "most", "every", "each", "either", "neither", "such", "rather", "quite",
    "very", "really", "often", "always", "never", "sometimes", "maybe",
    "perhaps", "probably", "actually", "currently", "exactly", "really",
    "just", "only", "also", "even", "still", "yet", "already", "hence",
    "therefore", "thus", "however", "moreover", "furthermore", "nevertheless",
    "tell", "show", "give", "make", "take", "want", "need", "find", "search",
    "look", "see", "know", "think", "explain", "describe", "list", "name",
    "compare", "contrast", "differ", "different", "similar", "between",
    "year", "years", "today", "tomorrow", "yesterday", "now", "currently",
    "version", "versions", "thing", "things", "stuff", "case", "cases",
    "fact", "facts", "value", "values", "time", "times",
})


def extract_entities(query: str, max_seeds: int = 6) -> list[str]:
    """Pull lowercase candidate entity strings from a query.

    Strategy: proper-noun phrases first (highest signal), then non-stopword
    content tokens of >=4 chars. Returns up to `max_seeds` distinct entries
    in priority order.
    """
    if not query:
        return []
    seeds: list[str] = []
    seen: set[str] = set()

    # Multi-word proper-noun phrases first
    for m in _PROPER_NOUN_RE.finditer(query):
        phrase = m.group(0).strip().lower()
        if phrase and phrase not in seen and phrase not in _STOP_WORDS:
            seen.add(phrase)
            seeds.append(phrase)
            if len(seeds) >= max_seeds:
                return seeds

    # Single content tokens
    for m in _WORD_RE.finditer(query.lower()):
        tok = m.group(0)
        if tok in _STOP_WORDS or tok in seen:
            continue
        seen.add(tok)
        seeds.append(tok)
        if len(seeds) >= max_seeds:
            break

    return seeds


# ---------------------------------------------------------------------------
# Adjacency cache
# ---------------------------------------------------------------------------

def _build_adjacency() -> tuple[dict[str, list[tuple[str, float]]], dict[tuple[str, str], list[int]]]:
    """Build (entity -> neighbors, (entity_pair -> fact_ids)) from kg_facts.

    Lower-cases entity strings so case differences in storage don't fragment
    the graph ("Apple" and "apple" are the same node). Edges are undirected;
    weight = confidence (defaults 0.5 if missing).
    """
    db = get_db()
    rows = db.fetchall(
        "SELECT id, subject, object, confidence FROM kg_facts "
        "WHERE valid_to IS NULL AND superseded_by IS NULL"
    )

    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    fact_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    nodes: set[str] = set()

    for row in rows:
        s = (row["subject"] or "").strip().lower()
        o = (row["object"] or "").strip().lower()
        if not s or not o or s == o:
            continue
        w = row["confidence"] if row["confidence"] is not None else 0.5
        adj[s].append((o, w))
        adj[o].append((s, w))
        # Symmetric key for fact lookup
        key = (s, o) if s <= o else (o, s)
        fact_index[key].append(row["id"])
        nodes.add(s)
        nodes.add(o)
        if len(nodes) > _PPR_MAX_NODES:
            logger.warning(
                "[PPR] node cap reached (%d), truncating graph build",
                _PPR_MAX_NODES,
            )
            break

    logger.info(
        "[PPR] adjacency built: %d nodes, %d edges (from %d facts)",
        len(nodes), sum(len(v) for v in adj.values()) // 2, len(rows),
    )
    return dict(adj), dict(fact_index)


def _get_adjacency() -> tuple[dict[str, list[tuple[str, float]]], dict[tuple[str, str], list[int]]]:
    """Return cached adjacency or rebuild if expired."""
    global _ADJ_CACHE, _FACT_INDEX_CACHE, _CACHE_BUILT_AT
    with _CACHE_LOCK:
        now = time.monotonic()
        if (
            _ADJ_CACHE is None
            or _FACT_INDEX_CACHE is None
            or now - _CACHE_BUILT_AT > _PPR_CACHE_TTL
        ):
            _ADJ_CACHE, _FACT_INDEX_CACHE = _build_adjacency()
            _CACHE_BUILT_AT = now
        return _ADJ_CACHE, _FACT_INDEX_CACHE


def invalidate_cache() -> None:
    """Force rebuild on next call. Use when KG mutations are large/important."""
    global _ADJ_CACHE, _FACT_INDEX_CACHE, _CACHE_BUILT_AT
    with _CACHE_LOCK:
        _ADJ_CACHE = None
        _FACT_INDEX_CACHE = None
        _CACHE_BUILT_AT = 0.0


# ---------------------------------------------------------------------------
# PPR computation
# ---------------------------------------------------------------------------

def compute_ppr(
    seeds: list[str],
    *,
    restart_prob: float = _PPR_RESTART_PROB,
    max_iter: int = _PPR_MAX_ITER,
    tolerance: float = _PPR_TOLERANCE,
    top_k: int = 100,
) -> dict[str, float]:
    """Personalized PageRank with restart distribution biased toward `seeds`.

    Standard iterative PPR: r_{t+1} = (1-alpha) * P^T r_t + alpha * s
    where P is the row-normalized weighted adjacency and s is the unit vector
    over seed entities.

    Returns the top_k highest-scoring entities as {entity: score}. Score sums
    to ~1.0 over all reachable nodes.
    """
    if not seeds:
        return {}

    adj, _ = _get_adjacency()
    if not adj:
        return {}

    # Map seeds onto graph nodes. A seed may not be present (query mentions
    # something not in the KG) — those contribute nothing. Seeds that match
    # share the restart mass equally.
    seed_nodes = [s for s in seeds if s in adj]
    if not seed_nodes:
        # No exact seed in graph — try substring match as fallback. This catches
        # cases like query="bitcoin" matching node "bitcoin price" in the KG.
        seed_nodes = []
        seed_set = set()
        for seed in seeds:
            for node in adj:
                if seed in node or node in seed:
                    if node not in seed_set:
                        seed_set.add(node)
                        seed_nodes.append(node)
                        if len(seed_nodes) >= 8:
                            break
            if len(seed_nodes) >= 8:
                break
    if not seed_nodes:
        return {}

    # Initial mass on seeds
    seed_mass = 1.0 / len(seed_nodes)
    rank: dict[str, float] = {n: seed_mass for n in seed_nodes}
    teleport: dict[str, float] = dict(rank)

    for it in range(max_iter):
        new_rank: dict[str, float] = defaultdict(float)
        # Random-walk component: spread current mass to neighbors, weighted by edge
        for node, mass in rank.items():
            neighbors = adj.get(node)
            if not neighbors:
                # Dangling node — mass evaporates back to teleport (handled below)
                new_rank[node] += (1.0 - restart_prob) * mass
                continue
            total_w = sum(w for _, w in neighbors)
            if total_w <= 0:
                continue
            spread = (1.0 - restart_prob) * mass
            for nb, w in neighbors:
                new_rank[nb] += spread * (w / total_w)

        # Teleport component
        for n, m in teleport.items():
            new_rank[n] += restart_prob * m

        # Convergence check
        delta = sum(abs(new_rank.get(k, 0.0) - rank.get(k, 0.0)) for k in set(new_rank) | set(rank))
        rank = dict(new_rank)
        if delta < tolerance:
            logger.debug("[PPR] converged at iter %d (delta=%.5f)", it + 1, delta)
            break

    # Return top-k entities by score
    top = sorted(rank.items(), key=lambda kv: -kv[1])[:top_k]
    return dict(top)


# ---------------------------------------------------------------------------
# Fact scoring
# ---------------------------------------------------------------------------

def rank_facts_by_ppr(
    fact_rows: list,
    seeds: list[str],
    *,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Score and rank facts by PPR score of their endpoints.

    Returns list of (fact_id, score) tuples in descending score order.
    A fact's score = max(ppr[subj], ppr[obj]) so a fact with one strong
    endpoint is preferred over one with two weak endpoints (avoids
    long-walk noise).
    """
    if not fact_rows or not seeds:
        return []

    ppr = compute_ppr(seeds, top_k=300)
    if not ppr:
        return []

    scored: list[tuple[int, float]] = []
    for row in fact_rows:
        s = (row["subject"] or "").strip().lower()
        o = (row["object"] or "").strip().lower()
        score = max(ppr.get(s, 0.0), ppr.get(o, 0.0))
        if score > 0:
            scored.append((row["id"], score))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def is_enabled() -> bool:
    """Check whether PPR retrieval is enabled via config flag."""
    return bool(getattr(config, "ENABLE_PPR_RETRIEVAL", False))
