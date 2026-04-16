#!/usr/bin/env python3
"""Idempotent skill corpus migration — audit cleanup pass.

Applies three classes of fix to the production skills DB:

1. Pattern fixes (regex-literal bugs, anchor-brittleness, capture groups)
   ids: 33, 36, 40, 58, 60, 61, 64, 74, 76, 77, 78, 82

2. Dedup deletions
   C1 (crypto price): keep id=33  → delete ids 28, 50, 66, 73, 75, 84, 90, 95
   C2 (lang compare): keep id=77  → delete ids 72, 89, 91, 92, 93
   C3 (Ethereum):     keep id=76  → delete ids 79, 81
   C4 (AI research):  keep ids 39, 78 → delete ids 37, 38, 80
   C5 (Fed rate):     keep id=64  → delete id=52
   C6 (pseudoscience dup): keep id=86 → delete id=88

Before count: 68 skills
Expected after count: 48 skills (20 deletions)

Safe to re-run: all operations are guarded by current-state checks.

Usage:
    python scripts/migrate_skills.py [--db /path/to/nova.db] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("migrate_skills")

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each entry: (id, description, new_trigger, new_steps_or_None, new_answer_or_sentinel)
# new_steps=None → keep existing steps
# new_answer=KEEP → keep existing answer_template
KEEP = object()  # sentinel: don't change answer_template

PATTERN_FIXES: list[tuple] = [
    # id=33: C1 canonical — add \b anchoring, (?i) flag, expand coin list
    (
        33,
        "C1 canonical: add \\b anchoring + (?i), expand altcoins",
        r"(?i)\b(?:crypto(?:currency)?|bitcoin|btc|ethereum|eth|solana|sol"
        r"|xrp|bnb|cardano|ada|dogecoin|doge)\b.*\b(?:price[sd]?|market"
        r"|movement|value|status|worth)\b",
        None,
        KEEP,
    ),
    # id=36: anchor-brittle + broken {date} in step args
    # Each arm of the alternation now self-contains its trailing whitespace so there
    # is no gap before the optional adjective (big|major).
    (
        36,
        "anchor-brittle + broken {date} + missing space before adjective in 2nd arm",
        r"(?i)(?:what|how)\s+(?:is|are)\s+(?:the\s+)?markets?\s+(?:doing|moving|performing)?"
        r"(?:\s+(?:today|right\s+now|now))?|"
        r"(?:any\s+|what\s+(?:are|were)\s+(?:the\s+)?)(?:(?:big|major)\s+)?"
        r"(?:moves?|updates?|news)\s+in\s+(?:the\s+)?markets?",
        [
            {
                "tool": "web_search",
                "args_template": {"query": "stock market news today market movements"},
                "output_key": "market_news",
            },
            {
                "tool": "web_search",
                "args_template": {"query": "S&P 500 Nasdaq Dow Jones today"},
                "output_key": "index_data",
            },
        ],
        KEEP,
    ),
    # id=40: regex-literal [a-z0-9\s]+, the? bug
    # Use bounded .{} instead of (?:\w+\s+){N} to avoid nested-quantifier ReDoS flag.
    (
        40,
        "regex-literal [a-z0-9\\s]+, the? → bounded .{} pattern (no nested quantifiers)",
        r"(?i)\b(?:search|find|check)\b.{0,20}\b(?:latest|newest|current)\b.{1,40}\b(?:model|version)\b",
        None,
        KEEP,
    ),
    # id=58: anchor-brittle + broken {entity} (unnamed group 1 → add named group)
    (
        58,
        "anchor-brittle (^...$) + broken {entity} → named group + relaxed anchor",
        r"(?i)what\s+do\s+you\s+know\s+about\s+(?P<entity>.+?)"
        r"(?:\s+(?:from|in)\s+(?:your\s+)?(?:knowledge\s+graph?|memory))?$",
        None,
        KEEP,
    ),
    # id=60: anchor-brittle + broken {gold_value}/{silver_value} in calculator step
    (
        60,
        "anchor-brittle (^...\\s*$) + broken calculator args → drop broken step",
        r"(?i)(?:compare|difference\s+between|ratio\s+of)\s+"
        r"(?:gold\s+(?:and|vs\.?|versus)\s+silver|silver\s+(?:and|vs\.?|versus)\s+gold"
        r"|precious\s+metals?)",
        [
            {
                "tool": "web_search",
                "args_template": {"query": "current gold price per ounce USD"},
                "output_key": "gold_price_result",
            },
            {
                "tool": "web_search",
                "args_template": {"query": "current silver price per ounce USD"},
                "output_key": "silver_price_result",
            },
            # Calculator step removed — {gold_value}/{silver_value} have no structured
            # extraction path from free-text search results. LLM synthesises from above.
        ],
        KEEP,
    ),
    # id=61: anchor-brittle + broken {commodity} (unnamed group → named group)
    (
        61,
        "anchor-brittle (^...$) + broken {commodity} → named group",
        r"(?i)(?:what\s+is|what's)\s+(?P<commodity>gold|silver|copper|oil|bitcoin)"
        r"\s+(?:trading\s+at|price|worth|priced)(?:\s+(?:right\s+now|today|currently|now))?",
        None,
        KEEP,
    ),
    # id=64: C5 canonical — add (?i) flag
    (
        64,
        "C5 canonical: add (?i) flag",
        r"(?i)(?:current|latest)\s+(?:federal\s+)?(?:funds\s+rate|interest\s+rate)"
        r"(?:\s+and(?:\s+next)?\s+FOMC\s+meeting)?|(?:FOMC\s+meeting|Federal\s+Reserve\s+schedule)",
        None,
        KEEP,
    ),
    # id=74: regex-literal [timeframe] + broken browser/benchmark steps
    (
        74,
        "regex-literal [timeframe] + broken step args → simplified",
        r"(?i)(?:latest|new|recent)\s+AI\s+model\s+(?:releases?|launches?|announcements?)"
        r"(?:\s+(?:this\s+week|this\s+month|recently|in\s+\d{4}))?",
        [
            {
                "tool": "web_search",
                "args_template": {"query": "{query}"},
                "output_key": "search_result",
            },
            # Steps 1 (browser navigate {result_url_from_search}) and
            # 2 (benchmark search with {specific_model_names} {timeframe}) removed —
            # undefined placeholders; URL extraction from search results not supported.
        ],
        KEEP,
    ),
    # id=76: C3 canonical — fix case-sensitive literal phrase trigger, drop broken http_fetch steps
    (
        76,
        "C3 canonical: fix case-sensitive literal trigger + drop broken http_fetch steps",
        r"(?i)\bethereum\b.*\b(?:price|market|blockchain|knowledge|data|info(?:rmation)?)\b"
        r"|what\s+do\s+you\s+know\s+about\s+ethereum",
        [
            {
                "tool": "memory_search",
                "args_template": {"query": "{query}"},
                "output_key": "result",
            },
            {
                "tool": "web_search",
                "args_template": {"query": "{query} price market cap CoinGecko"},
                "output_key": "search_result",
            },
            {
                "tool": "knowledge_search",
                "args_template": {"query": "{query} blockchain smart contracts"},
                "output_key": "knowledge_result",
            },
            # http_fetch steps removed — used raw {query} in CoinGecko URL path,
            # producing invalid URLs like /coins/Ethereum price. Web search suffices.
        ],
        KEEP,
    ),
    # id=77: C2 canonical — add (?i), expand language list, fix broken answer template refs
    # Use bounded .{} instead of .*? to avoid double-wildcard ReDoS flag.
    # Drop web\s*API (has \s*) → web API|webAPI.
    (
        77,
        "C2 canonical: add (?i), expand langs, bounded .{} wildcards, fix answer template",
        r"(?i)(?:Rust|Go|Python|Node\.js|JavaScript|Java|C\+\+|TypeScript)"
        r"\b.{0,60}\b(?:vs|versus|and).{0,60}\b"
        r"(?:web API|webAPI|backend|server|framework|performance|benchmark)",
        None,
        # Fix answer template: {query1} and {query2} are undefined — replace with {query}
        # The LLM will fill in language names from context.
        "| Category | Language A | Language B |\n"
        "| :--- | :--- | :--- |\n"
        "| **Raw Performance** | {perf_result} | |\n"
        "| **Ecosystem** | {ecosystem_result} | |\n\n"
        "Based on search results for: {query}",
    ),
    # id=78: C4 canonical #2 — add (?i), broaden temporal alternatives
    (
        78,
        "C4 canonical #2: add (?i) + broaden to this month",
        r"(?i)(?:latest|newest|recent)\s+AI\s+model\s+releases?"
        r"(?:\s+(?:this\s+week|this\s+month|recently))?",
        None,
        KEEP,
    ),
    # id=82: regex-literal [year] + broken {year} in step args
    (
        82,
        "regex-literal [year] + broken {year} in step args",
        r"(?i)(?:best investment opportunities?(?:\s+20\d{2})?\s+market\s+trends?"
        r"|what\s+(?:to|should\s+I)\s+invest\s+in(?:\s+20\d{2})?"
        r"|current\s+affairs\s+investment\s+advice)",
        [
            {
                "tool": "web_search",
                "args_template": {"query": "{query} current market trends"},
                "output_key": "search_result_1",
            },
            {
                "tool": "knowledge_search",
                "args_template": {"query": "{query} investment market trends"},
                "output_key": "knowledge_result",
            },
            {
                "tool": "memory_search",
                "args_template": {"query": "{query} investment stock crypto Bitcoin"},
                "output_key": "memory_result",
            },
            {
                "tool": "context_detail",
                "args_template": {"category": "lesson", "item_id": 153},
                "output_key": "context_result",
            },
        ],
        KEEP,
    ),
]

# ---------------------------------------------------------------------------
# Dedup clusters: {canonical_id: [ids_to_delete], reason}
# ---------------------------------------------------------------------------

DEDUP_CLUSTERS: dict[int, tuple[list[int], str]] = {
    33: ([28, 50, 66, 73, 75, 84, 90, 95], "C1 crypto price — 9→1"),
    77: ([72, 89, 91, 92, 93],             "C2 lang compare — 6→1"),
    76: ([79, 81],                          "C3 Ethereum — 3→1"),
    39: ([37, 38, 80],                      "C4 AI research — 5→2 (keep 39+78)"),
    64: ([52],                              "C5 Fed rate — 2→1"),
    86: ([88],                              "C6 pseudoscience dup — exact duplicate"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_skill(conn: sqlite3.Connection, skill_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()


def validate_pattern(pattern: str) -> str | None:
    """Compile the pattern and return error string if invalid."""
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return str(e)


def apply_pattern_fix(
    conn: sqlite3.Connection,
    fix: tuple,
    dry_run: bool,
) -> bool:
    skill_id, description, new_trigger, new_steps, new_answer = fix

    row = get_skill(conn, skill_id)
    if row is None:
        log.warning("  SKIP fix id=%d: row not found", skill_id)
        return False

    err = validate_pattern(new_trigger)
    if err:
        log.error("  ABORT fix id=%d: invalid pattern — %s", skill_id, err)
        return False

    current_trigger = row["trigger_pattern"]
    current_steps = row["steps"]
    current_answer = row["answer_template"]

    new_steps_json = json.dumps(new_steps) if new_steps is not None else current_steps
    final_answer = current_answer if new_answer is KEEP else new_answer

    if (current_trigger == new_trigger
            and new_steps_json == current_steps
            and final_answer == current_answer):
        log.info("  SKIP id=%d '%s': already fixed", skill_id, row["name"])
        return False

    log.info("  FIX id=%d '%s': %s", skill_id, row["name"], description)
    if current_trigger != new_trigger:
        log.info("    trigger OLD: %s", current_trigger[:100])
        log.info("    trigger NEW: %s", new_trigger[:100])
    if new_steps is not None:
        old_tools = [s.get("tool") for s in (json.loads(current_steps) if current_steps else [])]
        new_tools = [s.get("tool") for s in new_steps]
        if old_tools != new_tools:
            log.info("    steps OLD: %s", old_tools)
            log.info("    steps NEW: %s", new_tools)

    if not dry_run:
        conn.execute(
            "UPDATE skills SET trigger_pattern=?, steps=?, answer_template=? WHERE id=?",
            (new_trigger, new_steps_json, final_answer, skill_id),
        )
        conn.commit()
    return True


def apply_dedup_deletions(
    conn: sqlite3.Connection,
    dry_run: bool,
) -> int:
    deleted = 0
    for canonical_id, (del_ids, reason) in DEDUP_CLUSTERS.items():
        canonical = get_skill(conn, canonical_id)
        if canonical is None:
            log.warning("  WARN: canonical id=%d not found — skipping cluster '%s'",
                        canonical_id, reason)
            continue
        log.info("  CLUSTER [%s]", reason)
        log.info("    KEEP id=%d '%s'", canonical_id, canonical["name"])
        for del_id in del_ids:
            row = get_skill(conn, del_id)
            if row is None:
                log.info("    SKIP delete id=%d: already gone", del_id)
                continue
            log.info("    DELETE id=%d '%s' (src=%s, used=%d)",
                     del_id, row["name"], row["source"], row["times_used"])
            if not dry_run:
                conn.execute("DELETE FROM skills WHERE id = ?", (del_id,))
                deleted += 1
    if not dry_run:
        conn.commit()
    return deleted


def spot_check(conn: sqlite3.Connection) -> None:
    """Verify fixed patterns fire on representative queries and reject rejection corpus."""
    log.info("\n--- Spot-check: firing tests ---")

    tests = [
        # (skill_id, should_match_queries, should_miss_queries)
        (
            33,  # get_live_crypto_prices
            ["What's the BTC price?", "current ethereum market", "solana value today",
             "what is the xrp price", "dogecoin worth"],
            ["how do I cook pasta?", "what time is it in Tokyo?"],
        ),
        (
            36,  # market_news_summary
            ["what are the markets doing today?", "how are markets moving right now?",
             "any major moves in the markets", "hey what is the market doing"],
            ["buy me bitcoin", "tell me a joke"],
        ),
        (
            40,  # latest_model_web_search
            ["search for the latest llama model", "find the newest claude 3.5 version",
             "check for the latest GPT model available on OpenAI"],
            ["what time is it?", "recommend a book"],
        ),
        (
            58,  # knowledge_graph_query
            ["what do you know about bitcoin from your knowledge graph",
             "what do you know about my job",
             "what do you know about my investment strategy"],
            ["tell me a joke", "what is quantum computing"],
        ),
        (
            60,  # compare_precious_metal_prices
            ["compare gold vs silver", "difference between gold and silver",
             "ratio of silver vs gold", "compare precious metals"],
            ["bitcoin price", "what time is it?"],
        ),
        (
            61,  # get_live_commodity_price
            ["what is gold trading at", "what's silver worth right now",
             "what is bitcoin priced today", "what is copper worth"],
            ["what time is it in Tokyo?", "tell me a joke"],
        ),
        (
            74,  # search_and_summarize_ai_releases
            ["latest AI model releases this week", "new AI model announcements recently",
             "recent AI model launches"],
            ["how do I cook pasta?", "what is quantum computing"],
        ),
        (
            76,  # ethereum_data_retrieval
            ["ethereum price today", "ethereum market data", "ethereum blockchain info",
             "what do you know about ethereum"],
            ["bitcoin price", "latest tech news"],
        ),
        (
            77,  # compare_languages_for_web_api
            ["Rust vs Go for web API", "Python and Node.js backend performance",
             "TypeScript vs JavaScript for backend", "Java versus Rust server framework"],
            ["tell me a joke", "ethereum price"],
        ),
        (
            82,  # investment_opportunity_analysis
            ["best investment opportunities 2026 market trends",
             "what to invest in 2026", "current affairs investment advice",
             "what should I invest in market trends"],
            ["how do I cook pasta?", "ethereum market data"],
        ),
    ]

    ok = err = 0
    for skill_id, should_match, should_miss in tests:
        row = get_skill(conn, skill_id)
        if row is None:
            log.warning("  SKIP spot-check id=%d: not found", skill_id)
            continue
        pattern = row["trigger_pattern"]
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            log.error("  INVALID pattern id=%d: %s", skill_id, e)
            err += 1
            continue

        for q in should_match:
            if regex.search(q):
                ok += 1
            else:
                log.error("  MISS  id=%d '%s': expected match for: %s",
                          skill_id, row["name"], q)
                err += 1

        for q in should_miss:
            if not regex.search(q):
                ok += 1
            else:
                log.error("  FP    id=%d '%s': unexpected match for: %s",
                          skill_id, row["name"], q)
                err += 1

    log.info("  Spot-check results: %d passed, %d failed", ok, err)
    return ok, err


def count_skills(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/data/nova.db",
                        help="Path to nova.db (default: /data/nova.db)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        sys.exit(1)

    conn = open_db(db_path)
    before_count = count_skills(conn)
    log.info("Migration start — skills before: %d%s",
             before_count, "  [DRY RUN]" if args.dry_run else "")

    # -----------------------------------------------------------------------
    # Phase 1: Pattern fixes
    # -----------------------------------------------------------------------
    log.info("\n=== Phase 1: Pattern fixes ===")
    fixes_applied = 0
    for fix in PATTERN_FIXES:
        result = apply_pattern_fix(conn, fix, args.dry_run)
        if result:
            fixes_applied += 1
    log.info("Pattern fixes applied: %d", fixes_applied)

    # -----------------------------------------------------------------------
    # Phase 2: Dedup deletions
    # -----------------------------------------------------------------------
    log.info("\n=== Phase 2: Dedup deletions ===")
    deleted = apply_dedup_deletions(conn, args.dry_run)
    log.info("Skills deleted: %d", deleted)

    # -----------------------------------------------------------------------
    # Phase 3: Spot-check
    # -----------------------------------------------------------------------
    log.info("\n=== Phase 3: Spot-check ===")
    ok, err = spot_check(conn)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    after_count = count_skills(conn)
    log.info(
        "\n=== Summary ===\n"
        "  Skills before:  %d\n"
        "  Fixes applied:  %d\n"
        "  Deletions:      %d\n"
        "  Skills after:   %d\n"
        "  Spot-check:     %d passed / %d failed\n"
        "  Dry run:        %s",
        before_count, fixes_applied, deleted, after_count,
        ok, err,
        "yes (no changes written)" if args.dry_run else "no (changes committed)",
    )

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
