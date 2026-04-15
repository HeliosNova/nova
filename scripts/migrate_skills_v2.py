#!/usr/bin/env python3
"""Idempotent skill corpus migration v2 — quality audit pass #2.

Applies two classes of fix that were out of scope in migrate_skills.py (v1):

A. Over-broad pattern narrowing
   id=44  real_time_price_lookup   — no (?i), no \\b, unbounded .* wildcards
   id=45  real_time_data_search    — DELETE: pure temporal, 5/25 broadness hits

B. Broken step-arg placeholders ({name} with no matching (?P<name>...) capture group)
   id=32  check_ev_availability_and_market  — {year} undefined → add (?P<year>...)
   id=57  crypto_asset_status_check         — {asset} undefined → add (?P<asset>...)
   id=62  crypto_price_comparison           — {coin_name} undefined → add (?P<coin_name>...)
   id=65  stock_earnings_lookup             — {company} unnamed, {first_result_url} broken →
                                              named group + remove broken browser step
   id=85  stock_analysis                    — {stock1}/{stock2}/{stock3} undefined →
                                              add (?i)/\\b, drop broken steps 2-4

Before count: 48 skills (after v1 migration)
Expected after count: 47 skills (1 deletion: id=45)

Safe to re-run: all operations check current DB state before acting.

Usage:
    python scripts/migrate_skills_v2.py [--db /path/to/nova.db] [--dry-run]
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
log = logging.getLogger("migrate_skills_v2")

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

KEEP = object()  # don't change this field


# ---------------------------------------------------------------------------
# Pattern fixes  (id, description, new_trigger, new_steps_or_None, new_answer)
# ---------------------------------------------------------------------------

FIXES_V2: list[tuple] = [
    # ------------------------------------------------------------------
    # id=44  real_time_price_lookup
    #   OLD: (current|right now|today|latest).*(price|cost|value|rate).*
    #   Bugs: no (?i), no \b anchors, unbounded .* before and after price term.
    #   FIX:  add (?i), add \b anchors, bound the wildcard to .{0,80}.
    #         Trailing .* removed — matching ends at the price/rate word.
    # ------------------------------------------------------------------
    (
        44,
        r"narrow: add (?i)+\b, bound .* → .{0,80}, remove trailing .*",
        r"(?i)\b(?:current|right\s+now|today|latest)\b.{0,80}\b(?:price|cost|value|rate)\b",
        None,   # keep existing steps (use {query})
        KEEP,
    ),

    # ------------------------------------------------------------------
    # id=32  check_ev_availability_and_market
    #   OLD: (should i buy|is it a good time to buy|what is available for)
    #         (an? electric car|ev|electric vehicle)(s)? (this year|in \d{4}|now)
    #   Bugs: no (?i), {year} in both step args has no (?P<year>...) capture group.
    #   FIX:  add (?i), convert year group → (?P<year>...) named group,
    #         tighten spacing to \s+, add \b anchors.
    #   Steps unchanged: {year} is now resolved by the named capture group.
    # ------------------------------------------------------------------
    (
        32,
        r"add (?i) + (?P<year>...) named group for {year} in step args; fix 'an EV' coverage",
        (
            r"(?i)\b(?:should\s+I\s+buy|is\s+it\s+a\s+good\s+time\s+to\s+buy"
            r"|what\s+is\s+available\s+for)\s+"
            # Optional article (a/an) before any EV term — original only had it before
            # "electric car"; this also covers "an EV" / "an electric vehicle".
            r"(?:an?\s+)?(?:electric\s+(?:car|vehicle)|ev)s?\b\s+"
            r"(?P<year>this\s+year|in\s+\d{4}|now)\b"
        ),
        None,   # keep existing steps — {year} now resolves via named group
        KEEP,
    ),

    # ------------------------------------------------------------------
    # id=57  crypto_asset_status_check
    #   OLD: (tell me|what is|whats going on with)\s+(bitcoin|btc|...|xmr)\s+
    #         (adoption|price|status|news|investment|market|current affairs)
    #   Bugs: no (?i), no \b anchors, {asset} in step args has no (?P<asset>...) group.
    #   FIX:  add (?i) + \b anchors, convert asset group → (?P<asset>...).
    #         Handle "whats" / "what's" via what'?s.
    #   Steps unchanged: {asset} now resolves via named group.
    # ------------------------------------------------------------------
    (
        57,
        r"add (?i)+\b + (?P<asset>...) named group for {asset} in step args",
        (
            r"(?i)\b(?:tell\s+me|what\s+is|what'?s\s+going\s+on\s+with)\b\s+"
            r"(?P<asset>bitcoin|btc|ethereum|eth|xrp|solana|ada|doge|cardano|ripple"
            r"|polkadot|dot|shiba|matic|polygon|avax|avalanche|chainlink|link"
            r"|uniswap|uni|litecoin|ltc|monero|xmr)\b\s+"
            r"\b(?:adoption|price|status|news|investment|market|current\s+affairs)\b"
        ),
        None,   # keep existing steps — {asset} now resolves via named group
        KEEP,
    ),

    # ------------------------------------------------------------------
    # id=62  crypto_price_comparison
    #   OLD: (compare|check|get)\s+(bitcoin|ethereum|...|crypto)\s+price(s)?\s*
    #         (and|with|vs|versus)?\s*([a-z]+)
    #   Bugs: no (?i), no \b anchors, {coin_name} in step args has no named group,
    #         trailing ([a-z]+) is an unnamed broad wildcard.
    #   FIX:  add (?i) + \b anchors, convert first coin group → (?P<coin_name>...),
    #         replace unnamed trailing group with non-capturing optional.
    #   Steps unchanged: {coin_name} now resolves via named group.
    # ------------------------------------------------------------------
    (
        62,
        r"add (?i)+\b + (?P<coin_name>...) named group for {coin_name} in step args",
        (
            r"(?i)\b(?:compare|check|get)\b\s+"
            r"(?P<coin_name>bitcoin|ethereum|solana|btc|eth|sol|cryptocurrency|crypto)\b\s+"
            r"prices?\s*(?:\b(?:and|with|vs\.?|versus)\b\s+[a-z]+)?"
        ),
        None,   # keep existing steps — {coin_name} now resolves via named group
        KEEP,
    ),

    # ------------------------------------------------------------------
    # id=65  stock_earnings_lookup
    #   OLD: (What|Find|Get)\s+(the)?\s+([A-Z]{2,5})?\s*(AAPL|...|TWTR)?\s+
    #         (earnings|...|recent)
    #   Bugs: no (?i), {company} in step[0] has no named group, step[1] uses
    #         {first_result_url} which is fundamentally broken (URL cannot be
    #         extracted from free-text search results via the template system).
    #         Ticker list is also duplicated.
    #   FIX:  add (?i), merge unnamed+enumerated company groups → (?P<company>...),
    #         deduplicate ticker list, change step[0] args from {company} → {query}
    #         (always resolves; ticker is present in original query anyway),
    #         remove step[1] (browser + {first_result_url}).
    # ------------------------------------------------------------------
    (
        65,
        (
            r"add (?i), (?P<company>...) named group, dedup tickers, "
            r"fix step[0] {company}→{query}, remove broken browser/{first_result_url} step[1]"
        ),
        (
            r"(?i)\b(?:what|find|get)\s+(?:the\s+)?"
            r"(?P<company>AAPL|MSFT|GOOGL|AMZN|TSLA|META|NVDA|AMD|INTC|CRM|ORCL|ADBE"
            r"|NFLX|DIS|PYPL|UBER|LYFT|RIVN|LCID|F|GM|BAC|JPM|WFC|GS|MS|C|V|MA|AXP"
            r"|KO|PEP|MCD|SBUX|TGT|WMT|HD|LOW|NKE|LULU|ZM|SNAP|TWTR|FB|GOOG"
            r"|[A-Z]{2,5})?\s+"
            r"(?:earnings|financial\s+results|revenue|EPS)\s+"
            r"(?:last\s+quarter|previous\s+quarter|Q[1-4]\s*\d{2,4}|recent)\b"
        ),
        [
            {
                "tool": "web_search",
                # Use {query} (always resolves) instead of {company} (was undefined).
                # The ticker symbol appears in the original query, so {query} carries it.
                "args_template": {"query": "{query} last quarter earnings revenue EPS"},
                "output_key": "search_results",
            }
            # step[1] browser/{first_result_url} removed:
            # {first_result_url} is undefined — the template system cannot extract
            # a URL from free-text web_search results. LLM synthesises from step[0].
        ],
        KEEP,
    ),

    # ------------------------------------------------------------------
    # id=85  stock_analysis
    #   OLD: (top trending tech stocks|compare P/E ratios|find stock valuation)
    #   Bugs: no (?i), no \b anchors, steps 2-4 use {stock1}/{stock2}/{stock3}
    #         which are all undefined — no capture groups extract those names.
    #   FIX:  add (?i) + \b anchors, keep steps 0-1 (valid), remove steps 2-4.
    # ------------------------------------------------------------------
    (
        85,
        r"add (?i)+\b to trigger; remove broken steps 2-4 ({stock1}/{stock2}/{stock3} undefined)",
        r"(?i)\b(?:top\s+trending\s+tech\s+stocks|compare\s+P/E\s+ratios?|find\s+stock\s+valuation)\b",
        [
            {
                "tool": "web_search",
                "args_template": {"query": "{query} top trending tech stocks today"},
                "output_key": "trending_stocks",
            },
            {
                "tool": "browser",
                "args_template": {
                    "action": "navigate",
                    "url": "https://stockanalysis.com/trending/",
                },
                "output_key": "page_data",
            },
            # steps 2-4 removed — {stock1}, {stock2}, {stock3} are undefined placeholders;
            # dynamic stock-name extraction from prior step results is not supported by
            # the args_template system. LLM synthesises P/E analysis from steps 0-1.
        ],
        KEEP,
    ),
]


# ---------------------------------------------------------------------------
# Deletions  [(id, reason), ...]
# ---------------------------------------------------------------------------

DELETES_V2: list[tuple[int, str]] = [
    (
        45,
        "pure temporal pattern (?i)(current|right\\s+now|today|latest) — 5/25 broadness hits, "
        "no domain constraint; intent cannot be salvaged without creating a different skill",
    ),
]


# ---------------------------------------------------------------------------
# Broadness test corpus (mirrored from app/core/skills.py)
# ---------------------------------------------------------------------------

_BROADNESS_TEST_QUERIES = [
    "What's the weather like today?",
    "Tell me a joke",
    "How do I cook pasta?",
    "What is quantum computing?",
    "Recommend a good book",
    "How tall is Mount Everest?",
    "Translate hello to Spanish",
    "What time is it in Tokyo?",
    "How much does a Tesla cost?",
    "What's the price of gold?",
    "Compare Python and JavaScript",
    "How do I fix a flat tire?",
    "Who won the World Cup?",
    "What should I eat for dinner?",
    "How much is a flight to Paris?",
    "Today I want to learn how to play chess",
    "Give me the current bus schedule for downtown",
    "What's the latest gossip about celebrity drama?",
    "Tell me what's happening right now in my neighborhood",
    "I need the most recent study tips for exams",
    "¿Cuál es el clima hoy?",
    "今天天气怎么样？",
    "ما هو الطقس اليوم؟",
    "आज मौसम कैसा है?",
    "今日の天気は何ですか？",
]


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
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return str(e)


def broadness_hits(pattern: str) -> int:
    """Count how many of the 25 broadness queries the pattern matches."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return 99
    return sum(1 for q in _BROADNESS_TEST_QUERIES if regex.search(q))


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

    hits = broadness_hits(new_trigger)
    if hits >= 2:
        log.error(
            "  ABORT fix id=%d: new pattern still too broad (%d/25 broadness hits)",
            skill_id, hits,
        )
        return False

    current_trigger = row["trigger_pattern"]
    current_steps   = row["steps"]
    current_answer  = row["answer_template"]

    new_steps_json = json.dumps(new_steps) if new_steps is not None else current_steps
    final_answer   = current_answer if new_answer is KEEP else new_answer

    if (current_trigger == new_trigger
            and new_steps_json == current_steps
            and final_answer == current_answer):
        log.info("  SKIP id=%d '%s': already fixed", skill_id, row["name"])
        return False

    log.info("  FIX id=%d '%s': %s", skill_id, row["name"], description)
    if current_trigger != new_trigger:
        log.info("    trigger OLD: %s", current_trigger[:120])
        log.info("    trigger NEW: %s", new_trigger[:120])
    if new_steps is not None:
        old_tools = [s.get("tool") for s in (json.loads(current_steps) if current_steps else [])]
        new_tools = [s.get("tool") for s in new_steps]
        if old_tools != new_tools:
            log.info("    steps OLD tools: %s", old_tools)
            log.info("    steps NEW tools: %s", new_tools)
    log.info("    broadness: %d/25 hits (threshold <2) ✓", hits)

    if not dry_run:
        conn.execute(
            "UPDATE skills SET trigger_pattern=?, steps=?, answer_template=? WHERE id=?",
            (new_trigger, new_steps_json, final_answer, skill_id),
        )
        conn.commit()
    return True


def apply_deletions(
    conn: sqlite3.Connection,
    dry_run: bool,
) -> int:
    deleted = 0
    for skill_id, reason in DELETES_V2:
        row = get_skill(conn, skill_id)
        if row is None:
            log.info("  SKIP delete id=%d: already gone", skill_id)
            continue
        log.info("  DELETE id=%d '%s': %s", skill_id, row["name"], reason[:80])
        if not dry_run:
            conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            deleted += 1
    if not dry_run:
        conn.commit()
    return deleted


# ---------------------------------------------------------------------------
# Spot-check
# ---------------------------------------------------------------------------

def spot_check(conn: sqlite3.Connection) -> tuple[int, int]:
    """Verify each fixed pattern fires on paraphrases and rejects unrelated queries."""
    log.info("\n--- Spot-check: pattern tests ---")

    tests = [
        # (skill_id, should_match, should_miss)
        (
            44,  # real_time_price_lookup (narrowed)
            [
                "what is the current price of gold",
                "today's exchange rate for euros",
                "what's the latest cost of a flight to Tokyo",
            ],
            [
                "what's the weather today",            # temporal only — no price term
                "give me the current bus schedule",    # temporal + non-price domain
                "what's the latest gossip",            # temporal only — no price term
            ],
        ),
        (
            32,  # check_ev_availability_and_market
            [
                "should I buy an electric car this year",
                "is it a good time to buy an EV now",
                "what is available for electric vehicles in 2026",
            ],
            [
                "should I buy a Tesla",                # no EV phrase + year
                "what's the best EV on the market",   # no intent verb + year
                "electric car prices today",           # no intent verb phrase
            ],
        ),
        (
            57,  # crypto_asset_status_check
            [
                "tell me bitcoin price",
                "what is ethereum adoption",
                "what's going on with monero news",
            ],
            [
                "buy bitcoin",                         # no intent verb
                "ethereum blockchain technology",      # no category word at end
                "latest crypto market overview",       # no intent verb + named asset
            ],
        ),
        (
            62,  # crypto_price_comparison
            [
                "compare bitcoin prices",
                "check ethereum price with litecoin",
                "get btc price vs eth",
            ],
            [
                "what's the bitcoin price",            # no compare/check/get verb
                "compare programming languages",       # coin not in list
                "tell me about crypto",                # no compare/check/get verb
            ],
        ),
        (
            65,  # stock_earnings_lookup
            [
                "what AAPL earnings last quarter",
                "find MSFT revenue Q3 2025",
                "get TSLA financial results recent",
            ],
            [
                "how do I file quarterly taxes",       # no earnings + quarter phrase
                "tell me about stocks",                # no earnings + quarter
                "what is a P/E ratio",                 # no earnings + quarter
            ],
        ),
        (
            85,  # stock_analysis
            [
                "top trending tech stocks today",
                "compare P/E ratios for tech companies",
                "find stock valuation for Apple",
            ],
            [
                "what are trending stocks",            # no exact phrase match
                "compare stock prices",                # no P/E ratio phrase
                "how do I value a stock",              # no 'find stock valuation' phrase
            ],
        ),
    ]

    ok = err = 0
    for skill_id, should_match, should_miss in tests:
        row = get_skill(conn, skill_id)
        if row is None:
            log.warning("  SKIP spot-check id=%d: not found (deleted?)", skill_id)
            continue
        pattern = row["trigger_pattern"]
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            log.error("  INVALID pattern id=%d: %s", skill_id, e)
            err += 1
            continue

        hits = broadness_hits(pattern)
        if hits >= 2:
            log.error(
                "  BROAD  id=%d '%s': %d/25 broadness hits (threshold <2)",
                skill_id, row["name"], hits,
            )
            err += 1
        else:
            log.info("  BROADNESS id=%d: %d/25 hits ✓", skill_id, hits)
            ok += 1

        for q in should_match:
            if regex.search(q):
                log.info("  HIT   id=%d: %-55s → ✓", skill_id, repr(q))
                ok += 1
            else:
                log.error("  MISS  id=%d '%s': expected match for: %s",
                          skill_id, row["name"], q)
                err += 1

        for q in should_miss:
            if not regex.search(q):
                log.info("  SKIP  id=%d: %-55s → ✓ (rejected)", skill_id, repr(q))
                ok += 1
            else:
                log.error("  FP    id=%d '%s': unexpected match for: %s",
                          skill_id, row["name"], q)
                err += 1

    # Verify id=45 is gone
    if get_skill(conn, 45) is not None:
        log.error("  FAIL  id=45 real_time_data_search still present (should be deleted)")
        err += 1
    else:
        log.info("  DELETE confirmed: id=45 real_time_data_search is gone ✓")
        ok += 1

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
    log.info("Migration v2 start — skills before: %d%s",
             before_count, "  [DRY RUN]" if args.dry_run else "")

    # -----------------------------------------------------------------------
    # Phase 1: Pattern fixes
    # -----------------------------------------------------------------------
    log.info("\n=== Phase 1: Pattern fixes ===")
    fixes_applied = 0
    for fix in FIXES_V2:
        if apply_pattern_fix(conn, fix, args.dry_run):
            fixes_applied += 1
    log.info("Pattern fixes applied: %d", fixes_applied)

    # -----------------------------------------------------------------------
    # Phase 2: Deletions
    # -----------------------------------------------------------------------
    log.info("\n=== Phase 2: Deletions ===")
    deleted = apply_deletions(conn, args.dry_run)
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
