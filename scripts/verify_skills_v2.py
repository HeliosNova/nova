#!/usr/bin/env python3
"""Verification script for migrate_skills_v2.py — Part B fixes.

Reads the live DB and asserts the five broken-step-arg skills are
in the correct post-migration state:

  id=32  check_ev_availability_and_market  — (?P<year>...) present, {year} valid
  id=57  crypto_asset_status_check         — (?P<asset>...) present, {asset} valid
  id=62  crypto_price_comparison           — (?P<coin_name>...) present
  id=65  stock_earnings_lookup             — (?P<company>...) present, browser step gone
  id=85  stock_analysis                   — steps 2-4 removed (no {stock1}/{stock2}/{stock3})

Exits 0 if all checks pass; exits 1 on any failure.

Usage:
    python scripts/verify_skills_v2.py [--db /path/to/nova.db]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("verify_skills_v2")


_BROADNESS_TEST_QUERIES = [
    "What's the weather like today?", "Tell me a joke", "How do I cook pasta?",
    "What is quantum computing?", "Recommend a good book",
    "How tall is Mount Everest?", "Translate hello to Spanish",
    "What time is it in Tokyo?", "How much does a Tesla cost?",
    "What's the price of gold?", "Compare Python and JavaScript",
    "How do I fix a flat tire?", "Who won the World Cup?",
    "What should I eat for dinner?", "How much is a flight to Paris?",
    "Today I want to learn how to play chess",
    "Give me the current bus schedule for downtown",
    "What's the latest gossip about celebrity drama?",
    "Tell me what's happening right now in my neighborhood",
    "I need the most recent study tips for exams",
    "¿Cuál es el clima hoy?", "今天天气怎么样？", "ما هو الطقس اليوم؟",
    "आज मौसम कैसा है?", "今日の天気は何ですか？",
]


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_skill(conn: sqlite3.Connection, skill_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()


def broadness_hits(pattern: str) -> int:
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return 99
    return sum(1 for q in _BROADNESS_TEST_QUERIES if rx.search(q))


def check_pass(label: str) -> None:
    log.info("  PASS  %s", label)


def check_fail(label: str) -> None:
    log.error("  FAIL  %s", label)


def verify_skill(
    conn: sqlite3.Connection,
    skill_id: int,
    *,
    must_have_groups: list[str],
    must_not_have_step_args: list[str],
    must_not_have_step_tools: list[str],
    should_match: list[str],
    should_miss: list[str],
) -> int:
    """Return number of failures."""
    row = get_skill(conn, skill_id)
    if row is None:
        check_fail(f"id={skill_id}: skill not found in DB")
        return 1

    fail = 0
    pattern = row["trigger_pattern"]
    name = row["name"]
    log.info("id=%d '%s'", skill_id, name)

    # 1. Named groups present
    named_groups = set(re.findall(r'\(\?P<(\w+)>', pattern))
    for g in must_have_groups:
        if g in named_groups:
            check_pass(f"  named group (?P<{g}>...) present in trigger")
        else:
            check_fail(f"  named group (?P<{g}>...) MISSING from trigger")
            fail += 1

    # 2. Broadness
    hits = broadness_hits(pattern)
    if hits < 2:
        check_pass(f"  broadness: {hits}/25 hits (threshold <2)")
    else:
        check_fail(f"  broadness: {hits}/25 hits — still too broad!")
        fail += 1

    # 3. Step arg placeholders no longer broken
    steps = json.loads(row["steps"]) if row["steps"] else []
    all_args_text = " ".join(
        json.dumps(s.get("args_template", {})) for s in steps
    )
    for bad_placeholder in must_not_have_step_args:
        if bad_placeholder not in all_args_text:
            check_pass(f"  broken placeholder {bad_placeholder!r} gone from steps")
        else:
            check_fail(f"  broken placeholder {bad_placeholder!r} still in step args")
            fail += 1

    # 4. Removed step tools
    step_tools = [s.get("tool") for s in steps]
    for bad_tool in must_not_have_step_tools:
        if bad_tool not in step_tools:
            check_pass(f"  broken tool '{bad_tool}' step removed")
        else:
            check_fail(f"  broken tool '{bad_tool}' step still present")
            fail += 1

    # 5. Spot check: pattern fires on expected queries
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        check_fail(f"  pattern fails to compile: {e}")
        return fail + 1

    for q in should_match:
        if rx.search(q):
            check_pass(f"  HIT  {q!r}")
        else:
            check_fail(f"  MISS {q!r}")
            fail += 1

    for q in should_miss:
        if not rx.search(q):
            check_pass(f"  SKIP {q!r} (rejected)")
        else:
            check_fail(f"  FP   {q!r} (unexpected match)")
            fail += 1

    return fail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/data/nova.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        sys.exit(1)

    conn = open_db(db_path)
    total_fail = 0

    # --- id=32 ---
    total_fail += verify_skill(
        conn, 32,
        must_have_groups=["year"],
        must_not_have_step_args=[],          # {year} was broken; now valid via named group
        must_not_have_step_tools=[],
        should_match=[
            "should I buy an electric car this year",
            "is it a good time to buy an EV now",
            "what is available for electric vehicles in 2026",
        ],
        should_miss=[
            "should I buy a Tesla",
            "what's the best EV on the market",
            "electric car prices today",
        ],
    )

    # --- id=57 ---
    total_fail += verify_skill(
        conn, 57,
        must_have_groups=["asset"],
        must_not_have_step_args=[],          # {asset} was broken; now valid via named group
        must_not_have_step_tools=[],
        should_match=[
            "tell me bitcoin price",
            "what is ethereum adoption",
            "what's going on with monero news",
        ],
        should_miss=[
            "buy bitcoin",
            "ethereum blockchain technology",
            "latest crypto market overview",
        ],
    )

    # --- id=62 ---
    total_fail += verify_skill(
        conn, 62,
        must_have_groups=["coin_name"],
        must_not_have_step_args=[],          # {coin_name} was broken; now valid via named group
        must_not_have_step_tools=[],
        should_match=[
            "compare bitcoin prices",
            "check ethereum price with litecoin",
            "get btc price vs eth",
        ],
        should_miss=[
            "what's the bitcoin price",
            "compare programming languages",
            "tell me about crypto",
        ],
    )

    # --- id=65 ---
    total_fail += verify_skill(
        conn, 65,
        must_have_groups=["company"],
        must_not_have_step_args=["{first_result_url}"],  # fundamentally broken placeholder
        must_not_have_step_tools=["browser"],              # browser step removed
        should_match=[
            "what AAPL earnings last quarter",
            "find MSFT revenue Q3 2025",
            "get TSLA financial results recent",
        ],
        should_miss=[
            "how do I file quarterly taxes",
            "tell me about stocks",
            "what is a P/E ratio",
        ],
    )

    # --- id=85 ---
    total_fail += verify_skill(
        conn, 85,
        must_have_groups=[],
        must_not_have_step_args=["{stock1}", "{stock2}", "{stock3}"],
        must_not_have_step_tools=[],
        should_match=[
            "top trending tech stocks today",
            "compare P/E ratios for tech companies",
            "find stock valuation for Apple",
        ],
        should_miss=[
            "what are trending stocks",
            "compare stock prices",
            "how do I value a stock",
        ],
    )

    # --- id=45 should be gone ---
    row45 = get_skill(conn, 45)
    if row45 is None:
        check_pass("id=45 real_time_data_search deleted (was 5/25 broadness hits)")
    else:
        check_fail("id=45 real_time_data_search still present — delete migration failed")
        total_fail += 1

    if total_fail == 0:
        log.info("\nAll Part B verifications PASSED ✓")
    else:
        log.error("\n%d verification(s) FAILED", total_fail)
        sys.exit(1)


if __name__ == "__main__":
    main()
