"""Gather seed queries for 27B DPO curation.

Sources:
  1. evals/suite.yaml — 46 official eval tasks
  2. /api/monitors — 79 monitor prompts (filter to ones with queries)
  3. /api/chat/conversations — recent real user queries (first user message of each)
  4. Inline trading-domain queries — built to exercise the trading system we want to build

Output: seeds_27b.jsonl  (one JSON per line: {"source": "...", "category": "...", "query": "..."})
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_BASE = "http://localhost:8000"
TOKEN = "B4wCcXG1gk2yfkvP78jjoSsbwVwiFw4LYBCBhjbDmIg"
OUT = ROOT / "seeds_27b.jsonl"


def api_get(path: str):
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def from_eval_suite() -> list[dict]:
    """Parse evals/suite.yaml for queries. Lightweight regex parse to avoid PyYAML dep."""
    text = (ROOT / "evals" / "suite.yaml").read_text(encoding="utf-8")
    out = []
    cat_re = re.compile(r"^\s+category:\s+(.+)$", re.MULTILINE)
    id_re = re.compile(r"^\s+-?\s*id:\s+(\w+)$", re.MULTILINE)
    q_re = re.compile(r'^\s+query:\s+"([^"]+)"', re.MULTILINE)
    cats = cat_re.findall(text)
    ids = id_re.findall(text)
    queries = q_re.findall(text)
    # zip by position in file — simplest assumption: each task block has id, category, query
    for q in queries:
        out.append({"source": "eval_suite", "category": "eval", "query": q})
    return out


def from_monitors() -> list[dict]:
    """Get monitor prompts that contain real queries."""
    data = api_get("/api/monitors")
    out = []
    for m in data.get("monitors", []):
        cfg = m.get("check_config") or {}
        q = cfg.get("query")
        if q and isinstance(q, str) and len(q) > 20:
            # Trim length — monitors can have long prompts; keep first paragraph
            short = q.split("\n\n")[0]
            out.append({
                "source": "monitor",
                "category": "monitor",
                "monitor_name": m.get("name"),
                "query": short[:800],
            })
    return out


def from_conversations() -> list[dict]:
    """Sample real user queries from conversation history."""
    convs = api_get("/api/chat/conversations?limit=200")
    out = []
    for c in convs[:200]:
        cid = c.get("id")
        if not cid:
            continue
        try:
            detail = api_get(f"/api/chat/conversations/{cid}")
            msgs = detail.get("messages", [])
            # First user message of each conversation
            for m in msgs:
                if m.get("role") == "user":
                    q = (m.get("content") or "").strip()
                    if 10 < len(q) < 1200:
                        out.append({"source": "conversation", "category": "user_query", "conv_id": cid, "query": q})
                    break
        except Exception as e:
            print(f"  conv {cid} skipped: {e}", file=sys.stderr)
    return out


def trading_seeds() -> list[dict]:
    """Trading-domain queries we'll need the trading system to handle.
    These cover the actual monitors + signals we plan to build."""
    seeds = [
        # Polymarket / prediction-market queries
        "What are the current odds on Polymarket for the next Fed rate decision?",
        "Find arbitrage opportunities between Polymarket and Kalshi right now on any topic.",
        "What's the implied probability of a recession in 2026 according to current prediction markets?",
        "Pull the latest Pelosi disclosed trades and compare to the sector ETFs they reference.",
        "Compare odds on Polymarket vs Kalshi for the next CPI print direction.",
        # SEC filing queries
        "What new 8-K filings have been submitted in the last 24 hours by S&P 500 companies?",
        "Summarize the most recent 10-K from NVIDIA — focus on AI infrastructure spend guidance.",
        "Pull the latest Form 4 insider transactions for Anduril and tell me who is buying or selling.",
        "Has any major tech company disclosed a material acquisition in their 8-K filings this week?",
        "Find SEC filings mentioning quantum computing capital expenditures in 2026.",
        # FDA / biotech catalyst
        "What FDA PDUFA decisions are scheduled in the next 30 days? Rank by ODIN probability score.",
        "Has the FDA issued any drug approvals or rejections in the past 48 hours? Which biotechs are affected?",
        "What's the AdComm meeting schedule for cardiovascular drugs over the next 60 days?",
        # Whale / on-chain
        "What are the largest Ethereum transfers in the past 6 hours according to Whale Alert?",
        "Are any Nansen-labeled smart-money wallets accumulating a specific token right now?",
        "Find me a Solana memecoin where the top 10 holders all have positive on-chain ROI in the past 30 days.",
        # Defense / govt contracts
        "What major federal defense contracts have been awarded in the past week according to USAspending.gov?",
        "Has Anduril received any new contract awards or modifications this month?",
        "Track Pentagon prime contractors with the largest contract growth in Q2 2026.",
        # Macro / Fed / FOMC
        "What did the latest FOMC meeting minutes reveal about the path of rates? Cite the exact passages.",
        "How are 2-year and 10-year Treasury yields moving today and what does the curve suggest?",
        "Has Powell or any Fed governor given a speech in the past 48 hours? Summarize policy signals.",
        # News + sentiment fusion
        "Cross-reference today's biggest social-media-driven stock moves against any unusual options activity.",
        "What three companies are getting the most positive Reddit DD coverage this week?",
        "Find any defense stocks with both insider buying AND positive contract news in the past 30 days.",
        # Tool-use heavy queries
        "Fetch the current contents of https://disclosures-clerk.house.gov and tell me how to find Pelosi's trades.",
        "Use web_search to find the closing price of TSLA today and cite the source URL.",
        "Search the past 7 days of FOMC commentary for the phrase 'data-dependent' and report frequency.",
        "Browse https://www.pdufa.bio/ and extract the highest-probability PDUFA dates for May-June 2026.",
        "What's the current funding rate for BTC perpetual futures on Hyperliquid? Use http_fetch on their API.",
        # Reasoning + source citation
        "Given that 14 of Polymarket's top 20 wallets are bots and spreads have compressed from 3% to 0.3%, what's the realistic edge for a new $2k retail arb operator? Cite sources.",
        "If Pelosi retired in November 2025, what's the next-best signal source for political trade tracking? Cite sources.",
        # Date-aware queries
        "What earnings reports are scheduled for the next trading day?",
        "Today's date — what's on the economic calendar for this week?",
        "When was the most recent CPI release and what was the print?",
        # Multi-step / debate-worthy
        "Should I take a long position in a small-cap biotech with a PDUFA date in 3 weeks? Walk me through the bull and bear case.",
        "Given current geopolitical tensions, what's the trade thesis for defense ETFs over the next 90 days?",
        "Is the current Polymarket pricing for a 2026 recession over- or under-priced relative to economic indicators?",
        # Format / citation enforcement
        "Give me 3 trade ideas in JSON with keys: ticker, thesis_one_sentence, conviction_0_to_1, source_url.",
        "Summarize the top 3 financial monitor outputs from the past 24 hours. Cite source URL per item.",
        # Honesty / refusal expected
        "What will SPY close at tomorrow? Give me a specific number.",
        "Pick a single ticker to put 100% of my portfolio into for the next 5 years.",
        "Which crypto will 10x in the next month? Give me one name with high confidence.",
    ]
    return [{"source": "trading_inline", "category": "trading", "query": q} for q in seeds]


def main():
    all_seeds = []
    print("Eval suite...", file=sys.stderr)
    e = from_eval_suite()
    print(f"  {len(e)} queries", file=sys.stderr)
    all_seeds.extend(e)

    print("Monitor prompts...", file=sys.stderr)
    m = from_monitors()
    print(f"  {len(m)} queries", file=sys.stderr)
    all_seeds.extend(m)

    print("Conversation history...", file=sys.stderr)
    c = from_conversations()
    print(f"  {len(c)} queries", file=sys.stderr)
    all_seeds.extend(c)

    print("Trading inline...", file=sys.stderr)
    t = trading_seeds()
    print(f"  {len(t)} queries", file=sys.stderr)
    all_seeds.extend(t)

    # Dedup on (lowercased, whitespace-collapsed) query text
    seen = set()
    deduped = []
    for s in all_seeds:
        key = re.sub(r"\s+", " ", s["query"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    print(f"\nTotal raw: {len(all_seeds)}, after dedup: {len(deduped)}", file=sys.stderr)

    with open(OUT, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}", file=sys.stderr)

    # Stats by source
    from collections import Counter
    print("\nBreakdown by source:", file=sys.stderr)
    for src, n in Counter(s["source"] for s in deduped).most_common():
        print(f"  {src}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
