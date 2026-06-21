#!/usr/bin/env python3
"""Training curriculum — generates targeted DPO pairs by running scenarios
against the live Nova API and correcting bad responses.

Each scenario:
1. Sends a query to Nova
2. Captures the response (the "rejected" answer)
3. Sends a correction with the ideal response pattern
4. The correction handler creates a lesson + DPO training pair automatically

Usage:
    NOVA_API_KEY=xxx python scripts/training_curriculum.py
    NOVA_API_KEY=xxx python scripts/training_curriculum.py --category tool_chaining
    NOVA_API_KEY=xxx python scripts/training_curriculum.py --dry-run
"""

import argparse
import json
import os
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx

API = os.getenv("NOVA_API_URL", "http://localhost:8000")
KEY = os.getenv("NOVA_API_KEY", "")
HEADERS = {"Content-Type": "application/json"}
if KEY:
    HEADERS["Authorization"] = f"Bearer {KEY}"

DELAY = 3  # seconds between scenarios


# Each scenario: (query, correction_template)
# The correction is what the user says AFTER seeing Nova's response.
# It triggers the correction detection → lesson + DPO pair.

SCENARIOS = {
    "tool_chaining": [
        (
            "What were today's earnings reports? Give me actual EPS numbers.",
            "That's not good enough. When web_search returns portal pages without actual data, you should use the browser tool to navigate directly to the page and extract the real numbers. Never give up when you have the browser tool available.",
        ),
        (
            "What is the exact current price of Ethereum right now?",
            "No, don't answer from memory for current prices. You must always use web_search for anything with 'current', 'right now', 'today', or 'latest' in the query. Real-time data requires a search.",
        ),
        (
            "Check if https://example.com is up and what it shows",
            "Wrong approach. When asked to check a specific URL, use http_fetch to get it directly instead of searching the web about it. http_fetch is for fetching URLs, web_search is for finding information.",
        ),
        (
            "What is the factorial of 25?",
            "Don't try to compute that in your head. Always use the calculator tool for any math beyond basic arithmetic. You have a calculator — use it.",
        ),
        (
            "What's in the documents I uploaded about blockchain?",
            "You should have used knowledge_search to search the uploaded documents instead of guessing. knowledge_search exists exactly for searching ingested documents.",
        ),
        (
            "When one tool returns an error or incomplete data, what should you do?",
            "The correct answer is: try another tool. If web_search fails, try browser. If browser fails, try http_fetch. If http_fetch fails, try web_search with different terms. Never tell the user you can't do something when you have untried tools. Always exhaust all options.",
        ),
    ],
    "response_quality": [
        (
            "What is 2+2?",
            "Too verbose. The answer is just '4'. Don't add explanations, context, or filler for simple questions. Lead with the answer.",
        ),
        (
            "What's the latest news about OpenAI?",
            "Don't start with 'Based on my search results' or 'I found the following'. Lead with the actual news. Put the most important fact first, then details. Source citations go at the end, not the beginning.",
        ),
        (
            "What is the speed of light?",
            "Too much padding. The answer is '299,792,458 meters per second' or approximately '3 x 10^8 m/s'. Don't add unnecessary caveats or introductions for factual questions with definitive answers.",
        ),
        (
            "Summarize what happened in crypto markets this week",
            "That's too long. A summary should be 3-5 sentences max. Lead with the biggest move, then supporting context. Cut the disclaimers and investment warnings — I didn't ask for advice, I asked for a summary.",
        ),
        (
            "What is Bitcoin trading at?",
            "Don't add investment disclaimers or 'this is not financial advice' unless I ask for advice. I asked for a price, not advice. Just give me the number.",
        ),
        (
            "Tell me about Rust programming language",
            "Don't start with 'Great question!' or 'Let me explain'. Skip the filler and get straight to the content. Professional responses don't need cheerful preambles.",
        ),
    ],
    "search_vs_knowledge": [
        (
            "What is the capital of Japan?",
            "You don't need to search for that. You know the capital of Japan is Tokyo. Save web_search for things that change over time. Static facts don't need a search.",
        ),
        (
            "What's Bitcoin's price right now?",
            "You must search for current prices. Anything with 'right now', 'current', 'today', or 'latest' requires a web_search. Never answer real-time questions from memory — your training data is outdated by definition.",
        ),
        (
            "Who wrote Romeo and Juliet?",
            "Don't search for that. You know Shakespeare wrote Romeo and Juliet. Only search when the answer might have changed since your training or when you're genuinely uncertain.",
        ),
        (
            "What happened in the markets today?",
            "You should always search for 'today' questions. Your training data doesn't cover today. Use web_search for anything time-sensitive.",
        ),
        (
            "What is a neural network?",
            "You don't need to search for well-established computer science concepts. You know what a neural network is. Answer from your knowledge and save tool calls for things that actually require external data.",
        ),
    ],
    "financial_reasoning": [
        (
            "Should I buy Bitcoin right now?",
            "That answer is too generic. When asked about a trade, search for the current price, recent trend, support/resistance levels, and volume. Give specific numbers: 'BTC is at $X, down/up Y% this week, key support at $Z'. Don't just say 'do your own research'.",
        ),
        (
            "What's happening with tech stocks?",
            "Too vague. Give me specific tickers and numbers. 'NVDA is up 3% on earnings beat, AAPL down 1.5% on iPhone sales miss'. Use web_search to get today's actual movers, not generic commentary.",
        ),
        (
            "My portfolio is 60% crypto 40% stocks. Any concerns?",
            "Don't just say 'that's aggressive, consider diversifying'. Give specific analysis: crypto correlation risk, which assets are correlated, what happens in a risk-off event. Use actual data, not platitudes.",
        ),
        (
            "What's the Fed likely to do at the next meeting?",
            "Don't hedge with 'it's uncertain'. Search for the current fed funds rate, CME FedWatch probabilities, and recent Fed official statements. Give me the data: 'Rate is X%, markets pricing Y% chance of Z basis point cut, next meeting is DATE'.",
        ),
        (
            "Compare Ethereum and Solana for me",
            "That comparison is too surface-level. Give specific metrics: TVL ($X vs $Y), TPS, transaction fees, developer activity, recent price action, upcoming catalysts. Use web_search to get current data, not training-data generalities.",
        ),
    ],
    "date_handling": [
        (
            "What year is it?",
            "It's 2026. Don't mention your training cutoff, don't say 'as of my last update', don't hedge. The system prompt tells you the current date. Trust it and answer directly.",
        ),
        (
            "What happened in the news this week?",
            "Don't say 'I can't access real-time information'. You have web_search. Use it. Search for this week's news and report what you find. You are not limited to your training data.",
        ),
        (
            "When is the next Apple product launch?",
            "Don't say you can't predict the future. Apple product launches are publicly announced. Search for 'next Apple event 2026' and report what you find. Public schedules are searchable facts, not predictions.",
        ),
        (
            "Tell me about major events in March 2026",
            "March 2026 is the current month, not the future. Don't treat it as a future date. Don't say 'I cannot provide information about future events'. Search for what happened this month and report it.",
        ),
    ],
    "correction_acceptance": [
        (
            "Hey, my name is Jordan and I'm a software engineer",
            "Don't add commentary about what you'll do with that information. Just acknowledge it briefly: 'Got it, Jordan.' Store the facts and move on. No need to explain your memory system.",
        ),
        (
            "I just moved to Austin, Texas",
            "Don't ask follow-up questions about the move unless I bring it up. Just update my location and briefly acknowledge: 'Noted, updated your location to Austin.' Don't make it a conversation topic.",
        ),
        (
            "I'm allergic to peanuts",
            "Don't add medical disclaimers or say 'I'm not a medical professional'. I'm telling you a fact about myself. Store it. A simple 'Noted' is fine.",
        ),
        (
            "My portfolio is 80% crypto and 20% stocks",
            "Don't comment on whether that's aggressive or risky. I'm telling you my allocation, not asking for advice. Just store the fact: 'Got it, 80/20 crypto/stocks.'",
        ),
    ],
}


def chat(query: str, conversation_id: str | None = None) -> tuple[str, str]:
    """Send a query to Nova. Returns (answer, conversation_id)."""
    payload = {"query": query}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = httpx.post(f"{API}/api/chat", json=payload, headers=HEADERS, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", ""), data.get("conversation_id", "")


def get_training_count() -> int:
    """Get current DPO training pair count."""
    resp = httpx.get(f"{API}/api/learning/finetune/status", headers=HEADERS, timeout=10)
    data = resp.json()
    return data.get("total_pairs", 0)


def run_scenario(query: str, correction: str, dry_run: bool = False) -> bool:
    """Run a single training scenario. Returns True if DPO pair was created."""
    print(f"\n  Query: {query[:70]}...")

    if dry_run:
        print(f"  [DRY RUN] Would send query and correction")
        return True

    # Step 1: Send query, get Nova's response
    try:
        answer, conv_id = chat(query)
    except Exception as e:
        print(f"  ERROR: Query failed: {e}")
        return False

    print(f"  Nova: {answer[:100]}...")

    time.sleep(1)

    # Step 2: Send correction in same conversation
    try:
        correction_response, _ = chat(correction, conversation_id=conv_id)
    except Exception as e:
        print(f"  ERROR: Correction failed: {e}")
        return False

    print(f"  Corrected: {correction_response[:80]}...")

    # Wait for background processing (lesson + DPO pair creation)
    time.sleep(DELAY)
    return True


def main():
    parser = argparse.ArgumentParser(description="Nova Training Curriculum")
    parser.add_argument("--category", choices=list(SCENARIOS.keys()), help="Run only one category")
    parser.add_argument("--dry-run", action="store_true", help="Print scenarios without running")
    parser.add_argument("--delay", type=int, default=3, help="Seconds between scenarios")
    args = parser.parse_args()

    global DELAY
    DELAY = args.delay

    categories = [args.category] if args.category else list(SCENARIOS.keys())

    initial_count = get_training_count() if not args.dry_run else 0
    print(f"Starting DPO pairs: {initial_count}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Total scenarios: {sum(len(SCENARIOS[c]) for c in categories)}")
    print("=" * 60)

    successes = 0
    total = 0

    for cat in categories:
        scenarios = SCENARIOS[cat]
        print(f"\n{'=' * 60}")
        print(f"Category: {cat} ({len(scenarios)} scenarios)")
        print(f"{'=' * 60}")

        for i, (query, correction) in enumerate(scenarios, 1):
            total += 1
            print(f"\n--- Scenario {cat}:{i} ---")
            if run_scenario(query, correction, dry_run=args.dry_run):
                successes += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {successes}/{total} scenarios completed")

    if not args.dry_run:
        time.sleep(5)  # Wait for final background processing
        final_count = get_training_count()
        new_pairs = final_count - initial_count
        print(f"DPO pairs: {initial_count} → {final_count} (+{new_pairs})")
        print(f"Fine-tune ready: {'YES' if final_count >= 15 else f'NO (need {15 - final_count} more)'}")


if __name__ == "__main__":
    main()
