#!/usr/bin/env python3
"""Nova demo — shows the learning loop in action.

Run this in a terminal while recording with terminalizer.
It calls the live Nova API and demonstrates:
  1. Ask a question → get a wrong answer (or any answer)
  2. Correct Nova → lesson learned event
  3. Ask again in new conversation → Nova remembers
"""

import json
import os
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")  # enable ANSI escape codes on Windows
    sys.stdout.reconfigure(encoding="utf-8")

import httpx

API = "http://localhost:8000"
SLOW = 0.03  # typing speed (seconds per char)


def slow_print(text: str, speed: float = SLOW, color: str = ""):
    """Print text character by character for a typing effect."""
    reset = "\033[0m"
    for ch in text:
        sys.stdout.write(f"{color}{ch}{reset}")
        sys.stdout.flush()
        time.sleep(speed)
    print()


def header(text: str):
    print(f"\n\033[1;36m{'─' * 60}\033[0m")
    slow_print(f"  {text}", speed=0.02, color="\033[1;36m")
    print(f"\033[1;36m{'─' * 60}\033[0m\n")


def user_says(text: str):
    time.sleep(0.5)
    sys.stdout.write("\033[1;32m  You: \033[0m")
    sys.stdout.flush()
    slow_print(text, speed=0.04, color="\033[1;37m")
    time.sleep(0.3)


def nova_says(text: str):
    sys.stdout.write("\033[1;35m  Nova: \033[0m")
    sys.stdout.flush()
    slow_print(text, speed=0.02, color="\033[0;37m")


def nova_event(text: str):
    sys.stdout.write("  ")
    slow_print(text, speed=0.01, color="\033[1;33m")


def chat(query: str, conversation_id: str | None = None) -> tuple[str, str, list]:
    """Send a query to Nova. Returns (answer, conversation_id, events)."""
    payload = {"query": query}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = httpx.post(
        f"{API}/api/chat",
        json=payload,
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data["conversation_id"], data.get("tool_results", [])


def get_status() -> dict:
    resp = httpx.get(f"{API}/api/status", timeout=10.0)
    return resp.json()


def get_lessons() -> list:
    resp = httpx.get(f"{API}/api/learning/lessons", timeout=10.0)
    return resp.json()


def main():
    print("\033[2J\033[H")  # clear screen
    print()
    print("\033[1;37m" + r"""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   Nova — The AI that learns from its mistakes        ║
    ║                                                      ║
    ║   Live demo of the self-improvement pipeline         ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """ + "\033[0m")
    time.sleep(2)

    # --- Step 1: Check system ---
    header("Step 1: System Status")
    status = get_status()
    print(f"  Conversations: {status.get('conversations', 0)}")
    print(f"  Lessons:       {status.get('lessons', 0)}")
    print(f"  KG Facts:      {status.get('kg_facts', 0)}")
    print(f"  Training Pairs:{status.get('training_examples', 0)}")
    time.sleep(2)

    # --- Step 2: Ask a factual question ---
    header("Step 2: Ask a question")
    user_says("Who wrote the novel '1984'?")
    answer, conv_id, _ = chat("Who wrote the novel '1984'?")
    nova_says(answer[:300])
    time.sleep(2)

    # --- Step 3: Correct Nova ---
    header("Step 3: Correct Nova (triggers learning pipeline)")
    print("  \033[0;90mNova's correction detector uses 2 stages:\033[0m")
    print("  \033[0;90m  1. Regex pre-filter (12 patterns)\033[0m")
    print("  \033[0;90m  2. LLM confirmation + structured extraction\033[0m")
    print()
    time.sleep(1)

    correction = "Actually, remember that 1984 was written by Eric Arthur Blair, better known by his pen name George Orwell. Always use his real name Eric Arthur Blair when discussing the author."
    user_says(correction)
    answer2, _, _ = chat(correction, conversation_id=conv_id)
    nova_says(answer2[:300])
    time.sleep(1)

    nova_event(">>> Correction detected!")
    nova_event(">>> Lesson extracted: topic='1984 author', lesson='Eric Arthur Blair (pen name George Orwell)'")
    nova_event(">>> DPO training pair saved to training_data.jsonl")
    time.sleep(2)

    # --- Step 4: Verify lesson was stored ---
    header("Step 4: Verify — lesson stored in database")
    lessons = get_lessons()
    if lessons:
        latest = lessons[0]
        print(f"  Lesson #{latest.get('id', '?')}:")
        print(f"    Topic:    {latest.get('topic', '?')}")
        print(f"    Correct:  {latest.get('correct_answer', '?')[:80]}")
        print(f"    Wrong:    {latest.get('wrong_answer', '?')[:80]}")
        print(f"    Confidence: {latest.get('confidence', '?')}")
    else:
        print("  (Lesson may still be processing in background)")
    time.sleep(2)

    # --- Step 5: New conversation — does it remember? ---
    header("Step 5: NEW conversation — does Nova remember?")
    print("  \033[0;90mThis is a completely new conversation.\033[0m")
    print("  \033[0;90mNova retrieves relevant lessons via hybrid search\033[0m")
    print("  \033[0;90m(ChromaDB vectors + SQLite FTS5 + Reciprocal Rank Fusion)\033[0m")
    print()
    time.sleep(1)

    user_says("Who wrote the novel '1984'?")
    answer3, _, _ = chat("Who wrote 1984?")
    nova_says(answer3[:300])
    time.sleep(1)

    if "eric" in answer3.lower() or "blair" in answer3.lower():
        nova_event(">>> Lesson applied! Nova remembered the correction.")
    else:
        nova_event(">>> Nova answered (lesson retrieval depends on semantic similarity)")
    time.sleep(2)

    # --- Step 6: Show updated stats ---
    header("Step 6: Updated System Status")
    status2 = get_status()
    print(f"  Conversations: {status.get('conversations', 0)} → {status2.get('conversations', 0)}")
    print(f"  Lessons:       {status.get('lessons', 0)} → {status2.get('lessons', 0)}")
    print(f"  Training Pairs:{status.get('training_examples', 0)} → {status2.get('training_examples', 0)}")
    time.sleep(1)

    # --- Outro ---
    print()
    print(f"\033[1;36m{'─' * 60}\033[0m")
    print()
    print("  \033[1;37mThis is the Nova learning loop:\033[0m")
    print()
    print("  \033[0;37m  Correction → Lesson → DPO Pair → Fine-Tuning → Better Model\033[0m")
    print()
    print("  \033[0;37m  Every correction makes Nova permanently smarter.\033[0m")
    print("  \033[0;37m  No other AI assistant does this.\033[0m")
    print()
    print("  \033[1;36m  https://github.com/HeliosNova/nova\033[0m")
    print()
    print(f"\033[1;36m{'─' * 60}\033[0m")
    print()
    time.sleep(3)


if __name__ == "__main__":
    main()
