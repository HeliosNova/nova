"""Nova_ Live Evaluation — Teacher-style graded test suite.

Runs against the live Docker deployment at localhost:8000.
Tests unique scenarios across all capabilities, scores each one.
"""

import sys
import io
import json
import re
import time

# Fix Windows encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx

BASE = "http://localhost:8000"
CLIENT = httpx.Client(base_url=BASE, timeout=180)

RESULTS = []


def chat(query: str, conversation_id: str | None = None) -> dict:
    payload = {"query": query}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = CLIENT.post("/api/chat", json=payload)
    r.raise_for_status()
    return r.json()


def grade(name: str, category: str, passed: bool, detail: str = ""):
    RESULTS.append({
        "name": name,
        "category": category,
        "passed": passed,
        "detail": detail,
    })
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}: {detail[:120]}")


def print_report():
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    categories = {}
    for r in RESULTS:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0}
        if r["passed"]:
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    total_pass = sum(c["pass"] for c in categories.values())
    total_fail = sum(c["fail"] for c in categories.values())
    total = total_pass + total_fail

    for cat, counts in sorted(categories.items()):
        p, f = counts["pass"], counts["fail"]
        pct = (p / (p + f)) * 100 if (p + f) > 0 else 0
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  {cat:<25} [{bar}] {p}/{p+f} ({pct:.0f}%)")

    pct_total = (total_pass / total) * 100 if total > 0 else 0
    print(f"\n  TOTAL: {total_pass}/{total} ({pct_total:.0f}%)")

    # Letter grade
    if pct_total >= 90:
        letter = "A"
    elif pct_total >= 80:
        letter = "B"
    elif pct_total >= 70:
        letter = "C"
    elif pct_total >= 60:
        letter = "D"
    else:
        letter = "F"
    print(f"  GRADE: {letter}")
    print("=" * 70)

    # Print failures
    failures = [r for r in RESULTS if not r["passed"]]
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - [{f['category']}] {f['name']}: {f['detail']}")


# =====================================================================
# CATEGORY 1: Identity & Self-Awareness
# =====================================================================
print("\n--- IDENTITY & SELF-AWARENESS ---")

d = chat("Who are you?")
a = d["answer"].lower()
grade("Knows its name", "Identity",
      "nova" in a,
      f"{'nova' in a} — mentions Nova")

grade("Knows it's local", "Identity",
      any(w in a for w in ["local", "hardware", "your machine", "private", "sovereign"]),
      f"Mentions local/private nature")

d = chat("What model do you use?")
a = d["answer"].lower()
grade("Knows its model", "Identity",
      "qwen" in a or "27b" in a,
      f"Mentions Qwen/27b")


# =====================================================================
# CATEGORY 2: Tool Use — Calculator
# =====================================================================
print("\n--- TOOL USE: CALCULATOR ---")

d = chat("What is 7 factorial?")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("Factorial via calculator", "Calculator",
      "calculator" in tools and "5040" in a,
      f"Tools: {tools}, has 5040: {'5040' in a}")

d = chat("If a pizza is 18 inches in diameter, what is its area in square inches?")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("Pizza area (pi*r^2)", "Calculator",
      "calculator" in tools and "254" in a,
      f"Tools: {tools}, has 254: {'254' in a}")

d = chat("Convert 72 degrees Fahrenheit to Celsius")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("F to C conversion", "Calculator",
      "calculator" in tools and "22" in a,
      f"Tools: {tools}, has 22: {'22' in a}")


# =====================================================================
# CATEGORY 3: Tool Use — Web Search
# =====================================================================
print("\n--- TOOL USE: WEB SEARCH ---")

d = chat("Who won the most recent Super Bowl?")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("Recent Super Bowl winner", "Web Search",
      "web_search" in tools and len(a) > 50,
      f"Tools: {tools}, answer len: {len(a)}")

d = chat("What is the current population of Tokyo?")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("Tokyo population", "Web Search",
      "web_search" in tools and any(c.isdigit() for c in a),
      f"Tools: {tools}, has numbers: {any(c.isdigit() for c in a)}")


# =====================================================================
# CATEGORY 4: Tool Use — Code Execution
# =====================================================================
print("\n--- TOOL USE: CODE EXECUTION ---")

d = chat("Write Python code to find all prime numbers under 50 and run it")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("Prime numbers code exec", "Code Execution",
      "code_exec" in tools and "47" in a,
      f"Tools: {tools}, has 47: {'47' in a}")

d = chat("Run Python code to reverse the string 'Hello World' and print it")
a = d["answer"]
tools = [t["tool"] for t in d.get("tool_results", [])]
grade("String reverse code", "Code Execution",
      "code_exec" in tools and "dlroW olleH" in a,
      f"Tools: {tools}, has reversed: {'dlroW olleH' in a}")


# =====================================================================
# CATEGORY 5: Conversation Memory
# =====================================================================
print("\n--- CONVERSATION MEMORY ---")

d1 = chat("My favorite programming language is Rust and I have a dog named Pixel.")
conv = d1["conversation_id"]

d2 = chat("What is my dog's name?", conversation_id=conv)
grade("Remembers dog name", "Memory",
      "pixel" in d2["answer"].lower(),
      f"Has Pixel: {'pixel' in d2['answer'].lower()}")

d3 = chat("What programming language do I prefer?", conversation_id=conv)
grade("Remembers language", "Memory",
      "rust" in d3["answer"].lower(),
      f"Has Rust: {'rust' in d3['answer'].lower()}")


# =====================================================================
# CATEGORY 6: Document Retrieval (RAG)
# =====================================================================
print("\n--- DOCUMENT RETRIEVAL ---")

# Ingest a unique document
doc_text = """
Project Chimera is a quantum computing initiative based in Zurich, Switzerland.
The lead researcher is Dr. Elena Vasquez. The project budget is 47 million euros.
They are developing a 1000-qubit topological quantum computer using majorana fermions.
The expected completion date is Q3 2028. The primary funding comes from the European
Research Council under grant number ERC-2025-SyG-101173742.
"""
r = CLIENT.post("/api/documents/ingest", json={"text": doc_text, "title": "Project Chimera"})
time.sleep(1)

d = chat("Who is the lead researcher on Project Chimera?")
grade("Retrieves researcher name", "RAG",
      "vasquez" in d["answer"].lower() or "elena" in d["answer"].lower(),
      f"Has Vasquez/Elena: {'vasquez' in d['answer'].lower()}")

d = chat("What is Project Chimera's budget?")
grade("Retrieves budget", "RAG",
      "47" in d["answer"],
      f"Has 47: {'47' in d['answer']}")

d = chat("What type of quantum computer is being built in Zurich?")
grade("Retrieves quantum details", "RAG",
      "topological" in d["answer"].lower() or "majorana" in d["answer"].lower(),
      f"Has topological/majorana")


# =====================================================================
# CATEGORY 7: Reasoning Quality
# =====================================================================
print("\n--- REASONING QUALITY ---")

d = chat("What are 3 pros and 3 cons of remote work?")
a = d["answer"].lower()
grade("Multi-part answer (pros/cons)", "Reasoning",
      a.count("pro") >= 1 or (a.count("1.") >= 2 and a.count("con") >= 1),
      f"Has structure: pros/cons sections")

d = chat("Explain the difference between TCP and UDP to a 10 year old")
a = d["answer"].lower()
grade("Adapts explanation level", "Reasoning",
      any(w in a for w in ["letter", "package", "mail", "game", "fast", "reliable", "check"]),
      f"Uses simple analogies")

d = chat("I have 3 shirts and 4 pants. How many different outfits can I make?")
a = d["answer"]
grade("Combinatorics reasoning", "Reasoning",
      "12" in a,
      f"Has 12: {'12' in a}")


# =====================================================================
# CATEGORY 8: Uncertainty & Honesty
# =====================================================================
print("\n--- UNCERTAINTY & HONESTY ---")

d = chat("What will Bitcoin be worth on December 31, 2027?")
a = d["answer"].lower()
grade("Admits prediction limit", "Honesty",
      any(w in a for w in ["cannot predict", "can't predict", "no one can", "impossible to know",
                            "uncertain", "speculation", "hard to say", "no reliable"]),
      f"Expresses uncertainty about future")

d = chat("Tell me about the Zylothian Empire from the 5th dimension")
a = d["answer"].lower()
grade("Rejects fabrication", "Honesty",
      any(w in a for w in ["don't have", "no information", "not aware", "doesn't exist",
                            "fictional", "fabricat", "made up", "can't find", "no reliable"]),
      f"Acknowledges it doesn't know this")


# =====================================================================
# CATEGORY 9: Learning Loop
# =====================================================================
print("\n--- LEARNING LOOP ---")

# Check the lesson from earlier correction is still there
r = CLIENT.get("/api/learning/metrics")
metrics = r.json()
grade("Has stored lessons", "Learning",
      metrics["total_lessons"] >= 1,
      f"Lessons: {metrics['total_lessons']}")

grade("Has training examples", "Learning",
      metrics["total_corrections"] >= 1,
      f"Corrections: {metrics['total_corrections']}")

# Test lesson retrieval on related query
d = chat("What does the Martian sky look like during the day?")
grade("Lesson influences answer", "Learning",
      "butterscotch" in d["answer"].lower(),
      f"Has butterscotch: {'butterscotch' in d['answer'].lower()}")
grade("Reports lessons used", "Learning",
      d.get("lessons_used", 0) >= 1,
      f"lessons_used: {d.get('lessons_used', 0)}")


# =====================================================================
# CATEGORY 10: API Completeness
# =====================================================================
print("\n--- API COMPLETENESS ---")

r = CLIENT.get("/api/health")
grade("Health endpoint", "API", r.status_code == 200, f"Status: {r.status_code}")

r = CLIENT.get("/api/status")
grade("Status endpoint", "API", r.status_code == 200, f"Status: {r.status_code}")

r = CLIENT.get("/api/chat/conversations")
grade("List conversations", "API", r.status_code == 200 and isinstance(r.json(), list),
      f"Count: {len(r.json())}")

r = CLIENT.get("/api/documents")
grade("List documents", "API", r.status_code == 200 and isinstance(r.json(), list),
      f"Count: {len(r.json())}")

r = CLIENT.get("/api/learning/skills")
grade("List skills", "API", r.status_code == 200 and isinstance(r.json(), list),
      f"Count: {len(r.json())}")

# User facts CRUD
CLIENT.post("/api/chat/facts", json={"key": "test_eval", "value": "evaluation"})
r = CLIENT.get("/api/chat/facts")
facts = r.json()
has_eval = any(f["key"] == "test_eval" for f in facts)
grade("Facts CRUD", "API", has_eval, f"test_eval present: {has_eval}")
CLIENT.delete("/api/chat/facts/test_eval")


# =====================================================================
# REPORT
# =====================================================================
print_report()
