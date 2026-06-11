"""Multi-agent ceiling probe — HTTP/SSE against the live service.

Rather than measuring "what breaks", this measures useful-work-per-target
across progressively larger parallel-decomposable queries. Runs against the
running Nova instance; no in-process imports required.

For each N in [3, 5, 7, 10]:
  1. Build a parallel query with N distinct canonical targets
  2. POST /api/chat/stream, collect DONE (decomposed, agent_count) + merged text
  3. Score coverage = (# canonical targets mentioned) / N
  4. Also track: tools used, wall time, char length
  5. Flag ceiling = first N where coverage < 0.9 OR decomposed==False

Config context: at runtime MAX_AGENT_COUNT=5 caps sub-agents; for N>5 each
sub-agent gets multiple entities, so we learn what happens when the work
is compressed into fewer agents than targets.

Run: python scripts/multi_agent_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


API_URL = os.getenv("NOVA_API_URL", "http://localhost:8000")
API_KEY = os.getenv("NOVA_API_KEY", "")
# Fall back to .env if the key wasn't passed
if not API_KEY:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("NOVA_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break


@dataclass
class Probe:
    n: int
    query: str
    targets: list[str]
    aliases: dict[str, list[str]] = field(default_factory=dict)


PROBES: list[Probe] = [
    # Queries use "compare ... side by side" + multiple question marks to
    # deliberately hit the decomposition signal threshold (>=3 points).
    Probe(
        n=3,
        query=(
            "Compare Python, Ruby, and Perl side by side. What does each do best? "
            "Who created each? When was each released?"
        ),
        targets=["Python", "Ruby", "Perl"],
    ),
    Probe(
        n=5,
        query=(
            "Compare these in-memory caches side by side: Redis, Memcached, Valkey, "
            "DragonflyDB, and KeyDB. What is each best at? What is the distinctive "
            "feature of each? How does each differ from the others?"
        ),
        targets=["Redis", "Memcached", "Valkey", "DragonflyDB", "KeyDB"],
        aliases={"KeyDB": ["KeyDB", "Keydb"], "DragonflyDB": ["DragonflyDB", "Dragonfly"]},
    ),
    Probe(
        n=7,
        query=(
            "Compare these programming languages side by side: Python, Go, Rust, "
            "TypeScript, Kotlin, Swift, and Ruby. What is each best at? What is "
            "each's typing discipline? Who uses each most today?"
        ),
        targets=["Python", "Go", "Rust", "TypeScript", "Kotlin", "Swift", "Ruby"],
        aliases={"Go": ["Go", "Golang"], "TypeScript": ["TypeScript", "TS"]},
    ),
    Probe(
        n=10,
        query=(
            "Compare these L1 blockchains side by side: Cosmos, Polkadot, Solana, "
            "Avalanche, Fantom, Near, Aptos, Sui, Sei, and Celestia. What consensus "
            "does each use? What is each's distinctive architecture? Who uses each?"
        ),
        targets=[
            "Cosmos", "Polkadot", "Solana", "Avalanche", "Fantom",
            "Near", "Aptos", "Sui", "Sei", "Celestia",
        ],
    ),
]


def coverage(answer: str, targets: list[str], aliases: dict[str, list[str]]) -> tuple[int, list[str]]:
    hit, missing = 0, []
    for t in targets:
        candidates = aliases.get(t, [t])
        if any(re.search(rf"\b{re.escape(c)}\b", answer, re.IGNORECASE) for c in candidates):
            hit += 1
        else:
            missing.append(t)
    return hit, missing


def sse_stream(query: str, timeout: int = 300) -> dict:
    """POST /api/chat/stream and parse SSE events. Blocks until DONE or ERROR."""
    body = json.dumps({"query": query, "conversation_id": None}).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/api/chat/stream",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}),
        },
    )

    tokens: list[str] = []
    tool_names: list[str] = []
    decomposed = False
    agent_count = 0
    error = ""
    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = ""
            for raw in resp:
                chunk = raw.decode("utf-8", errors="replace")
                buf += chunk
                while "\n\n" in buf:
                    event_text, buf = buf.split("\n\n", 1)
                    event_type = "message"
                    data_text = ""
                    for line in event_text.splitlines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_text += line[5:].lstrip()
                    if not data_text or data_text == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    et = event_type.lower()
                    if et == "token":
                        tokens.append(payload.get("text", "") or payload.get("content", ""))
                    elif et == "tool_use":
                        tool = payload.get("tool") or payload.get("name")
                        if tool:
                            tool_names.append(tool)
                    elif et == "done":
                        decomposed = bool(payload.get("decomposed"))
                        agent_count = int(payload.get("agent_count", 0) or 0)
                        break
                    elif et == "error":
                        error = payload.get("message", "")
                        break
                if error:
                    break
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        error = f"HTTP error: {e}"

    elapsed = time.perf_counter() - start
    return {
        "answer": "".join(tokens).strip(),
        "tools_invoked": sorted(set(tool_names)),
        "tool_calls_total": len(tool_names),
        "decomposed": decomposed,
        "agent_count": agent_count,
        "wall_seconds": round(elapsed, 2),
        "error": error or None,
    }


def run_probe(probe: Probe) -> dict:
    r = sse_stream(probe.query)
    hit, missing = coverage(r["answer"], probe.targets, probe.aliases)
    return {
        "n": probe.n,
        **r,
        "answer_chars": len(r["answer"]),
        "coverage_hit": hit,
        "coverage_total": probe.n,
        "coverage_ratio": round(hit / probe.n, 3) if probe.n else 0.0,
        "missing_targets": missing,
        "chars_per_agent": round(len(r["answer"]) / max(r["agent_count"], 1), 1),
        "answer_preview": r["answer"][:600],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--out", default="ceiling_report.json")
    args = ap.parse_args()

    selected = [p for p in PROBES if p.n <= args.max]
    print(f"Running {len(selected)} probes: N={[p.n for p in selected]}", flush=True)

    results = []
    for probe in selected:
        print(f"\n--- N={probe.n}: {probe.query[:80]}...", flush=True)
        r = run_probe(probe)
        print(
            f"    decomposed={r['decomposed']} agents={r['agent_count']} "
            f"coverage={r['coverage_hit']}/{r['coverage_total']} "
            f"chars={r['answer_chars']} wall={r['wall_seconds']:.1f}s"
            + (f" ERROR={r['error']}" if r.get("error") else ""),
            flush=True,
        )
        if r["missing_targets"]:
            print(f"    missing: {r['missing_targets']}", flush=True)
        results.append(r)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "api_url": API_URL,
        "probes": results,
    }
    Path(args.out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nReport: {args.out}")

    print("\n" + "=" * 80)
    print(f"{'N':>3} {'decomp':>7} {'agents':>7} {'coverage':>10} {'chars':>6} {'wall':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['n']:>3} {str(r['decomposed']):>7} {r['agent_count']:>7} "
            f"{r['coverage_hit']}/{r['coverage_total']:>8} {r['answer_chars']:>6} "
            f"{r['wall_seconds']:>7.1f}s"
        )
    print("=" * 80)

    # Ceiling = first N where coverage drops below 0.9 OR decomposition fails
    ceiling = None
    for r in results:
        if r["coverage_ratio"] < 0.9 or not r["decomposed"]:
            ceiling = r["n"]
            break
    if ceiling is None:
        print(f"Ceiling: not reached at N={results[-1]['n']}")
    else:
        print(f"Ceiling: useful work degrades at N={ceiling}")


if __name__ == "__main__":
    main()
