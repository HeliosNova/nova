"""A/B comparison: Q8_0 vs fp16 — proper methodology.

- Warmup query on each model before timing
- 300s timeout (not 120)
- Measures speed, length, tool use, and answer content
- Unbuffered output
"""
import asyncio
import sys
import time
import json
import subprocess
import httpx

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)

QUERIES = [
    "What is the capital of Australia?",
    "What is 15% of 4250 dollars?",
    "Compare Rust and Go for building web APIs in a table",
    "What is Bitcoin trading at right now?",
    "Search for the latest AI model releases this week and summarize the top 3",
    "Who created Python?",
    "What do you know about Ethereum from your knowledge graph?",
    "Write a Python function to check if a string is a valid email address",
]

API_URL = "http://localhost:8000"
TIMEOUT = 300

async def run_query(client, query, api_key):
    start = time.time()
    try:
        resp = await client.post(
            f"{API_URL}/api/chat",
            json={"query": query, "conversation_id": f"ab-{hash(query) % 99999}"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            return {
                "time": round(elapsed, 1),
                "length": len(data.get("answer", "")),
                "tools": [t["tool"] for t in data.get("tool_results", [])],
                "answer": data.get("answer", "")[:200],
            }
        return {"time": round(elapsed, 1), "length": 0, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"time": round(time.time() - start, 1), "length": 0, "error": str(e)[:80]}

async def switch_model(client, model, api_key):
    resp = await client.patch(
        f"{API_URL}/api/config",
        json={"LLM_MODEL": model},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    print(f"  Switched to {model}: {resp.json()}", flush=True)
    await asyncio.sleep(3)

async def warmup(client, model, api_key):
    print(f"  Warming up {model}...", flush=True)
    r = await run_query(client, "hello", api_key)
    print(f"  Warmup done ({r['time']}s)", flush=True)

async def main():
    api_key = subprocess.check_output(
        ["docker", "exec", "nova-app", "python", "-c",
         "from app.config import config; print(config.API_KEY)"],
        text=True,
    ).strip()

    models = [
        ("nova-ft-v8-q8", "Q8_0 (9.5GB, 128K ctx)"),
        ("nova-ft-v8-fp16", "fp16 (17.9GB, ~32K ctx)"),
    ]

    all_results = {}

    async with httpx.AsyncClient() as client:
        for model_id, label in models:
            print(f"\n{'='*70}", flush=True)
            print(f"MODEL: {label}", flush=True)
            print(f"{'='*70}", flush=True)

            await switch_model(client, model_id, api_key)
            await warmup(client, model_id, api_key)

            results = []
            for q in QUERIES:
                r = await run_query(client, q, api_key)
                results.append(r)
                status = f"{r['length']:>5} chars" if r['length'] else f"ERROR: {r.get('error','?')[:40]}"
                print(f"  [{r['time']:>6.1f}s] {status} | {q[:55]}", flush=True)

            all_results[model_id] = results

        # Switch back to Q8
        await switch_model(client, "nova-ft-v8-q8", api_key)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    for model_id, label in models:
        results = all_results[model_id]
        completed = [r for r in results if r['length'] > 0]
        avg_time = sum(r['time'] for r in completed) / max(len(completed), 1)
        avg_len = sum(r['length'] for r in completed) / max(len(completed), 1)
        errors = sum(1 for r in results if r['length'] == 0)
        print(f"\n{label}:")
        print(f"  Completed: {len(completed)}/{len(results)}")
        print(f"  Avg time (completed only): {avg_time:.1f}s")
        print(f"  Avg length: {avg_len:.0f} chars")
        print(f"  Errors/timeouts: {errors}")

    # Side by side
    print(f"\n{'Query':<45} {'Q8 time':>8} {'fp16 time':>9} {'Q8 len':>7} {'fp16 len':>8}", flush=True)
    print("-" * 80, flush=True)
    q8 = all_results[models[0][0]]
    fp = all_results[models[1][0]]
    for i, query in enumerate(QUERIES):
        qt = f"{q8[i]['time']:.1f}s"
        ft = f"{fp[i]['time']:.1f}s"
        ql = str(q8[i]['length']) if q8[i]['length'] else "ERR"
        fl = str(fp[i]['length']) if fp[i]['length'] else "ERR"
        print(f"{query[:45]:<45} {qt:>8} {ft:>9} {ql:>7} {fl:>8}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
