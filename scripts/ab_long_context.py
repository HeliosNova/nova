"""Long context test: Q8 (128K) vs fp16 (32K).

Builds a conversation with 20+ turns of history, then asks a question
that requires remembering earlier context. Tests what the bigger
context window actually buys you.
"""
import asyncio
import sys
import time
import subprocess
import httpx

sys.stdout.reconfigure(line_buffering=True)

API_URL = "http://localhost:8000"
TIMEOUT = 300

# 20 conversation turns to fill context
HISTORY = [
    "My name is Alex and I work at a fintech startup in Austin",
    "We're building a trading platform using Python and FastAPI",
    "Our main database is PostgreSQL with TimescaleDB for time series",
    "We use Redis for caching and Celery for task queues",
    "The frontend is React with TypeScript and TanStack Query",
    "We deploy on AWS EKS with Terraform for infrastructure",
    "Our biggest challenge right now is latency on order execution",
    "We're seeing 200ms p99 latency and need to get it under 50ms",
    "The bottleneck is between our order service and the exchange API",
    "We tried connection pooling but it only helped a little",
    "Our team has 5 engineers and we're hiring 2 more backend devs",
    "The CEO wants to add crypto trading by Q3",
    "We're evaluating ccxt vs building our own exchange connectors",
    "Security is critical - we need SOC2 compliance by end of year",
    "We had a near-miss incident last month with a race condition in order matching",
    "Our monitoring uses Datadog but we're over budget on it",
    "I personally prefer Grafana + Prometheus for cost reasons",
    "We're also looking at moving from REST to gRPC for internal services",
    "The order matching engine is the most critical component",
    "It needs to handle 10k orders per second at peak",
]

# Question that requires remembering the full conversation
RECALL_QUESTION = (
    "Based on everything I've told you about our company, team, and technical challenges, "
    "give me a prioritized action plan for the next quarter. Reference specific details "
    "I mentioned — the latency target, the crypto deadline, the SOC2 requirement, "
    "the monitoring budget issue, and the hiring plan."
)

async def run_test(client, model_id, label, api_key):
    print(f"\n{'='*70}", flush=True)
    print(f"MODEL: {label}", flush=True)
    print(f"{'='*70}", flush=True)

    # Switch model
    await client.patch(
        f"{API_URL}/api/config",
        json={"LLM_MODEL": model_id},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await asyncio.sleep(3)

    # Warmup
    print("  Warming up...", flush=True)
    await client.post(
        f"{API_URL}/api/chat",
        json={"query": "hello", "conversation_id": f"ctx-warmup-{model_id}"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    print("  Warm.", flush=True)

    conv_id = f"ctx-long-{model_id}"

    # Build conversation history
    print(f"  Sending {len(HISTORY)} history messages...", flush=True)
    for i, msg in enumerate(HISTORY):
        r = await client.post(
            f"{API_URL}/api/chat",
            json={"query": msg, "conversation_id": conv_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        status = r.status_code
        data = r.json() if status == 200 else {}
        alen = len(data.get("answer", ""))
        print(f"    [{i+1:>2}/{len(HISTORY)}] {status} | {alen:>4} chars | {msg[:50]}", flush=True)

    # Now the real test — ask the recall question
    print(f"\n  === RECALL TEST ===", flush=True)
    print(f"  Q: {RECALL_QUESTION[:80]}...", flush=True)
    start = time.time()
    r = await client.post(
        f"{API_URL}/api/chat",
        json={"query": RECALL_QUESTION, "conversation_id": conv_id},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    elapsed = time.time() - start
    data = r.json() if r.status_code == 200 else {}
    answer = data.get("answer", "")

    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"  Length: {len(answer)} chars", flush=True)
    print(f"  Answer:", flush=True)
    print(f"  {answer[:2000]}", flush=True)

    # Check which details were recalled
    details = {
        "latency (50ms)": "50" in answer or "50ms" in answer.lower(),
        "crypto Q3": "q3" in answer.lower() or "crypto" in answer.lower(),
        "SOC2": "soc2" in answer.lower() or "soc 2" in answer.lower(),
        "Datadog budget": "datadog" in answer.lower() or "monitoring" in answer.lower(),
        "hiring (2 backend)": "hiring" in answer.lower() or "2 " in answer.lower(),
        "gRPC migration": "grpc" in answer.lower(),
        "10k orders/sec": "10k" in answer.lower() or "10,000" in answer,
        "ccxt": "ccxt" in answer.lower(),
        "Grafana": "grafana" in answer.lower(),
        "race condition": "race" in answer.lower(),
    }
    recalled = sum(1 for v in details.values() if v)
    print(f"\n  Details recalled: {recalled}/{len(details)}", flush=True)
    for detail, found in details.items():
        print(f"    {'Y' if found else 'N'} {detail}", flush=True)

    return {"time": elapsed, "length": len(answer), "recalled": recalled, "total": len(details)}

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

    results = {}
    async with httpx.AsyncClient() as client:
        for model_id, label in models:
            results[model_id] = await run_test(client, model_id, label, api_key)

        # Switch back to Q8
        await client.patch(
            f"{API_URL}/api/config",
            json={"LLM_MODEL": "nova-ft-v8-q8"},
            headers={"Authorization": f"Bearer {api_key}"},
        )

    print(f"\n{'='*70}", flush=True)
    print("LONG CONTEXT COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    for model_id, label in models:
        r = results[model_id]
        print(f"\n{label}:")
        print(f"  Time: {r['time']:.1f}s")
        print(f"  Answer length: {r['length']} chars")
        print(f"  Details recalled: {r['recalled']}/{r['total']}")

if __name__ == "__main__":
    asyncio.run(main())
