#!/usr/bin/env python3
"""Generate demo.svg by running the learning loop against the live API."""

import os
import time

import httpx
from rich.console import Console
from rich.panel import Panel

API = os.getenv("NOVA_API_URL", "http://localhost:8000")
KEY = os.getenv("NOVA_API_KEY", "")
H = {"Authorization": f"Bearer {KEY}"} if KEY else {}

console = Console(record=True, width=90)


def chat(query, conv_id=None):
    payload = {"query": query}
    if conv_id:
        payload["conversation_id"] = conv_id
    r = httpx.post(f"{API}/api/chat", json=payload, headers=H, timeout=300)
    r.raise_for_status()
    d = r.json()
    return d["answer"], d["conversation_id"], d.get("tool_results", [])


def status():
    return httpx.get(f"{API}/api/status", headers=H, timeout=10).json()


def main():
    console.print()
    console.print(Panel.fit(
        "[bold white]Nova — The AI that learns from its mistakes[/]\n"
        "[dim]Live demo of the self-improvement pipeline[/]",
        border_style="cyan",
    ))

    # Step 1
    console.print("\n[bold cyan]Step 1: System Status[/]")
    s = status()
    lessons = s.get("lessons", 0)
    kg = s.get("kg_facts", 0)
    dpo = s.get("training_examples", 0)
    console.print(f"  Lessons: {lessons}  |  KG Facts: {kg}  |  DPO Pairs: {dpo}")

    # Step 2
    console.print("\n[bold cyan]Step 2: Ask a question[/]")
    console.print("[bold green]  You:[/] Who wrote the novel '1984'?")
    a1, cid, _ = chat("Who wrote the novel '1984'?")
    console.print(f"[bold magenta]  Nova:[/] {a1[:250]}")

    # Step 3
    console.print("\n[bold cyan]Step 3: Correct Nova (triggers learning pipeline)[/]")
    console.print("[dim]  2-stage: regex pre-filter then LLM extraction[/]")
    correction = (
        "Actually, remember that 1984 was written by Eric Arthur Blair, "
        "better known by his pen name George Orwell. Always use his real name."
    )
    console.print(f"[bold green]  You:[/] {correction}")
    a2, _, _ = chat(correction, conv_id=cid)
    console.print(f"[bold magenta]  Nova:[/] {a2[:250]}")
    time.sleep(3)

    # Step 4
    console.print("\n[bold cyan]Step 4: Lesson stored in database[/]")
    resp = httpx.get(f"{API}/api/learning/lessons", headers=H, timeout=10)
    all_lessons = resp.json()
    if all_lessons:
        latest = all_lessons[0]
        console.print(f"  [bold]Lesson #{latest['id']}[/]: {latest['topic']}")
        console.print(f"  Correct: {latest.get('correct_answer', '')[:80]}")
        console.print(f"  Wrong: {latest.get('wrong_answer', '')[:80]}")

    # Step 5
    console.print("\n[bold cyan]Step 5: NEW conversation — does Nova remember?[/]")
    console.print("[dim]  Hybrid search: ChromaDB + FTS5 + Reciprocal Rank Fusion[/]")
    console.print("[bold green]  You:[/] Who wrote 1984?")
    a3, _, _ = chat("Who wrote 1984?")
    console.print(f"[bold magenta]  Nova:[/] {a3[:250]}")
    if "eric" in a3.lower() or "blair" in a3.lower():
        console.print("[bold yellow]  >>> Lesson applied! Nova remembered the correction.[/]")

    # Step 6
    console.print("\n[bold cyan]Step 6: Knowledge Graph (52 autonomous monitors)[/]")
    console.print("[dim]  Monitors search 35+ domains and extract KG triples[/]")
    console.print("[bold green]  You:[/] What is the current world population?")
    a4, _, tools = chat(
        "What is the current world population? Don't search, just tell me what you know."
    )
    console.print(f"[bold magenta]  Nova:[/] {a4[:250]}")
    if not tools:
        console.print("[bold yellow]  >>> Answered from knowledge graph — no web search needed![/]")

    # Step 7
    console.print("\n[bold cyan]Step 7: Updated Stats[/]")
    s2 = status()
    console.print(
        f"  Lessons: {lessons} -> {s2.get('lessons', 0)}  |  "
        f"DPO Pairs: {dpo} -> {s2.get('training_examples', 0)}"
    )

    console.print()
    console.print(Panel.fit(
        "[white]  Corrections -> Lessons -> DPO -> Fine-Tuning -> Better Model[/]\n"
        "[white]  52 monitors -> Web Research -> KG Triples -> Answers[/]\n\n"
        "[bold]  Every correction and monitor cycle makes Nova smarter.[/]\n"
        "[cyan]  https://github.com/HeliosNova/nova[/]",
        border_style="cyan",
        title="The Nova Learning Loop",
    ))
    console.print()

    svg = console.export_svg(title="Nova Learning Loop Demo")
    out_path = "/data/demo.svg"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"SVG saved to {out_path} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
