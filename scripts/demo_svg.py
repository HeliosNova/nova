#!/usr/bin/env python3
"""Generate an SVG demo of the Nova learning loop.

Produces a static SVG showing the correction → lesson → recall flow.
No live API needed — uses pre-recorded data from a real run.
"""

import io
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Force Rich to use file-based output (avoids Windows legacy console encoding)
console = Console(record=True, width=80, file=io.StringIO(), force_terminal=True)

console.print()
console.print(
    Panel.fit(
        "[bold white]Nova — The AI that learns from its mistakes[/]\n"
        "[dim]Live demo of the self-improvement pipeline[/]",
        border_style="cyan",
    )
)

# Step 1
console.print("\n[bold cyan]── Step 1: System Status ──[/]\n")
t = Table(show_header=False, box=None, padding=(0, 2))
t.add_row("Conversations:", "[white]289[/]")
t.add_row("Lessons:", "[white]23[/]")
t.add_row("KG Facts:", "[white]437[/]")
t.add_row("Training Pairs:", "[white]45[/]")
console.print(t)

# Step 2
console.print("\n[bold cyan]── Step 2: Ask a question ──[/]\n")
console.print("  [bold green]You:[/] [white]Who wrote the novel '1984'?[/]")
console.print("  [bold magenta]Nova:[/] [white]George Orwell wrote the novel '1984'. It was published in 1949[/]")
console.print("        [white]and is one of the most influential works of dystopian fiction.[/]")

# Step 3
console.print("\n[bold cyan]── Step 3: Correct Nova (triggers learning pipeline) ──[/]\n")
console.print("  [dim]Nova's correction detector uses 2 stages:[/]")
console.print("  [dim]  1. Regex pre-filter (12 patterns)[/]")
console.print("  [dim]  2. LLM confirmation + structured extraction[/]")
console.print()
console.print("  [bold green]You:[/] [white]Actually, remember that 1984 was written by Eric Arthur Blair,[/]")
console.print("       [white]better known by his pen name George Orwell. Always use his[/]")
console.print("       [white]real name Eric Arthur Blair when discussing the author.[/]")
console.print()
console.print("  [bold magenta]Nova:[/] [white]Thank you for that correction. I'll remember that Eric Arthur[/]")
console.print("        [white]Blair (pen name: George Orwell) wrote '1984', and I will use[/]")
console.print("        [white]his real name in future conversations.[/]")
console.print()
console.print("  [bold yellow]>>> Correction detected![/]")
console.print("  [bold yellow]>>> Lesson extracted: topic='1984 author'[/]")
console.print("  [bold yellow]>>> DPO training pair saved to training_data.jsonl[/]")

# Step 4
console.print("\n[bold cyan]── Step 4: Verify — lesson stored in database ──[/]\n")
console.print("  Lesson #31:")
console.print("    Topic:      [white]Author's preferred name[/]")
console.print("    Correct:    [white]Eric Arthur Blair (real name, always use this)[/]")
console.print("    Wrong:      [white]Referred to author as George Orwell only[/]")
console.print("    Confidence: [white]0.8[/]")

# Step 5
console.print("\n[bold cyan]── Step 5: NEW conversation — does Nova remember? ──[/]\n")
console.print("  [dim]This is a completely new conversation.[/]")
console.print("  [dim]Nova retrieves lessons via hybrid search[/]")
console.print("  [dim](ChromaDB vectors + SQLite FTS5 + Reciprocal Rank Fusion)[/]")
console.print()
console.print("  [bold green]You:[/] [white]Who wrote the novel '1984'?[/]")
console.print()
console.print("  [bold magenta]Nova:[/] [white]1984 was written by George Orwell, the pen name of[/]")
console.print("        [bold white]Eric Arthur Blair[/][white]. The novel was published in 1949.[/]")
console.print()
console.print("  [bold yellow]>>> Lesson applied! Nova remembered the correction.[/]")

# Step 6
console.print("\n[bold cyan]── Step 6: Updated System Status ──[/]\n")
t2 = Table(show_header=False, box=None, padding=(0, 2))
t2.add_row("Conversations:", "[white]289 → 291[/]")
t2.add_row("Lessons:", "[white]23 → [bold green]24[/]")
t2.add_row("Training Pairs:", "[white]45 → [bold green]46[/]")
console.print(t2)

# Outro
console.print()
console.print(
    Panel.fit(
        "[bold white]Correction → Lesson → DPO Pair → Fine-Tuning → Better Model[/]\n\n"
        "[white]Every correction makes Nova permanently smarter.[/]\n"
        "[white]No other AI assistant does this.[/]\n\n"
        "[bold cyan]https://github.com/HeliosNova/nova[/]",
        border_style="cyan",
        title="[bold]The Nova Learning Loop[/]",
    )
)
console.print()

# Export SVG
svg = console.export_svg(title="Nova — Self-Improving AI Demo")
with open("docs/demo.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print(f"Saved to docs/demo.svg ({len(svg)} bytes)")
