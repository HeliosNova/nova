"""DPO training pairs for the 5 OpenClaude-inspired features.

Generates pairs that teach the model to:
1. Use context_detail tool for lazy context retrieval
2. Understand auto-background promotion messages
3. Respect isolation boundaries
4. Use summary format [L/K/R/D + ID]
5. Respond to event-driven triggers appropriately

Run: python scripts/dpo_openclaude_features.py
Appends to /data/training_data.jsonl
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

TRAINING_DATA_PATH = os.environ.get("TRAINING_DATA_PATH", "/data/training_data.jsonl")

PAIRS = [
    # ============================================================
    # Category 1: Lazy Context Retrieval — context_detail tool usage
    # ============================================================
    {
        "query": "What's the lesson about Python decorators?",
        "chosen": 'I see a lesson summary about Python decorators in my context. Let me get the full details.\n\n{"tool": "context_detail", "args": {"category": "lesson", "item_id": 42}}\n\nBased on the full lesson: Decorators in Python are functions that modify other functions. The key insight is that `@decorator` is syntactic sugar for `func = decorator(func)`. Always use `functools.wraps` to preserve the original function\'s metadata.',
        "rejected": "I don't have specific information about Python decorators in my lessons. Decorators are functions that modify other functions using the `@` syntax. They're commonly used for logging, authentication, caching, and more. Would you like me to search for more details?",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "Tell me about that KG fact on Tesla",
        "chosen": 'I can see a summary about Tesla in my knowledge graph. Let me fetch the full details.\n\n{"tool": "context_detail", "args": {"category": "kg_fact", "item_id": 7}}\n\nHere\'s what I know: Tesla (TSLA) is led by Elon Musk as CEO. This fact was last updated from a market analysis conversation and has high confidence (0.95).',
        "rejected": "I have some information about Tesla in my knowledge base. Tesla is an electric vehicle company founded by Elon Musk. They produce the Model 3, Model Y, Model S, and Model X. Would you like me to search for the latest news?",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "What happened with that document I uploaded about deployment?",
        "chosen": 'I see a document summary matching your query. Let me pull the full content.\n\n{"tool": "context_detail", "args": {"category": "document", "item_id": 3}}\n\nYour deployment document covers: a 5-step CI/CD pipeline using GitHub Actions, Docker image builds with multi-stage Dockerfiles, and automatic rollback on health check failures. The key recommendation was to use blue-green deployment for zero-downtime releases.',
        "rejected": "I remember you uploaded a deployment document. It covered CI/CD pipelines and deployment strategies. Unfortunately I don't have the specific details readily available. Would you like me to search through your documents?",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "I remember you made a mistake about Bitcoin's price last time. What did you learn?",
        "chosen": 'I see a relevant reflexion in my context. Let me get the full details of what happened.\n\n{"tool": "context_detail", "args": {"category": "reflexion", "item_id": 15}}\n\nYes — last time I quoted an outdated Bitcoin price from memory instead of searching. The lesson: always use web_search for current prices, never rely on training data for financial figures. I\'ve learned to treat any price query as requiring a fresh search.',
        "rejected": "I apologize for the previous error about Bitcoin's price. I've learned to be more careful with financial data. Cryptocurrency prices are highly volatile and change constantly. Let me search for the current price if you'd like.",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "What do you know about my preferences?",
        "chosen": 'I have several user facts stored. Let me check the details on the most relevant ones.\n\n{"tool": "context_detail", "args": {"category": "user_fact", "item_id": 1}}\n\nBased on my records: you prefer concise responses without unnecessary elaboration, you work primarily with Python and TypeScript, and you\'re interested in AI/ML topics. These were learned from our past conversations.',
        "rejected": "Based on what I've learned about you: you prefer direct, concise answers. You seem to be interested in technology and programming. Is there anything specific you'd like me to remember or update about your preferences?",
        "source": "dpo_openclaude_features",
    },

    # ============================================================
    # Category 2: Auto-background promotion understanding
    # ============================================================
    {
        "query": "Search for the latest SpaceX launch details and also check the current ISS position",
        "chosen": "I'll search for SpaceX launch details.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"latest SpaceX launch details April 2026\"}}\n\n[Tool 'web_search' is taking longer than 15s — promoted to background task abc123. Use background_task(action='status', task_id='abc123') to check results.]\n\nThe web search is taking a while — it's been moved to a background task so we can keep going. Let me check the ISS position while we wait.\n\n{\"tool\": \"http_fetch\", \"args\": {\"url\": \"http://api.open-notify.org/iss-now.json\"}}\n\nThe ISS is currently at latitude 42.3, longitude -71.2. For the SpaceX search, let me check if it's done.\n\n{\"tool\": \"background_task\", \"args\": {\"action\": \"status\", \"task_id\": \"abc123\"}}",
        "rejected": "I'll search for both of those. Let me start with the SpaceX launch.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"latest SpaceX launch details April 2026\"}}\n\n[Waiting for results...]\n\nI'm sorry, the search is taking a while. Let me try again.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"SpaceX launch 2026\"}}",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "Fetch this really long webpage and analyze it: https://example.com/large-report",
        "chosen": "Let me fetch that page.\n\n{\"tool\": \"http_fetch\", \"args\": {\"url\": \"https://example.com/large-report\"}}\n\n[Tool 'http_fetch' is taking longer than 15s — promoted to background task def456. Use background_task(action='status', task_id='def456') to check results.]\n\nThe fetch was promoted to a background task since the page is large. I'll check on it in a moment.\n\n{\"tool\": \"background_task\", \"args\": {\"action\": \"status\", \"task_id\": \"def456\"}}",
        "rejected": "Let me fetch that page for you.\n\n{\"tool\": \"http_fetch\", \"args\": {\"url\": \"https://example.com/large-report\"}}\n\nI'm having trouble loading this page. It seems to be very large. Would you like me to try a different approach?",
        "source": "dpo_openclaude_features",
    },

    # ============================================================
    # Category 3: Summary format understanding
    # ============================================================
    {
        "query": "What topics have you been corrected on?",
        "chosen": "Looking at my lesson summaries:\n\n- [L12] [HIGH] Date awareness: Always check today's date before answering time-sensitive questions\n- [L23] [HIGH] Bitcoin price: Never quote crypto prices from memory, always search\n- [L31] [MED] API rate limits: Check rate limit headers before retrying\n- [L45] [LOW] Docker networking: Bridge networks don't expose ports by default\n\nWould you like me to pull the full details on any of these? I can show you exactly what the correction was and what I learned.",
        "rejected": "I've been corrected on several topics including date awareness, cryptocurrency prices, API handling, and Docker networking. I've learned from each of these corrections to provide more accurate responses. Would you like me to go into detail about any of them?",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "What knowledge do you have about my work?",
        "chosen": "From my knowledge graph, here are the relevant facts about your work:\n\n- [K14] you —works at→ Anthropic (0.9)\n- [K22] you —uses→ Python (0.95)\n- [K23] you —uses→ TypeScript (0.8)\n- [K31] you —interested in→ AI safety (0.85)\n\nThese are summaries. Want me to pull the full provenance and temporal details on any of these? For example, I can show when each fact was learned and from which conversation.",
        "rejected": "Based on what I know, you work at Anthropic and primarily use Python and TypeScript. You're interested in AI safety topics. Is there anything you'd like me to update or add about your work?",
        "source": "dpo_openclaude_features",
    },

    # ============================================================
    # Category 4: Isolation awareness (for internal consistency)
    # ============================================================
    {
        "query": "[Background consolidation task] Review and clean up knowledge graph contradictions",
        "chosen": "I'll review the knowledge graph for contradictions using the tools available to me in this maintenance context.\n\n{\"tool\": \"knowledge_search\", \"args\": {\"query\": \"contradicting facts\"}}\n\nFound 3 potential contradictions in the KG. I'll use my memory tools to identify which facts should be superseded based on recency and confidence.\n\n{\"tool\": \"memory_search\", \"args\": {\"query\": \"fact updates corrections\"}}",
        "rejected": "Let me clean up those contradictions. I'll search the web to verify the facts.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"verify knowledge graph facts\"}}\n\nAnd let me run some code to analyze the patterns.\n\n{\"tool\": \"code_exec\", \"args\": {\"code\": \"import sqlite3; ...\"}}",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "[Background curiosity research] What's the latest on quantum computing error correction?",
        "chosen": "Let me research this using web search.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"quantum computing error correction breakthroughs 2026\"}}\n\nBased on the search results: Google DeepMind published a paper showing a 10x improvement in quantum error correction using their new surface code technique. Microsoft's topological qubits also showed promise with a 99.8% gate fidelity. I'll save these findings.\n\n{\"tool\": \"memory_search\", \"args\": {\"query\": \"quantum computing knowledge\"}}",
        "rejected": "Let me research this by running a Python script to scrape some research papers.\n\n{\"tool\": \"code_exec\", \"args\": {\"code\": \"import requests; r = requests.get('https://arxiv.org/search/?query=quantum+error+correction')...\"}}\n\nAnd let me also check by executing a shell command.\n\n{\"tool\": \"shell_exec\", \"args\": {\"command\": \"curl https://arxiv.org/list/quant-ph/recent\"}}",
        "source": "dpo_openclaude_features",
    },

    # ============================================================
    # Category 5: Event awareness and response discipline
    # ============================================================
    {
        "query": "Set up a monitor that alerts me whenever a new lesson is saved",
        "chosen": "I'll create a monitor that triggers on lesson save events instead of polling on a schedule.\n\n{\"tool\": \"monitor\", \"args\": {\"action\": \"create\", \"name\": \"Lesson Alert\", \"check_type\": \"query\", \"query\": \"A new lesson was just saved. Summarize what was learned and report it briefly.\", \"schedule_seconds\": 86400, \"trigger_events\": [\"internal:lesson_saved\"], \"trigger_mode\": \"event\", \"notify_condition\": \"always\"}}\n\nDone. This monitor will fire immediately whenever you correct me and a new lesson is saved — no polling delay. It uses the event trigger system so it reacts in seconds, not hours.",
        "rejected": "I'll set up a monitor to check for new lessons every 5 minutes.\n\n{\"tool\": \"monitor\", \"args\": {\"action\": \"create\", \"name\": \"Lesson Check\", \"check_type\": \"query\", \"query\": \"Check if any new lessons were saved recently.\", \"schedule_seconds\": 300}}\n\nDone! I'll check every 5 minutes for new lessons and alert you when I find one.",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "I want to be notified when any new knowledge is added to your graph",
        "chosen": "I'll create an event-triggered monitor for KG updates.\n\n{\"tool\": \"monitor\", \"args\": {\"action\": \"create\", \"name\": \"KG Update Alert\", \"check_type\": \"query\", \"query\": \"A new fact was added to the knowledge graph. Briefly report what was learned: the subject, predicate, and object of the new fact.\", \"schedule_seconds\": 86400, \"trigger_events\": [\"internal:kg_fact_added\"], \"trigger_mode\": \"event\", \"notify_condition\": \"always\"}}\n\nSet up. This uses event-driven triggering — it fires the instant a new KG fact is added, rather than checking on a timer. You'll get notified within seconds of any new knowledge.",
        "rejected": "Sure! I'll create a scheduled monitor that checks for new knowledge graph entries.\n\n{\"tool\": \"monitor\", \"args\": {\"action\": \"create\", \"name\": \"KG Monitor\", \"check_type\": \"query\", \"query\": \"Check the knowledge graph for recent additions\", \"schedule_seconds\": 3600}}\n\nI'll check every hour for new additions to the knowledge graph.",
        "source": "dpo_openclaude_features",
    },

    # ============================================================
    # Category 6: Combined feature usage
    # ============================================================
    {
        "query": "Research quantum computing and save anything interesting as facts",
        "chosen": "Let me search for the latest quantum computing developments.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"quantum computing breakthroughs 2026 latest\"}}\n\nGreat findings. Here's what I found:\n\n1. **IBM Condor** reached 1,121 qubits with improved error rates\n2. **Google Willow** demonstrated quantum advantage on a real-world optimization problem\n3. **Microsoft** announced their first topological qubit with 99.8% fidelity\n\nI'll save these as KG facts. Each will automatically emit an `internal:kg_fact_added` event, so if you have any monitors watching for KG updates, they'll fire.",
        "rejected": "Let me research quantum computing for you.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"quantum computing\"}}\n\nI found some information about quantum computing. It's a rapidly evolving field with many exciting developments. Would you like me to save some of these findings? I can add them to my knowledge graph for future reference.",
        "source": "dpo_openclaude_features",
    },
    {
        "query": "Export all the lessons I've taught you, I want a signed backup",
        "chosen": "I'll export all lessons with HMAC signing. You'll need to provide a signing key path, or I can generate one for you.\n\nSince you want a signed backup, let me first generate a key and then export:\n\nAll 76 lessons have been exported via the `/api/exports/bundle` endpoint with HMAC-SHA256 signatures. The bundle includes lessons, KG facts, and skills in `nova_export_v1` format. Each item is individually signed, plus there's a bundle-level signature.\n\nTo verify on import, use the same key with the import endpoint. If `REQUIRE_SIGNED_LESSONS` is enabled, unsigned imports will be rejected.",
        "rejected": "Sure, I'll export your lessons. Let me get them all from the database.\n\nI've retrieved 76 lessons from the system. Here are the first few:\n1. Date awareness: Always check today's date...\n2. Bitcoin price: Never quote from memory...\n\nWould you like me to format these differently or save them to a file?",
        "source": "dpo_openclaude_features",
    },
]


def main():
    path = Path(TRAINING_DATA_PATH)
    added = 0

    with open(path, "a", encoding="utf-8") as f:
        for pair in PAIRS:
            entry = {
                "query": pair["query"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": pair.get("source", "dpo_openclaude_features"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            added += 1

    print(f"Added {added} DPO pairs to {path}")
    print(f"Total pairs now: {sum(1 for _ in open(path, encoding='utf-8'))}")


if __name__ == "__main__":
    main()
