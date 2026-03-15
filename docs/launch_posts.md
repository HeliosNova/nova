# Nova Launch Posts — Ready to Copy/Paste

## 1. Hacker News (Show HN)

**Title:** Show HN: Nova – Self-hosted personal AI that learns from corrections and fine-tunes itself

**Body:**

Hey HN, I built Nova — a personal AI assistant that runs entirely on your hardware and actually gets smarter over time.

The core idea: every time you correct Nova, it extracts a lesson, generates a DPO training pair, and when enough pairs accumulate, it automatically fine-tunes itself with A/B evaluation before deploying the new model.

No other open-source AI assistant has this learning loop.

**What it does:**
- Correction detection (2-stage: regex + LLM) → lesson extraction → DPO training data → automated fine-tuning with A/B eval
- Temporal knowledge graph (20 predicates, fact supersession, provenance tracking)
- Hybrid retrieval (ChromaDB vectors + SQLite FTS5 + Reciprocal Rank Fusion)
- 21 tools, 4 messaging channels (Discord/Telegram/WhatsApp/Signal), 14 proactive monitors
- MCP client AND server (expose Nova's intelligence to Claude Code, Cursor, etc.)

**What it's not:**
- Not a ChatGPT wrapper — runs Qwen3.5:27b locally via Ollama, zero cloud dependency
- Not a LangChain/LangGraph project — single async pipeline, ~74 files of plain Python
- Not a coding agent — it's a personal assistant (but you can connect it to coding agents via MCP)

**Security:** 4-tier access control, prompt injection detection (4 categories), SSRF protection, HMAC skill signing, Docker hardening (read-only root, no-new-privileges, all caps dropped). Built with OWASP Agentic Security in mind — unlike certain 200K-star projects that got CVE'd within weeks of launch.

**Stack:** Python, FastAPI, httpx, Ollama, ChromaDB, SQLite, React. 1,443 tests.

No GPU? Set `LLM_PROVIDER=openai` and use cloud inference while keeping all data local.

https://github.com/HeliosNova/nova

---

## 2. Reddit r/LocalLLaMA

**Title:** Nova — self-hosted personal AI that learns from your corrections and fine-tunes itself (DPO + A/B eval, runs on RTX 3090)

**Body:**

I've been building Nova for a while and just open-sourced it. It's a personal AI assistant that runs Qwen3.5:27b on your own hardware (RTX 3090) and has a full self-improvement loop:

1. You ask a question, Nova gets it wrong
2. You correct it ("Actually, it's X")
3. Nova detects the correction (regex pre-filter + LLM confirmation)
4. Extracts a structured lesson (topic, wrong answer, correct answer)
5. Generates a DPO training pair {query, chosen, rejected}
6. On future similar queries, retrieves the lesson and gets it right
7. When enough training pairs accumulate, runs automated DPO fine-tuning with A/B evaluation

No other open-source project has this full pipeline.

**Beyond the learning loop:**
- Temporal knowledge graph (facts track when they were valid, supersession chains)
- Hybrid retrieval (ChromaDB + FTS5 + RRF fusion — not just vector search)
- 14 proactive monitors doing scheduled research, self-reflection, skill validation
- Curiosity engine — detects knowledge gaps and queues background research
- 4 messaging channels (Discord, Telegram, WhatsApp, Signal)
- MCP client + server

**Hardware:** RTX 3090 for local Qwen3.5:27b. Or set LLM_PROVIDER=openai/anthropic/google for cloud inference (data stays local).

**Not a LangChain project.** Single async pipeline, ~74 files of Python. No frameworks.

1,443 tests. AGPL-3.0.

https://github.com/HeliosNova/nova

---

## 3. Reddit r/selfhosted

**Title:** Nova — self-hosted personal AI assistant with learning, knowledge graph, and 4 messaging channels (Docker Compose, runs offline)

**Body:**

Just open-sourced Nova, a personal AI assistant designed for self-hosting.

**Why I built it:** Every "self-hosted AI" I tried was either a ChatGPT UI wrapper (Open WebUI), needed cloud APIs to function (OpenClaw), or had no memory between conversations. I wanted an AI that:
- Runs 100% offline on my hardware
- Remembers what I tell it across conversations
- Actually learns from its mistakes
- Is proactive (monitors things, researches topics, alerts me)
- Is secure by default

**What's in the Docker Compose:**
- Ollama (local LLM — Qwen3.5:27b)
- Nova API (FastAPI backend)
- React frontend
- SearXNG (privacy-respecting search)

`docker compose up -d` and you're running.

**Security:** Read-only root filesystem, no-new-privileges, all capabilities dropped, non-root user, 4-tier access control, prompt injection detection, SSRF protection, rate limiting, auth lockout. After seeing what happened with OpenClaw (CVE-2026-25253, ClawHavoc supply chain attack), I built security in from the start.

**Channels:** Talk to it via Discord, Telegram, WhatsApp, or Signal — all with phone-number allowlisting.

**No GPU?** Set `LLM_PROVIDER=openai` in .env. Cloud handles inference, all your data stays on your machine.

https://github.com/HeliosNova/nova

---

## 4. Reddit r/opensource

**Title:** Nova — AGPL-3.0 personal AI that learns from corrections and fine-tunes itself. 1,443 tests, zero cloud dependency.

**Body:**

Open-sourced Nova today. It's a personal AI assistant that runs on your hardware and gets permanently smarter through a self-improvement pipeline.

The differentiator: correct Nova once, it remembers forever. Correct it enough, it fine-tunes itself into a better model (automated DPO + A/B evaluation).

No other open-source project combines:
- Self-improving (corrections → lessons → DPO → fine-tuning)
- Sovereign (zero cloud dependency, bundled Ollama)
- Knowledge graph (temporal, with fact supersession)
- Hybrid retrieval (vectors + BM25 + reciprocal rank fusion)
- Proactive (14 scheduled monitors doing research, maintenance, self-testing)
- Secure (4-tier access, injection detection, HMAC signing, Docker hardening)

Stack: Python, FastAPI, SQLite, ChromaDB, Ollama, React
Tests: 1,443 across 57 files
License: AGPL-3.0

https://github.com/HeliosNova/nova

---

## 5. Dev.to / Medium

**Title:** I built the personal AI that OpenClaw should have been

**Tags:** ai, opensource, selfhosted, python

---

OpenClaw hit 216,000 GitHub stars in six weeks. It proved that millions of people want a personal AI assistant they can run themselves. Then came CVE-2026-25253. Then the ClawHavoc supply chain attack — 341 malicious skills, 9,000 compromised installations. Cisco and Palo Alto flagged it for a "lethal trifecta" of security risks: unrestricted tool access, no prompt injection detection, and plaintext credential storage.

I'd been building my own self-hosted AI assistant for months. When OpenClaw blew up — and then blew up differently — I decided to open-source it.

**It's called [Nova](https://github.com/HeliosNova/nova).**

## What makes Nova different

Every AI assistant answers questions. Nova is the only one that *learns from getting them wrong*.

```
You: "What's the capital of Australia?"
Nova: "Sydney"
You: "That's wrong, it's Canberra"

Nova detects the correction, extracts a lesson, generates a
DPO training pair, and updates its knowledge graph.

--- 3 months later, different conversation ---

You: "What's the capital of Australia?"
Nova: "Canberra"
```

That's not retrieval-augmented generation. That's not prompt engineering. The model itself got smarter.

## The learning loop — how it actually works

Nova has a 7-stage self-improvement pipeline. No other open-source project has anything close to this.

### Stage 1: Correction Detection

When you say "actually, it's X" or "that's wrong," Nova's 2-stage detector fires:

1. **Regex pre-filter** — 12 compiled patterns catch correction language ("actually," "that's wrong," "it should be," "remember that"). Fast, zero LLM cost.
2. **LLM confirmation** — if the regex matches, Nova sends the exchange to the LLM with a structured extraction prompt. It pulls out: what was wrong, what's correct, and a one-sentence lesson.

Why two stages? Because "actually, I was thinking about pasta tonight" isn't a correction. The regex catches candidates cheaply; the LLM filters false positives.

### Stage 2: Lesson Storage

Every confirmed correction becomes a lesson with four fields: `topic`, `wrong_answer`, `correct_answer`, and `lesson_text`. Lessons are stored in SQLite and indexed in ChromaDB for semantic search.

On future queries, Nova retrieves relevant lessons using hybrid search (vector similarity + BM25 keyword matching + Reciprocal Rank Fusion) and injects them into the system prompt: *"You got this wrong before. The capital of Australia is Canberra, not Sydney."*

### Stage 3: DPO Training Data

Every correction also generates a DPO (Direct Preference Optimization) training pair:

```json
{
  "query": "What's the capital of Australia?",
  "chosen": "The capital of Australia is Canberra.",
  "rejected": "The capital of Australia is Sydney.",
  "timestamp": "2026-03-15T14:23:01"
}
```

These accumulate in a JSONL file. When enough pairs exist, Nova can fine-tune its own base model.

### Stage 4: Automated Fine-Tuning

Nova includes an 8-step automated pipeline (`scripts/finetune_auto.py`):

1. Check readiness (minimum 50 new DPO pairs)
2. Load training data
3. Stop Ollama (free GPU VRAM)
4. Run DPO training via Unsloth
5. Export to GGUF
6. Restart Ollama
7. **A/B evaluation** — run holdout queries through both base and fine-tuned models, LLM-as-judge with randomized ordering to prevent position bias
8. Deploy only if the fine-tuned model wins >50% with positive average preference

The model literally gets smarter. Not through bigger context windows or better prompts — through actual weight updates from your corrections.

### Stage 5: Reflexion

Not every failure is an explicit correction. Sometimes Nova gives a bad answer and you just move on. Reflexion catches these *silent failures*:

- Empty or very short responses to complex queries
- Tool loop exhaustion (used all 5 rounds without a clean answer)
- Error phrases in the response ("I couldn't," "failed to")
- Hallucination indicators

Failed responses are stored as reflexions. On future similar queries, Nova retrieves them as warnings: *"You failed on a similar query before. Here's what went wrong."*

### Stage 6: Curiosity Engine

When Nova hedges ("I'm not sure"), admits ignorance, or a tool search returns nothing useful, the curiosity engine detects the knowledge gap and queues it for background research. A scheduled monitor (runs every hour) picks up the queue and researches the topics autonomously — results become knowledge graph triples.

### Stage 7: Success Patterns

High-quality responses (score >= 0.8) are stored as positive reinforcement. On similar future queries, Nova retrieves what worked: *"This approach worked well last time."*

## What's under the hood

Nova replaces a 9-node LangGraph pipeline with a single async generator function: `brain.think()`. About 1,400 lines of Python that orchestrate 5 stages:

1. **Gather context** — load user facts, lessons, knowledge graph, reflexions, retrieved documents, skills
2. **Build messages** — assemble system prompt from 8 prioritized blocks with truncation budget
3. **Generate + tool loop** — up to 5 rounds of LLM generation + tool execution (21 built-in tools)
4. **Refine** — multi-round self-critique, plan coverage check, reflexion quality assessment
5. **Post-process** — correction detection, fact extraction, KG updates, curiosity gap detection

No LangChain. No LangGraph. No agent frameworks. Just `async for event in think(query)`.

## Security — built in, not bolted on

After watching OpenClaw's security meltdown, I built Nova with the [OWASP Agentic Security Top 10](https://genai.owasp.org/) in mind:

| Risk | OpenClaw | Nova |
|------|----------|------|
| Unrestricted tool access | All tools always available | 4-tier access control (sandboxed/standard/full/none) |
| Prompt injection | No detection | 4-category heuristic detection on all external content |
| Credential exposure | Plaintext storage flagged | No hardcoded secrets, `.env` gitignored, HMAC skill signing |
| Training data poisoning | N/A (no learning) | Channel gating + confidence threshold for DPO pairs |
| Container security | Basic Docker | Read-only root, no-new-privileges, all capabilities dropped |
| Auth | Partial | Bearer token + per-IP brute-force lockout (10 failures = 5min ban) |

The prompt injection detector runs on every piece of external content — web search results, fetched pages, browser output, MCP tool results, imported skills. It checks 4 categories (role override, instruction injection, delimiter abuse, encoding tricks) with Unicode normalization and homoglyph detection. Suspicious content gets flagged, not stripped — the LLM sees it but is warned.

## The stack

- **Backend:** Python 3.11+, FastAPI, httpx, SQLite (WAL mode), ChromaDB
- **LLM:** Ollama (default: Qwen3.5:27b) or OpenAI/Anthropic/Google
- **Frontend:** React + TypeScript + Vite
- **Search:** SearXNG (privacy-respecting, self-hosted)
- **Deployment:** Docker Compose (4 services)
- **Tests:** 1,443 across 57 files (including security offensive, stress, and behavioral tests)

No GPU? Use `docker-compose.cloud.yml` — cloud handles inference, all data stays on your machine.

## What else it does

- **Temporal knowledge graph** — facts track when they were valid, with supersession chains and provenance. Query what was true at any point in time.
- **14 proactive monitors** — scheduled domain research, self-reflection, lesson quizzes, skill validation, system maintenance. Nova works even when you're not talking to it.
- **4 messaging channels** — Discord, Telegram, WhatsApp, Signal. All with phone-number allowlisting.
- **MCP dual-mode** — consumes external tools (client) AND exposes its intelligence to Claude Code, Cursor, etc. (server). No other personal AI does both.
- **21 built-in tools** — web search, calculator, code execution, browser, email, calendar, webhooks, file ops, shell, and more.
- **Voice** — local Whisper speech-to-text.
- **Desktop automation** — PyAutoGUI-based GUI control.

## Try it

```bash
git clone https://github.com/HeliosNova/nova.git
cd nova && cp .env.example .env
docker compose up -d
```

Or one-liner:
```bash
curl -fsSL https://raw.githubusercontent.com/HeliosNova/nova/main/install.sh | bash
```

AGPL-3.0. Issues and PRs welcome.

**https://github.com/HeliosNova/nova**
