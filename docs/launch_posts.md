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

## 5. Dev.to / Medium (longer form)

**Title:** I built the personal AI that OpenClaw should have been

**Subtitle:** Self-improving, sovereign, and actually secure. Here's how.

*(This would be a ~1500 word article covering the learning loop architecture, the security comparison, and the competitive landscape. Happy to draft the full article if you want.)*
