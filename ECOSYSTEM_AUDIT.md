# Helios Project Ecosystem Audit

**Date**: 2026-03-04
**Scope**: Full Helios ecosystem review with Nova as sole active project

---

## Ecosystem Overview

| Project | Size | Status |
|---------|------|--------|
| **nova_** | 91MB | Active — sole focus going forward |
| nova | 344MB | Old predecessor, kept as-is (historical) |
| forge | 262MB | Old iteration, kept as-is (historical) |
| helios_final | 260MB | Old iteration, kept as-is (historical) |
| Helios 90 | 3.0GB | Old version, kept as-is (historical) |
| Helios v1 proto | 3.5GB | Old version, kept as-is (historical) |
| Helios - original 2 | 8.4GB | Old version, kept as-is (historical) |
| archives | 23GB | Historical archive, kept as-is |
| Others | ~76KB | Nova2 docs, config files |

**Total ecosystem**: ~39GB. Only nova_ (91MB) is active.

---

## Nova Status: Production-Ready

### Architecture

```
User query -> brain.think()
  -> load context (history + facts + lessons + skills)
  -> classify intent (regex, no LLM)
  -> retrieve documents if needed (ChromaDB + FTS5 + RRF)
  -> build system prompt (8 prioritized blocks)
  -> generate response (Ollama /api/chat)
  -> tool loop if tool call detected (max 5 rounds)
  -> LLM self-critique (auto-regenerate if quality < 0.5)
  -> stream tokens via SSE
  -> post-response: corrections, fact extraction, skill creation
```

### Module Inventory (43 Python files)

**Core** (28 modules):
actions, auth, brain, chat, config, critique, custom_tools, database,
discord, documents, heartbeat, kg, learning (core + api), llm, main,
memory, monitors, planning, proactive, prompt, reflexion, retriever,
schema, skills, system, telegram

**Tools** (15 registered):
action_calendar, action_email, action_reminder, action_webhook, browser,
calculator, code_exec, file_ops, http_fetch, knowledge, memory_tool,
monitor_tool, screenshot, shell_exec, web_search

**Tests** (23 files):
conftest, e2e_live_test, eval_live, test_actions, test_auth,
test_behavioral, test_brain, test_core, test_critique, test_custom_tools,
test_desktop_tools, test_documents_api, test_foundation, test_heartbeat,
test_kg, test_learning, test_learning_api, test_planning, test_reflexion,
test_retriever, test_skills_hardening, test_system_api, test_tools

### Key Features

- **Brain pipeline**: intent -> tools -> critique -> stream -> reflexion
- **Token-by-token thinking** streaming with UI toggle
- **Persistent behavioral preferences** (fact/preference/instruction categories)
- **LLM self-critique** with auto-regeneration on quality < 0.5
- **DPO training data** collection + export API
- **Vision pipeline** (backend + frontend image upload with drag-drop/paste)
- **Hybrid retrieval** (ChromaDB + FTS5 + Reciprocal Rank Fusion)
- **Discord + Telegram** channels
- **Docker Compose** deployment (4 services: ollama, nova, frontend, searxng)

### No Critical Bugs

All 10/10 upgrades completed and deployed.

---

## Known Limits

1. **Model dependency** — Requires Ollama running with configured model (Qwen3.5:27b default)
2. **Vision** — Needs a multimodal model (e.g., llava) for image queries
3. **Extended thinking** — Requires model with `think: true` support (Qwen3+)
4. **Optional deps** — Calendar tool needs `ics`, browser tool needs `playwright`
5. **Single-user** — No multi-user auth by design
6. **Training export** — Manual fine-tuning workflow, no auto-train pipeline
7. **Telegram status** — Hardcodes `localhost` for health check

---

## Security Findings

### Bot Tokens Exposed in Old Project .env Files

The **same** Discord and Telegram bot tokens appear in 3 old project directories:

| File | Discord Token | Telegram Token |
|------|:---:|:---:|
| `archives/helios_final/.env` | Yes | Yes |
| `archives/Helios_Born/deploy/.env` | Yes | Yes |
| `nova/.env` (old project) | Yes | Yes |
| `archives/Helios_Born/deploy/.env.example` | Yes | Yes |

**Same tokens reused across all locations.**

### Recommendation

If these Discord/Telegram bots are still active:
- **Rotate both tokens immediately** via Discord Developer Portal and Telegram BotFather
- Update `nova_/.env` with the new tokens
- Old project `.env` files can be left as-is (tokens will be invalidated by rotation)

If the bots are no longer in use, no action needed — the tokens will remain inert.

---

## Future Directions (Not Planned)

Nova is feature-complete. Potential future work if desired:

- Multi-user authentication
- Auto fine-tuning pipeline (use exported DPO data to auto-retrain)
- Multi-image support in a single message
- Plugin/extension system for custom tools
- Mobile-responsive frontend refinement
