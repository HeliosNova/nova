# Contributing to Nova_

Thanks for your interest in contributing to Nova_!

## Development Setup

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Ollama 0.17.5+ with an NVIDIA GPU (RTX 3090 recommended)

### Local Development

```bash
# Clone the repo
git clone <repo-url> && cd nova_

# Copy environment template
cp .env.example .env
# Edit .env with your settings

# Start services
docker compose up

# Run tests (inside container)
docker exec nova-app sh -c "python -m pytest tests/ -v"
```

### Project Structure

- `app/core/` — Brain, LLM providers, learning, memory, knowledge graph
- `app/tools/` — Tool implementations (web search, browser, shell, MCP, etc.)
- `app/channels/` — Channel adapters (Discord, Telegram, WhatsApp, Signal)
- `app/monitors/` — Heartbeat system, proactive monitors
- `app/api/` — FastAPI route handlers
- `scripts/` — Fine-tuning, evaluation, MCP server runner
- `tests/` — Test suite
- `frontend/` — React frontend

## Code Style

- Async Python throughout — use `async`/`await`, never block the event loop
- No LangChain, no LangGraph — raw httpx to LLM providers
- Keep it lean — don't add features, config flags, or abstractions without discussion
- `learning.py` is the single source of truth for correction patterns — no duplicates
- Mock pattern: `patch("app.core.brain.llm")` for brain tests

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write/update tests for your changes
3. Ensure all tests pass: `python -m pytest tests/ -v`
4. Keep PRs focused — one concern per PR
5. Write a clear description of what and why

## What We're Looking For

- Bug fixes with test coverage
- Security improvements
- Performance optimizations
- New tool implementations
- Documentation improvements

## What We're NOT Looking For

- Framework migrations (no LangChain, no LangGraph)
- Major architectural changes without prior discussion
- Features that add config complexity without clear value
- Changes that break the single-pipeline architecture

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 license.
