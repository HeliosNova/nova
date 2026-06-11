"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _test_env(tmp_path, monkeypatch):
    """Set test environment variables so we never hit real services."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chromadb"))
    monkeypatch.setenv("TRAINING_DATA_PATH", str(tmp_path / "training.jsonl"))
    # Critical: redirect config overrides to a tmp path so production
    # /data/config_overrides.json doesn't override test env vars (e.g. enabling
    # shell exec when tests assume it's off).
    from pathlib import Path as _Path
    import app.config as _config_mod
    monkeypatch.setenv("CONFIG_OVERRIDES_PATH", str(tmp_path / "no_overrides.json"))
    # Also patch the module-level constant so _load_overrides picks it up via
    # the env-var fallback path.
    monkeypatch.setattr(_config_mod, "_OVERRIDES_PATH", _Path(tmp_path / "no_overrides.json"))
    # Guaranteed-dead ports (TCP discard) — NOT the real service ports. If the
    # production Docker stack is running on this machine, tests pointed at
    # localhost:11434/8888 would silently reach a REAL Ollama/SearXNG and
    # change behavior (4 tests failed exactly this way on 2026-06-10). Tests
    # must get instant connection-refused unless they explicitly mock.
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:27b")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe")
    monkeypatch.setenv("ENABLE_EXTENDED_THINKING", "false")
    monkeypatch.setenv("ENABLE_CRITIQUE", "false")
    monkeypatch.setenv("ENABLE_PLANNING", "false")
    monkeypatch.setenv("ENABLE_MODEL_ROUTING", "false")
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setenv("NOVA_API_KEY", "")
    monkeypatch.setenv("SYSTEM_ACCESS_LEVEL", "sandboxed")
    monkeypatch.setenv("ENABLE_SHELL_EXEC", "false")
    monkeypatch.setenv("ENABLE_SEMANTIC_SKILL_MATCHING", "false")  # opt-in per test
    monkeypatch.setenv("ENABLE_AUTONOMOUS_TOOL_CREATION", "false")  # opt-in per test
    monkeypatch.setenv("ENABLE_MULTI_AGENT", "false")  # opt-in per test
    monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "4")
    monkeypatch.setenv("MAX_AGENT_COUNT", "5")
    monkeypatch.setenv("AGENT_TASK_TIMEOUT", "90")
    monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "false")
    monkeypatch.setenv("ENABLE_EVAL_HARNESS", "true")
    monkeypatch.setenv("EVAL_SUITE_PATH", "evals/suite.yaml")
    monkeypatch.setenv("EVAL_REPORT_PATH", str(tmp_path / "eval_reports"))
    monkeypatch.setenv("EVAL_REGRESSION_TOLERANCE", "0.10")

    # Tuning parameters — deterministic values for tests
    monkeypatch.setenv("MAX_SYSTEM_TOKENS", "6000")
    monkeypatch.setenv("RESPONSE_TOKEN_BUDGET", "600")
    monkeypatch.setenv("RETRIEVAL_RELEVANCE_THRESHOLD", "0.15")
    monkeypatch.setenv("TEMPERATURE_DEFAULT", "0.7")
    monkeypatch.setenv("MIN_RRF_SCORE", "0.015")
    monkeypatch.setenv("DEDUP_JACCARD_THRESHOLD", "0.85")
    monkeypatch.setenv("REFLEXION_DECAY_DAYS", "90")
    monkeypatch.setenv("REFLEXION_DECAY_AMOUNT", "0.05")
    monkeypatch.setenv("REFLEXION_DISTANCE_THRESHOLD", "0.7")
    monkeypatch.setenv("SKILL_EMA_ALPHA", "0.15")
    monkeypatch.setenv("INJECTION_SUSPICIOUS_THRESHOLD", "0.3")
    monkeypatch.setenv("REFLEXION_FAILURE_THRESHOLD", "0.6")
    monkeypatch.setenv("REFLEXION_SUCCESS_THRESHOLD", "0.8")
    monkeypatch.setenv("KG_GRAPH_MAX_FRONTIER", "1000")
    monkeypatch.setenv("AUTH_MAX_TRACKED_IPS", "10000")

    # Recreate config from current env — the _ConfigProxy ensures all
    # modules that imported `config` automatically see the new values.
    from app.config import reset_config
    reset_config()

    # Clear DB singletons
    import app.database
    app.database._instances.clear()

    yield


@pytest.fixture
def db(tmp_path):
    """Get a fresh test database."""
    from app.database import SafeDB
    db = SafeDB(str(tmp_path / "test.db"))
    db.init_schema()
    yield db
    db.close()
