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
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8888")
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

    # Tuning parameters — deterministic values for tests
    monkeypatch.setenv("RESPONSE_TOKEN_BUDGET", "600")
    monkeypatch.setenv("RETRIEVAL_RELEVANCE_THRESHOLD", "0.15")
    monkeypatch.setenv("TEMPERATURE_DEFAULT", "0.7")
    monkeypatch.setenv("TEMPERATURE_INTERNAL", "0.3")
    monkeypatch.setenv("TEMPERATURE_REFLEXION", "0.4")
    monkeypatch.setenv("MIN_RRF_SCORE", "0.015")
    monkeypatch.setenv("DEDUP_JACCARD_THRESHOLD", "0.85")
    monkeypatch.setenv("REFLEXION_DECAY_DAYS", "90")
    monkeypatch.setenv("REFLEXION_DECAY_AMOUNT", "0.05")
    monkeypatch.setenv("REFLEXION_DISTANCE_THRESHOLD", "0.7")
    monkeypatch.setenv("SKILL_EMA_ALPHA", "0.15")
    monkeypatch.setenv("INJECTION_SUSPICIOUS_THRESHOLD", "0.3")
    monkeypatch.setenv("FACT_INJECTION_SKIP_THRESHOLD", "0.3")
    monkeypatch.setenv("FACT_CONFIDENCE_EXTRACTED", "0.65")
    monkeypatch.setenv("FACT_CONFIDENCE_USER", "0.9")
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
