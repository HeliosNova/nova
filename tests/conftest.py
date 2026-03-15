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
