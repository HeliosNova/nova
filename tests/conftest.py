"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Windows: the C runtime caps stdio handles at 512. A full-suite run (~2,400
# tests opening SQLite, ChromaDB, and socket handles) exhausts the cap late in
# the run and unrelated tests start dying with OSError errno 24 (Too many
# open files) inside ssl/certifi. Raise the cap to the CRT maximum once.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.cdll.msvcrt._setmaxstdio(8192)
    except Exception:
        pass


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
    # Isolate from the container's ambient production ENABLE_* overrides so
    # default-config tests are deterministic regardless of the runtime env (the
    # container bakes these true; tests assert the code defaults).
    monkeypatch.setenv("ENABLE_VOICE", "false")
    monkeypatch.setenv("ENABLE_TWO_PHASE_DREAM", "false")
    monkeypatch.setenv("ENABLE_DESKTOP_AUTOMATION", "false")
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
    # Unit tests exercise the IN-PROCESS code_exec path (mocked subprocess);
    # in the deployed container /exec_queue exists and would route execution
    # to the nova-exec sidecar, bypassing every mock (audit 2026-07-08).
    monkeypatch.setenv("EXEC_QUEUE_DIR", "/nonexistent-exec-queue")
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

    # Force LLM provider retries to 0 in tests. The dead-port redirect above is
    # meant to give "instant connection-refused", but retry_on_transient's default
    # 3x exponential backoff turns every UNMOCKED llm call into ~14s of sleeps —
    # silently slowing the suite and able to stall a multi-call path (the cause of
    # the 2026-06-27 full-suite hang in TestCritiqueBrainIntegration). A properly
    # mocked test never reaches this; an unmocked one now fails fast and loud.
    import app.core.providers._retry as _retry_mod
    import app.core.providers.ollama as _ollama_mod
    _orig_retry = _retry_mod.retry_on_transient

    async def _fast_retry(client, method, url, **kwargs):
        kwargs["max_retries"] = 0
        return await _orig_retry(client, method, url, **kwargs)

    monkeypatch.setattr(_retry_mod, "retry_on_transient", _fast_retry)
    monkeypatch.setattr(_ollama_mod, "retry_on_transient", _fast_retry)

    # Fail-fast on the headless browser too. The container ships Playwright, so
    # BrowserTool tests that assume "no Playwright" actually launch a real Chromium
    # and hang (no display / CDP). execute() wraps _ensure_browser in try/except and
    # every BrowserTool test asserts failure, so raising here gives them a fast,
    # correct error. A test that mocks the browser itself overrides this.
    import app.tools.browser as _browser_mod

    async def _no_browser(*_a, **_k):
        raise RuntimeError("browser launch disabled in tests")

    monkeypatch.setattr(_browser_mod.BrowserTool, "_ensure_browser", _no_browser)

    # Recreate config from current env — the _ConfigProxy ensures all
    # modules that imported `config` automatically see the new values.
    from app.config import reset_config
    reset_config()

    # Close + clear DB singletons. close_all() (not a bare _instances.clear())
    # is required since SafeDB went to one connection PER THREAD: clearing the
    # registry without closing leaks every per-thread WAL connection from the
    # prior test. Across the full suite those accumulate into thousands of open
    # -wal/-shm handles, and on the container's overlay/9p filesystem the next
    # WAL writer (init_schema) eventually hangs on shared-memory I/O. Closing
    # each instance releases them. (Host fs tolerated the leak; the container
    # did not — full-suite hang surfaced 2026-06-12.)
    import app.database
    app.database.close_all()

    yield


@pytest.fixture
def db(tmp_path):
    """Get a fresh test database."""
    from app.database import SafeDB
    db = SafeDB(str(tmp_path / "test.db"))
    db.init_schema()
    yield db
    db.close()
