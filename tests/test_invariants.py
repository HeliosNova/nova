"""Repository invariants (2026-09-02).

Deterministic checks for the drift that every audit kept re-finding by hand:
flags nobody reads, flags nobody documents, docs pointing at files that moved,
a monitor seeded with a check_type no handler dispatches, and a schema that
changed without anyone meaning it to. Each test names the fix in its message.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import app as _app_pkg

ROOT = Path(_app_pkg.__file__).resolve().parents[1]   # the tree the code actually imports from
APP = ROOT / "app"

# The deployed image ships app/, tests/, evals/ — not the repo's docs or
# .env.example. The doc/flag-documentation invariants are repo lint: they run
# on the host and in the ephemeral container (repo mounted at /app) and skip
# inside nova-app, where the files they lint deliberately do not exist.
needs_checkout = pytest.mark.skipif(
    not ((ROOT / ".env.example").exists() and (ROOT / "CLAUDE.md").exists()),
    reason="repo-lint test: run against a checkout (host or repo-mounted container)")
CONFIG = APP / "config.py"
ENV_EXAMPLE = ROOT / ".env.example"
SNAPSHOT = ROOT / "tests" / "schema_snapshot.json"

_FLAG_FIELD_RE = re.compile(r"^\s+(ENABLE_[A-Z_]+):\s*bool", re.M)
_READER_RES = (
    re.compile(r"\b[A-Za-z_]*(?:config|cfg)\.(ENABLE_[A-Z_]+)\b"),
    re.compile(r"getattr\(\s*[^,()]+,\s*[\"'](ENABLE_[A-Z_]+)[\"']"),
    re.compile(r"environ(?:\.get)?\(\s*[\"'](ENABLE_[A-Z_]+)[\"']"),
    re.compile(r"_env\(\s*[\"'](ENABLE_[A-Z_]+)[\"']"),
)
_ENV_LINE_RE = re.compile(r"^(ENABLE_[A-Z_]+)=", re.M)


def _py_files():
    for base in (APP, ROOT / "scripts"):
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _config_flags() -> set[str]:
    return set(_FLAG_FIELD_RE.findall(CONFIG.read_text(encoding="utf-8")))


def _flag_readers() -> set[str]:
    found: set[str] = set()
    for path in _py_files():
        if path == CONFIG:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for rx in _READER_RES:
            found.update(rx.findall(text))
    return found


# ------------------------------------------------------------------ flags

def test_every_config_flag_has_a_runtime_reader():
    """A flag without a reader is a lie in the UI: flipping it does nothing.
    Delete the field (and its .env.example line) or wire it."""
    inert = sorted(_config_flags() - _flag_readers())
    assert not inert, f"inert flags (no reader outside app/config.py): {inert}"


@needs_checkout
def test_every_config_flag_is_documented_in_env_example():
    documented = set(_ENV_LINE_RE.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))
    missing = sorted(_config_flags() - documented)
    assert not missing, f"add these to .env.example (feature-flags section): {missing}"


@needs_checkout
def test_every_documented_flag_exists_in_config():
    documented = set(_ENV_LINE_RE.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))
    ghosts = sorted(documented - _config_flags())
    assert not ghosts, f".env.example documents flags config.py no longer has: {ghosts}"


@needs_checkout
def test_default_model_agrees_between_config_and_env_example():
    cfg = re.search(r'_env\("LLM_MODEL",\s*"([^"]+)"\)', CONFIG.read_text(encoding="utf-8"))
    env = re.search(r"^LLM_MODEL=(\S+)", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M)
    assert cfg and env
    assert cfg.group(1) == env.group(1), "LLM_MODEL default drifted between config.py and .env.example"


# ------------------------------------------------------------------- docs

_DOC_PATH_RE = re.compile(
    r"`((?:app|scripts|tests|evals|docs|frontend|archive|socket-proxy|mcp_configs|searxng)/"
    r"[A-Za-z0-9_./-]+\.(?:py|yaml|yml|json|jsonl|md|txt|sh|conf|example|toml|ini|html|ts|tsx|js|css))`")
_MOVED_WORDS = re.compile(r"\b(archiv\w*|removed|retired|moved|deleted|obsolete|renamed)\b", re.I)


def _dead_doc_paths(doc: Path) -> list[str]:
    lines = doc.read_text(encoding="utf-8").splitlines()
    dead = []
    for i, line in enumerate(lines):
        for rel in _DOC_PATH_RE.findall(line):
            if (ROOT / rel).exists():
                continue
            context = " ".join(lines[max(0, i - 2): i + 3])
            if _MOVED_WORDS.search(context):
                continue   # documented as gone — that is the point of the sentence
            dead.append(f"{doc.name}:{i + 1} {rel}")
    return dead


@needs_checkout
@pytest.mark.parametrize("doc", ["CLAUDE.md", "README.md"])
def test_doc_paths_exist_or_are_marked_as_moved(doc):
    dead = _dead_doc_paths(ROOT / doc)
    assert not dead, "doc references files that do not exist (fix the path or say it moved):\n" + "\n".join(dead)


# --------------------------------------------------------------- monitors

def _seed_catalog(db):
    from app.monitors.monitor_store import MonitorStore
    store = MonitorStore(db)
    store.seed_defaults()
    return store


def test_every_seeded_check_type_has_a_dispatch_handler(db):
    """The dispatch table is the only route from a monitor row to code: a
    seed with an unregistered check_type runs as '[Unknown check_type]'
    forever (the 2026-08-17 duplicate-key bug hid Dream Consolidation)."""
    from app.monitors.heartbeat_loop import HeartbeatLoop
    store = _seed_catalog(db)
    missing = sorted({m.check_type for m in store.list_all()} - set(HeartbeatLoop._CHECK_DISPATCH))
    assert not missing, f"seeded check_types without a _CHECK_DISPATCH handler: {missing}"


def test_fast_lane_types_are_dispatchable():
    from app.monitors.heartbeat_loop import HeartbeatLoop
    src = (APP / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    m = re.search(r"_FAST_TYPES = \{([^}]*)\}", src)
    assert m
    fast = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert fast <= set(HeartbeatLoop._CHECK_DISPATCH)


def test_core_enabled_names_are_all_seeded(db):
    from app.monitors.monitor_store import MonitorStore
    store = _seed_catalog(db)
    names = {m.name for m in store.list_all()}
    ghosts = sorted(MonitorStore._CORE_ENABLED - names)
    assert not ghosts, f"_CORE_ENABLED names with no seed: {ghosts}"


def test_system_category_check_types_are_dispatchable():
    from app.monitors.heartbeat_loop import HeartbeatLoop
    from app.monitors.monitor_store import _SYSTEM_CATEGORY_CHECK_TYPES
    stray = sorted(_SYSTEM_CATEGORY_CHECK_TYPES - set(HeartbeatLoop._CHECK_DISPATCH))
    assert not stray, f"_SYSTEM_CATEGORY_CHECK_TYPES lists undispatchable types: {stray}"


# ----------------------------------------------------------------- schema

def _schema_of(db) -> dict[str, list[str]]:
    tables = [r["name"] for r in db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {t: [r["name"] for r in db.fetchall(f"PRAGMA table_info({t})")] for t in tables}


def test_fresh_schema_matches_the_committed_snapshot(db):
    """init_schema() on an empty file must produce exactly tests/schema_snapshot.json.
    A deliberate schema change regenerates it: UPDATE_SCHEMA_SNAPSHOT=1 pytest
    tests/test_invariants.py -k snapshot."""
    live = _schema_of(db)
    if os.environ.get("UPDATE_SCHEMA_SNAPSHOT") == "1":
        SNAPSHOT.write_text(json.dumps(live, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    assert SNAPSHOT.exists(), "run once with UPDATE_SCHEMA_SNAPSHOT=1 to create the snapshot"
    want = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    added = sorted(set(live) - set(want))
    dropped = sorted(set(want) - set(live))
    changed = {t: (want[t], live[t]) for t in set(live) & set(want) if want[t] != live[t]}
    assert not (added or dropped or changed), (
        f"schema drift — tables added {added}, dropped {dropped}, columns changed {changed}. "
        "If intended, regenerate with UPDATE_SCHEMA_SNAPSHOT=1.")


def test_init_schema_is_idempotent(db):
    before = _schema_of(db)
    db.init_schema()
    assert _schema_of(db) == before
