"""Cutoff timestamps must match SQLite's storage format, not ISO-8601.

SQLite's datetime('now') writes "YYYY-MM-DD HH:MM:SS" (SPACE separator).
Python's .isoformat() writes "YYYY-MM-DDTHH:MM:SS" (T). Those columns are
compared as STRINGS, and ' ' (0x20) sorts before 'T' (0x54), so every row dated
on the cutoff DAY compares as "older" regardless of its actual time.

Verified live against the running SQLite:
    '2026-08-23 10:00:00' < '2026-08-23T04:39:36'  ->  True   (wrong)
    '2026-08-23 10:00:00' < '2026-08-23 04:39:36'  ->  False  (right)

Consequences found in production code:
  memory.prune_old_conversations   `<`  deleted up to a full extra DAY
  daemon situational awareness     `>`  bug INVERTS: hid recent log entries
  daemon monitor-health count      `>`  under-counted recent errors
  api/daemon log endpoint          `>`  dropped the cutoff day from its window

Found by a systematic AST/regex hunt, not from any log line — this class is
silent by construction: the query succeeds and returns a plausible result.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta

import pytest

SQLITE_FMT = "%Y-%m-%d %H:%M:%S"


def test_the_underlying_string_comparison_is_really_broken():
    """Pin the root cause so the fix cannot be 'simplified' back later."""
    stored = "2026-08-23 10:00:00"          # later in the day
    iso_cutoff = "2026-08-23T04:39:36"      # earlier — should NOT match `<`
    sql_cutoff = "2026-08-23 04:39:36"
    assert stored < iso_cutoff, "the T-separator hazard no longer reproduces"
    assert not (stored < sql_cutoff), "space-separated cutoff must compare correctly"


def test_strftime_matches_sqlite_shape():
    got = datetime(2026, 8, 23, 4, 39, 36).strftime(SQLITE_FMT)
    assert got == "2026-08-23 04:39:36"
    assert "T" not in got


@pytest.mark.parametrize("module_path,func_name", [
    ("app.core.memory", "prune_old_conversations"),
])
def test_prune_uses_sqlite_format(module_path, func_name):
    import importlib
    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    # the prune cutoff must not be built with .isoformat()
    m = re.search(r"cutoff\s*=\s*\(datetime\.now\(\)[^\n]*", src)
    assert m, "prune cutoff assignment not found"
    assert ".isoformat()" not in m.group(0), (
        "conversation prune cutoff uses .isoformat(); its 'T' separator makes "
        "every row on the cutoff day compare as older and be deleted"
    )
    assert "strftime" in m.group(0)


def test_daemon_cutoffs_use_sqlite_format():
    from app.monitors import daemon
    src = inspect.getsource(daemon)
    for m in re.finditer(r"cutoff\w*\s*=\s*\(now\s*-\s*timedelta[^\n]*", src):
        assert ".isoformat()" not in m.group(0), (
            f"daemon cutoff still uses isoformat: {m.group(0).strip()!r} — with "
            f"`>` this EXCLUDES rows on the cutoff day"
        )


def test_api_daemon_cutoff_uses_sqlite_format():
    from app.api import daemon as api_daemon
    src = inspect.getsource(api_daemon)
    m = re.search(r"cutoff\s*=\s*\(datetime\.now\([^\n]*\n?[^\n]*", src)
    assert m and ".isoformat()" not in m.group(0), (
        "api/daemon log-window cutoff still uses isoformat"
    )
