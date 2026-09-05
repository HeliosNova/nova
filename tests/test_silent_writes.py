"""An exception swallowed around a write is a lie about success.

The caller carries on and reports the work as done. That is how the enrichment
gate logged "0 entail-dropped" after timing out, how 8 of 15 SEC filings shipped
unannotated with no record of why, and how fact banking could report
"3 candidates, 0 stored" while attributing the loss to nothing at all.

27 such sites existed on 2026-09-04. Twelve were given a log line — still
swallowed, because a digest must not die because one fact would not bank, but no
longer silent. The other fifteen are listed below with a reason each, and this
test fails if a SIXTEENTH appears anywhere new.

That is the point: the list is not a record of debt, it is a gate. Adding a
silent write now costs you a line here and a reason.
"""
from __future__ import annotations

import pathlib
from collections import Counter

import pytest

# scripts/ is NOT baked into the image (only scripts/__init__.py), so the
# in-container test path documented in CLAUDE.md cannot import these checks.
# Skip loudly rather than silently: a check that quietly does not run is worse
# than one that is absent, because it looks covered.
pytest.importorskip(
    "scripts.sanity_scan",
    reason="scripts/ is not in the image — run the suite with the tree mounted: "
           "docker run --rm -v \"<repo>:/app\" -w /app nova-app:latest python -m pytest")

from scripts.sanity_scan import silent_write_sites

APP = pathlib.Path(__file__).resolve().parent.parent / "app"

# (file, enclosing function) -> how many, and why it is allowed to be silent.
ACCEPTED: dict[tuple[str, str], int] = {
    # Idempotent DDL: the exception IS the "already exists" case, and these all
    # catch sqlite3.OperationalError specifically rather than everything.
    ("app/core/agent_workspace.py", "_ensure_failed_approaches_column"): 1,
    ("app/core/kg.py", "__init__"): 3,
    ("app/core/output_eval.py", "_ensure_table"): 1,
    ("app/core/reflexion.py", "__init__"): 1,
    ("app/database.py", "_run_migrations"): 1,

    # Best-effort cleanup of rows nothing reads: a leftover temp row is
    # cosmetic, and failing the caller over one would be worse.
    ("app/core/auto_tools.py", "_smoke_test_tool"): 3,
    ("app/monitors/dedup.py", "is_duplicate"): 1,

    # Not a swallow at all — there is a real fallback on the next line.
    ("app/core/dream.py", "_prune_reflexions"): 1,

    # Fetch ladders: each rung is ALLOWED to fail, that is what the next rung is
    # for, and the aggregate is already reported ("N item(s) had no usable body
    # -> bare link", with the hosts and byte counts named).
    ("app/monitors/deep_research.py", "_fetch_body"): 2,
    ("app/monitors/deep_research.py", "_overview_angles"): 1,
}


def _found() -> Counter:
    """Keyed repo-relative, so the list reads the same wherever the tree is
    mounted (the container puts it at /app)."""
    repo = APP.parent
    out: Counter = Counter()
    for f, fn in silent_write_sites(APP):
        rel = pathlib.Path(f).resolve().relative_to(repo).as_posix()
        out[(rel, fn)] += 1
    return out


def test_no_new_silent_write_paths():
    found = _found()
    new = {k: n for k, n in found.items() if n > ACCEPTED.get(k, 0)}
    assert not new, (
        "silent write path(s) added without a reason:\n  "
        + "\n  ".join(f"{f}::{fn}  ({n}, accepted {ACCEPTED.get((f, fn), 0)})"
                      for (f, fn), n in sorted(new.items()))
        + "\n\nEither log the exception, or add it to ACCEPTED in this file "
          "with a reason it is safe to lose silently.")


def test_the_accepted_list_has_not_gone_stale():
    """A fixed site left on the list makes the next one look accepted too."""
    found = _found()
    stale = {k: n for k, n in ACCEPTED.items() if found.get(k, 0) < n}
    assert not stale, (
        "ACCEPTED lists sites that no longer exist — drop them:\n  "
        + "\n  ".join(f"{f}::{fn}  (accepted {n}, found {found.get((f, fn), 0)})"
                      for (f, fn), n in sorted(stale.items())))


def test_the_detector_sees_a_swallowed_write(tmp_path):
    """Validated by breaking it on purpose, like the rest of the scans."""
    (tmp_path / "w.py").write_text(
        "def save(db):\n"
        "    try:\n"
        "        db.execute('INSERT INTO t VALUES (1)')\n"
        "    except Exception:\n"
        "        pass\n", encoding="utf-8")
    assert ("w.py", "save") in [(pathlib.Path(f).name, fn)
                                for f, fn in silent_write_sites(tmp_path)]


def test_the_detector_ignores_a_swallow_that_writes_nothing(tmp_path):
    """Only writes. A swallowed parse or lookup is a different argument."""
    (tmp_path / "r.py").write_text(
        "def look(d):\n"
        "    try:\n"
        "        return d['k']\n"
        "    except Exception:\n"
        "        pass\n", encoding="utf-8")
    assert silent_write_sites(tmp_path) == []


def test_a_logged_handler_is_not_a_silent_write(tmp_path):
    """The fix this test wants is a log line, so a logged handler must clear."""
    (tmp_path / "ok.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def save(db):\n"
        "    try:\n"
        "        db.execute('INSERT INTO t VALUES (1)')\n"
        "    except Exception as e:\n"
        "        logger.warning('write failed: %r', e)\n", encoding="utf-8")
    assert silent_write_sites(tmp_path) == []
