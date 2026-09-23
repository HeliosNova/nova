"""A validator refusal is not parser drift (2026-09-23).

"FORECAST line present but not stored — mint format drift?" is the
2026-09-07 instrument for forecasts LOST to the parser (44 in eleven days,
a quarter of storyline forecasts). The validator shipped on 2026-09-22
returned None on a refusal too, so the first consolidation under it
(01:46 UTC) refused 6 of 7 candidates and logged every one as drift — the
instrument would have read a healthy validator as a broken parser. The
minter now returns forecasts.REJECTED for a refusal; both call sites warn
only on None.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import app as _app_pkg
from app.core import forecasts

ROOT = Path(_app_pkg.__file__).resolve().parent
LINE = "FORECAST: The consortium publishes its charter | resolves 2026-12-31 | 0.6 confidence"


def test_rejected_is_falsy_and_not_none():
    assert forecasts.REJECTED is not None
    assert not forecasts.REJECTED


@pytest.mark.asyncio
async def test_the_three_outcomes_are_distinguishable(db):
    refused = AsyncMock(return_value=(False, "claim", None, "already happened"))
    with patch("app.core.forecasts.validate_candidate", refused):
        assert await forecasts.parse_and_store_forecast_ensembled(db, LINE) == forecasts.REJECTED
    assert await forecasts.parse_and_store_forecast_ensembled(db, "FORECAST: none") is None
    assert await forecasts.parse_and_store_forecast_ensembled(db, "no forecast here") is None
    assert db.fetchone("SELECT count(*) AS c FROM forecasts")["c"] == 0


def _drift_guard_tests_for_none(path: Path) -> bool:
    """True when every `... in out.upper()` drift warning in the file is guarded
    by an explicit `is None` comparison on the minter's result."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        src = ast.unparse(node.test)
        if "FORECAST:" not in src or "FORECAST: NONE" not in src:
            continue
        found = True
        if " is None" not in src:
            return False
    return found


@pytest.mark.parametrize("module", ["core/dossiers.py", "core/storylines.py"])
def test_the_drift_warning_keys_on_none(module):
    assert _drift_guard_tests_for_none(ROOT / module), module
