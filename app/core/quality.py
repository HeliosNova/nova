"""Shared quality assessment utilities — single source for critique and reflexion.

Consolidates duplicated constants and helper functions from critique.py
and reflexion.py into one module.
"""

from __future__ import annotations

from app.tools.base import TOOL_FAILURE_MARKERS


# Browser selector misses are soft failures — retriable, not hard errors
BROWSER_SELECTOR_HINTS = ("selector", "not found", "timed out waiting")


def all_tools_clean(tool_results: list[dict]) -> bool:
    """Return True if every tool result is clean (no hard failure markers).

    Browser selector misses are retriable, not hard failures.
    Used by both critique.py and reflexion.py.
    """
    for tr in tool_results:
        output = str(tr.get("output", "")).lower()
        error = str(tr.get("error", "")).lower()
        combined = output + " " + error
        if output.startswith("[tool"):
            return False
        if any(marker in combined for marker in TOOL_FAILURE_MARKERS):
            # Browser selector misses are soft — skip them
            if any(hint in combined for hint in BROWSER_SELECTOR_HINTS):
                continue
            return False
    return True
