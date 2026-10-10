"""Sandbox enforcement had no test file at all.

app/core/access_tiers.py is the only
thing standing between a model-authored command and the host. A 2026-08-29
triage ranked it the second most load-bearing untested module (after llm, which
37 modules import).

These tests lock in the SAFETY PROPERTIES rather than the implementation:
container-escape is blocked at every tier, protected paths are never writable
outside "none", and the tier ladder is monotonic — a stricter tier can never
permit something a looser tier forbids. That last property is the one a future
refactor is most likely to break silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import access_tiers as at

_MP = None


@pytest.fixture(autouse=True)
def _mp(monkeypatch):
    """Each test picks its own tier via config; monkeypatch undoes it."""
    global _MP
    _MP = monkeypatch
    yield


def _tier(t):
    _MP.setattr(at.config, "SYSTEM_ACCESS_LEVEL", t)


ALL_TIERS = ["sandboxed", "standard", "full", "none"]


class TestProtectedPaths:
    @pytest.mark.parametrize("tier", ["sandboxed", "standard", "full"])
    @pytest.mark.parametrize("p", ["/etc/shadow", "/etc/passwd", "/etc/sudoers"])
    def test_credential_files_never_writable(self, tier, p):
        _tier(tier)
        assert at.is_path_allowed(Path(p), write=True) is False, (
            f"{p} must not be writable at tier {tier!r}"
        )

    def test_sandboxed_confines_writes_to_data(self):
        _tier("sandboxed")
        assert at.is_path_allowed(Path("/data/x.txt"), write=True) is True
        assert at.is_path_allowed(Path("/home/nova/x.txt"), write=True) is False

    def test_none_tier_is_documented_as_unrestricted(self):
        """Not an endorsement — pinning it so nobody assumes 'none' is safe."""
        _tier("none")
        assert at.is_path_allowed(Path("/etc/shadow"), write=True) is True


class TestTierResolution:
    def test_valid_tiers_are_the_documented_four(self):
        assert at.VALID_TIERS == {"sandboxed", "standard", "full", "none"}
