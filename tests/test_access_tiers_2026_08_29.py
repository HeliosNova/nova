"""Sandbox enforcement had no test file at all.

app/core/access_tiers.py is 321 lines, imported by 16 modules, and is the only
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


@pytest.fixture(autouse=True)
def _clean_override():
    """Each test picks its own tier; never leak one into the next."""
    at.set_access_tier_override(None)
    yield
    at.set_access_tier_override(None)


def _tier(t):
    at.set_access_tier_override(t)


ALL_TIERS = ["sandboxed", "standard", "full", "none"]
ESCAPE = ["docker", "podman", "nsenter", "chroot", "unshare"]


class TestContainerEscape:
    """The one guarantee that must hold everywhere except explicit 'none'."""

    @pytest.mark.parametrize("tier", ["sandboxed", "standard", "full"])
    @pytest.mark.parametrize("cmd", ESCAPE)
    def test_escape_blocked_at_every_real_tier(self, tier, cmd):
        _tier(tier)
        assert at.is_command_blocked(cmd) is not None, (
            f"{cmd!r} is a container-escape primitive and must be blocked at "
            f"tier {tier!r}"
        )


class TestTierLadderIsMonotonic:
    """Stricter tiers must be supersets of looser ones. A refactor that reorders
    a set literal could silently invert this and nothing else would notice."""

    def _blocked(self, tier, cmds):
        _tier(tier)
        return {c for c in cmds if at.is_command_blocked(c) is not None}

    def test_sandboxed_blocks_at_least_what_standard_blocks(self):
        probe = ESCAPE + ["shutdown", "mkfs", "iptables", "systemctl",
                          "python3", "node", "ls", "cat"]
        assert self._blocked("standard", probe) <= self._blocked("sandboxed", probe)

    def test_standard_blocks_at_least_what_full_blocks(self):
        probe = ESCAPE + ["shutdown", "mkfs", "iptables", "systemctl",
                          "python3", "node", "ls", "cat"]
        assert self._blocked("full", probe) <= self._blocked("standard", probe)

    def test_interpreters_blocked_only_at_sandboxed(self):
        for cmd in ("python3", "node"):
            _tier("sandboxed")
            assert at.is_command_blocked(cmd) is not None
            _tier("standard")
            assert at.is_command_blocked(cmd) is None, (
                f"{cmd} should be permitted at standard — sandboxed is the tier "
                f"that blocks interpreters"
            )

    def test_system_commands_blocked_below_full(self):
        for cmd in ("shutdown", "mkfs", "iptables", "useradd"):
            for tier in ("sandboxed", "standard"):
                _tier(tier)
                assert at.is_command_blocked(cmd) is not None, f"{cmd} at {tier}"


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


class TestCodeExecutionGates:
    def test_ctypes_and_multiprocessing_always_blocked(self):
        for tier in ("sandboxed", "standard", "full"):
            _tier(tier)
            blocked = at.get_blocked_imports()
            assert "ctypes" in blocked and "multiprocessing" in blocked, tier

    def test_sandboxed_blocks_more_imports_than_full(self):
        _tier("sandboxed")
        strict = at.get_blocked_imports()
        _tier("full")
        loose = at.get_blocked_imports()
        assert loose <= strict, "full must not block more than sandboxed"
        assert len(strict) > len(loose)

    def test_sandboxed_blocks_dangerous_builtins(self):
        """NB: entries are SOURCE-SCAN PATTERNS, not bare names — callables are
        stored as 'eval(' so a substring scan cannot match `evaluate(`. Dunders
        like '__import__' are stored bare. Asserting bare names here was my own
        error on the first pass; encoding the real format so the next reader
        does not repeat it."""
        _tier("sandboxed")
        b = set(at.get_blocked_builtins())
        for pattern in ("eval(", "exec(", "open(", "compile(", "getattr("):
            assert pattern in b, f"{pattern!r} must be blocked at sandboxed"
        for bare in ("__import__", "__builtins__"):
            assert bare in b, f"{bare!r} must be blocked at sandboxed"

    def test_callable_patterns_keep_their_paren(self):
        """The paren is load-bearing: without it a substring scan for 'eval'
        would also flag a variable named `evaluation`."""
        _tier("sandboxed")
        for p in at.get_blocked_builtins():
            if not p.startswith("__"):
                assert p.endswith("("), (
                    f"{p!r} has no trailing paren — a bare-name scan would "
                    f"false-positive on identifiers containing it"
                )


class TestTierResolution:
    def test_invalid_override_falls_back_safely(self):
        at.set_access_tier_override("not-a-tier")
        # must not raise, and must not silently grant a permissive tier
        assert at.is_command_blocked("docker") is not None

    def test_override_round_trips(self):
        at.set_access_tier_override("standard")
        assert at.get_access_tier_override() == "standard"
        at.set_access_tier_override(None)
        assert at.get_access_tier_override() is None

    def test_valid_tiers_are_the_documented_four(self):
        assert at.VALID_TIERS == {"sandboxed", "standard", "full", "none"}
