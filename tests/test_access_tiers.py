"""Tests for system access tiers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import Config
from app.core.access_tiers import (
    get_blocked_builtins,
    get_blocked_imports,
    get_blocked_shell_commands,
    is_path_allowed,
)


def _with_tier(tier: str):
    """Patch config to use a specific access tier."""
    return patch("app.core.access_tiers.config",
                 type("C", (), {"SYSTEM_ACCESS_LEVEL": tier})())


class TestShellTiers:
    def test_sandboxed_blocks_interpreters(self):
        with _with_tier("sandboxed"):
            blocked = get_blocked_shell_commands()
            assert "python" in blocked
            assert "python3" in blocked
            assert "node" in blocked

    def test_standard_allows_interpreters(self):
        with _with_tier("standard"):
            blocked = get_blocked_shell_commands()
            assert "python" not in blocked
            assert "python3" not in blocked

    def test_standard_blocks_system(self):
        with _with_tier("standard"):
            blocked = get_blocked_shell_commands()
            assert "shutdown" in blocked
            assert "systemctl" in blocked

    def test_full_only_blocks_escape(self):
        with _with_tier("full"):
            blocked = get_blocked_shell_commands()
            assert "docker" in blocked
            assert "nsenter" in blocked
            assert "chroot" in blocked
            assert "shutdown" not in blocked
            assert "python" not in blocked

    def test_container_escape_always_blocked(self):
        for tier in ("sandboxed", "standard", "full"):
            with _with_tier(tier):
                blocked = get_blocked_shell_commands()
                assert "docker" in blocked
                assert "nsenter" in blocked


class TestFilesystemTiers:
    def test_sandboxed_allows_data(self):
        with _with_tier("sandboxed"):
            assert is_path_allowed(Path("/data/test.txt"), write=False)
            assert is_path_allowed(Path("/data/test.txt"), write=True)

    def test_sandboxed_blocks_etc(self):
        with _with_tier("sandboxed"):
            assert not is_path_allowed(Path("/etc/hosts"), write=False)
            assert not is_path_allowed(Path("/etc/hosts"), write=True)

    def test_standard_reads_anywhere(self):
        with _with_tier("standard"):
            assert is_path_allowed(Path("/tmp/test.txt"), write=False)
            assert is_path_allowed(Path("/usr/bin/env"), write=False)

    def test_standard_writes_limited(self):
        with _with_tier("standard"):
            assert is_path_allowed(Path("/data/test.txt"), write=True)
            assert is_path_allowed(Path("/tmp/test.txt"), write=True)
            assert is_path_allowed(Path("/home/nova/test.txt"), write=True)
            assert not is_path_allowed(Path("/etc/hosts"), write=True)

    def test_full_writes_most(self):
        with _with_tier("full"):
            assert is_path_allowed(Path("/data/test.txt"), write=True)
            assert is_path_allowed(Path("/tmp/test.txt"), write=True)

    def test_shadow_always_protected(self):
        for tier in ("sandboxed", "standard", "full"):
            with _with_tier(tier):
                assert not is_path_allowed(Path("/etc/shadow"), write=True)

    def test_ssh_always_protected(self):
        for tier in ("sandboxed", "standard", "full"):
            with _with_tier(tier):
                assert not is_path_allowed(Path("/root/.ssh/id_rsa"), write=True)


class TestCodeExecTiers:
    def test_sandboxed_blocks_os(self):
        with _with_tier("sandboxed"):
            blocked = get_blocked_imports()
            assert "os" in blocked
            assert "subprocess" in blocked
            assert "pathlib" in blocked

    def test_standard_allows_os_pathlib(self):
        with _with_tier("standard"):
            blocked = get_blocked_imports()
            assert "os" not in blocked
            assert "pathlib" not in blocked
            assert "glob" not in blocked
            assert "sys" not in blocked
            assert "subprocess" in blocked  # still blocked

    def test_full_minimal_blocks(self):
        with _with_tier("full"):
            blocked = get_blocked_imports()
            assert "ctypes" in blocked
            assert "multiprocessing" in blocked
            assert "os" not in blocked
            assert "subprocess" not in blocked

    def test_sandboxed_blocks_builtins(self):
        with _with_tier("sandboxed"):
            blocked = get_blocked_builtins()
            assert "eval(" in blocked
            assert "exec(" in blocked
            assert "__import__" in blocked

    def test_full_minimal_builtins(self):
        with _with_tier("full"):
            blocked = get_blocked_builtins()
            assert "__import__" in blocked  # always blocked
            assert "eval(" not in blocked   # allowed at full


class TestConfigValidation:
    def test_invalid_access_level_warns(self):
        cfg = Config(SYSTEM_ACCESS_LEVEL="stanard")
        warnings = cfg.validate()
        assert any("SYSTEM_ACCESS_LEVEL" in w for w in warnings)

    def test_valid_access_levels_no_warning(self):
        for tier in ("sandboxed", "standard", "full"):
            cfg = Config(SYSTEM_ACCESS_LEVEL=tier)
            warnings = cfg.validate()
            assert not any("SYSTEM_ACCESS_LEVEL" in w for w in warnings)

    def test_invalid_tier_falls_back_to_sandboxed(self):
        with _with_tier("typo"):
            blocked = get_blocked_shell_commands()
            # Falls back to sandboxed, which blocks interpreters
            assert "python" in blocked
