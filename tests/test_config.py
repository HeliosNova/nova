"""Tests for Config class and _ConfigProxy."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class TestConfigDefaults:
    """Test default Config creation."""

    def test_default_provider_is_ollama(self):
        from app.config import Config
        cfg = Config()
        assert cfg.LLM_PROVIDER == "ollama"

    def test_default_port_is_8000(self):
        from app.config import Config
        cfg = Config()
        assert cfg.PORT == 8000

    def test_default_max_history_messages(self):
        from app.config import Config
        cfg = Config()
        assert cfg.MAX_HISTORY_MESSAGES == 20

    def test_default_booleans(self):
        from app.config import Config
        cfg = Config()
        assert cfg.ENABLE_HEARTBEAT is True
        assert cfg.ENABLE_VOICE is False
        assert cfg.ENABLE_SHELL_EXEC is False


class TestConfigUpdate:
    """Test update() with type coercion."""

    def test_update_string_field(self):
        from app.config import Config
        cfg = Config()
        warnings = cfg.update(LLM_MODEL="gpt-4o")
        assert cfg.LLM_MODEL == "gpt-4o"
        assert isinstance(warnings, list)

    def test_update_bool_from_string(self):
        from app.config import Config
        cfg = Config()
        cfg.update(ENABLE_VOICE="true")
        assert cfg.ENABLE_VOICE is True
        cfg.update(ENABLE_VOICE="false")
        assert cfg.ENABLE_VOICE is False
        cfg.update(ENABLE_VOICE="1")
        assert cfg.ENABLE_VOICE is True

    def test_update_int_from_string(self):
        from app.config import Config
        cfg = Config()
        cfg.update(MAX_TOOL_ROUNDS="10")
        assert cfg.MAX_TOOL_ROUNDS == 10
        assert isinstance(cfg.MAX_TOOL_ROUNDS, int)

    def test_update_immutable_field_rejected(self):
        """Immutable fields (PORT, DB_PATH, etc.) should not be changed via update()."""
        from app.config import Config
        cfg = Config()
        original_port = cfg.PORT
        cfg.update(PORT="9090")
        assert cfg.PORT == original_port  # unchanged

    def test_update_ignores_unknown_keys(self):
        from app.config import Config
        cfg = Config()
        cfg.update(NONEXISTENT_KEY="foo")
        assert not hasattr(cfg, "NONEXISTENT_KEY") or getattr(cfg, "NONEXISTENT_KEY", None) != "foo"

    def test_update_ignores_private_keys(self):
        from app.config import Config
        cfg = Config()
        cfg.update(_SENSITIVE_FIELDS="overwritten")
        # Should still be a frozenset, not overwritten
        assert isinstance(cfg._SENSITIVE_FIELDS, frozenset)


class TestConfigValidate:
    """Test validate() catches invalid values."""

    def test_invalid_llm_provider(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "LLM_PROVIDER", "invalid_provider")
        warnings = cfg.validate()
        assert any("LLM_PROVIDER" in w for w in warnings)

    def test_invalid_port_zero(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "PORT", 0)
        warnings = cfg.validate()
        assert any("PORT" in w for w in warnings)

    def test_invalid_port_too_high(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "PORT", 99999)
        warnings = cfg.validate()
        assert any("PORT" in w for w in warnings)

    def test_valid_config_no_warnings(self):
        from app.config import Config
        cfg = Config()
        warnings = cfg.validate()
        assert warnings == []

    def test_cloud_provider_without_key(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "LLM_PROVIDER", "openai")
        object.__setattr__(cfg, "OPENAI_API_KEY", "")
        warnings = cfg.validate()
        assert any("OPENAI_API_KEY" in w for w in warnings)

    def test_invalid_system_access_level(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "SYSTEM_ACCESS_LEVEL", "elevated")
        warnings = cfg.validate()
        assert any("SYSTEM_ACCESS_LEVEL" in w for w in warnings)


class TestConfigOverrides:
    """Test _save_overrides and _load_overrides roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        import app.config as config_mod
        overrides_file = tmp_path / "overrides.json"

        # Patch the module-level _OVERRIDES_PATH
        original = config_mod._OVERRIDES_PATH
        config_mod._OVERRIDES_PATH = overrides_file
        try:
            from app.config import Config
            cfg = Config()
            object.__setattr__(cfg, "LLM_MODEL", "custom-model")
            object.__setattr__(cfg, "RETRIEVAL_TOP_K", 10)
            cfg._save_overrides(["LLM_MODEL", "RETRIEVAL_TOP_K"])

            assert overrides_file.exists()
            saved = json.loads(overrides_file.read_text())
            assert saved["LLM_MODEL"] == "custom-model"
            assert saved["RETRIEVAL_TOP_K"] == 10

            # Load into a fresh config — only mutable fields are loaded
            cfg2 = Config()
            cfg2._load_overrides()
            assert cfg2.LLM_MODEL == "custom-model"
            assert cfg2.RETRIEVAL_TOP_K == 10
        finally:
            config_mod._OVERRIDES_PATH = original

    def test_load_missing_file_is_noop(self, tmp_path):
        import app.config as config_mod
        overrides_file = tmp_path / "nonexistent.json"
        original = config_mod._OVERRIDES_PATH
        config_mod._OVERRIDES_PATH = overrides_file
        try:
            from app.config import Config
            cfg = Config()
            cfg._load_overrides()  # should not raise
            assert cfg.PORT == 8000  # default unchanged
        finally:
            config_mod._OVERRIDES_PATH = original


class TestConfigRedaction:
    """Test sensitive field redaction."""

    def test_to_dict_redacts_sensitive(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "OPENAI_API_KEY", "sk-secret-key-12345")
        d = cfg.to_dict(redact_sensitive=True)
        assert d["OPENAI_API_KEY"] == "***"

    def test_to_dict_no_redact(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "OPENAI_API_KEY", "sk-secret-key-12345")
        d = cfg.to_dict(redact_sensitive=False)
        assert d["OPENAI_API_KEY"] == "sk-secret-key-12345"

    def test_empty_sensitive_field_not_redacted(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "OPENAI_API_KEY", "")
        d = cfg.to_dict(redact_sensitive=True)
        assert d["OPENAI_API_KEY"] == ""

    def test_repr_redacts_sensitive(self):
        from app.config import Config
        cfg = Config()
        object.__setattr__(cfg, "OPENAI_API_KEY", "sk-secret-key-12345")
        r = repr(cfg)
        assert "sk-secret-key-12345" not in r
        assert "***" in r


class TestConfigProxy:
    """Test _ConfigProxy delegates to real config."""

    def test_proxy_reads_config_values(self):
        from app.config import config
        assert config.LLM_PROVIDER == "ollama"

    def test_proxy_reflects_reset(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "test-model-xyz")
        from app.config import reset_config, config
        reset_config()
        assert config.LLM_MODEL == "test-model-xyz"

    def test_proxy_repr(self):
        from app.config import config
        r = repr(config)
        assert "Config(" in r
