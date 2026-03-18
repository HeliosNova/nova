"""Nova configuration — loaded from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        import logging
        logging.getLogger(__name__).warning("Invalid integer for %s='%s', using default %d", key, val, default)
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        import logging
        logging.getLogger(__name__).warning("Invalid float for %s='%s', using default %s", key, val, default)
        return default


_OVERRIDES_PATH = Path(os.getenv("CONFIG_OVERRIDES_PATH", "/data/config_overrides.json"))


@dataclass
class Config:
    # LLM
    LLM_PROVIDER: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    LLM_MODEL: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5:27b"))
    OLLAMA_URL: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://ollama:11434"))

    # Multi-provider API keys + models
    OPENAI_API_KEY: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    OPENAI_MODEL: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o"))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    ANTHROPIC_MODEL: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
    GOOGLE_API_KEY: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))
    GOOGLE_MODEL: str = field(default_factory=lambda: _env("GOOGLE_MODEL", "gemini-2.0-flash"))

    # MCP (Model Context Protocol) — client (consume external MCP tools)
    ENABLE_MCP: bool = field(default_factory=lambda: _env("ENABLE_MCP", "true").lower() == "true")
    MCP_CONFIG_DIR: str = field(default_factory=lambda: _env("MCP_CONFIG_DIR", "/data/mcp"))

    # MCP Server (expose Nova as MCP server)
    ENABLE_MCP_SERVER: bool = field(default_factory=lambda: _env("ENABLE_MCP_SERVER", "true").lower() == "true")
    MCP_SERVER_NAME: str = field(default_factory=lambda: _env("MCP_SERVER_NAME", "nova"))

    # External skills (AgentSkills / OpenClaw)
    SKILLS_DIR: str = field(default_factory=lambda: _env("SKILLS_DIR", "/data/skills"))

    # Memory
    MAX_HISTORY_MESSAGES: int = field(default_factory=lambda: _env_int("MAX_HISTORY_MESSAGES", 20))
    MAX_LESSONS_IN_PROMPT: int = field(default_factory=lambda: _env_int("MAX_LESSONS_IN_PROMPT", 5))
    MAX_SKILLS_CHECK: int = field(default_factory=lambda: _env_int("MAX_SKILLS_CHECK", 10))

    # Context window management
    MAX_CONTEXT_TOKENS: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_TOKENS", 16000))
    RECENT_MESSAGES_KEEP: int = field(default_factory=lambda: _env_int("RECENT_MESSAGES_KEEP", 12))

    # Retrieval
    EMBEDDING_MODEL: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "nomic-embed-text-v2-moe"))
    RETRIEVAL_TOP_K: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    CHUNK_SIZE: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    RRF_K: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # Tools
    SEARXNG_URL: str = field(default_factory=lambda: _env("SEARXNG_URL", "http://searxng:8080"))
    CODE_EXEC_TIMEOUT: int = field(default_factory=lambda: _env_int("CODE_EXEC_TIMEOUT", 10))
    MAX_TOOL_ROUNDS: int = field(default_factory=lambda: _env_int("MAX_TOOL_ROUNDS", 5))
    SHELL_EXEC_TIMEOUT: int = field(default_factory=lambda: _env_int("SHELL_EXEC_TIMEOUT", 30))
    BROWSER_TIMEOUT: int = field(default_factory=lambda: _env_int("BROWSER_TIMEOUT", 30))
    BROWSER_CDP_URL: str = field(default_factory=lambda: _env("BROWSER_CDP_URL", ""))  # http:// CDP URL to connect to host browser
    TOOL_TIMEOUT: int = field(default_factory=lambda: _env_int("TOOL_TIMEOUT", 120))
    GENERATION_TIMEOUT: int = field(default_factory=lambda: _env_int("GENERATION_TIMEOUT", 480))
    ENABLE_SHELL_EXEC: bool = field(default_factory=lambda: _env("ENABLE_SHELL_EXEC", "false").lower() == "true")

    # Desktop automation (requires display server + PyAutoGUI)
    ENABLE_DESKTOP_AUTOMATION: bool = field(default_factory=lambda: _env("ENABLE_DESKTOP_AUTOMATION", "false").lower() == "true")
    DESKTOP_CLICK_DELAY: float = field(default_factory=lambda: _env_float("DESKTOP_CLICK_DELAY", 0.5))

    # Heartbeat / Proactive
    ENABLE_HEARTBEAT: bool = field(default_factory=lambda: _env("ENABLE_HEARTBEAT", "true").lower() == "true")
    HEARTBEAT_INTERVAL: int = field(default_factory=lambda: _env_int("HEARTBEAT_INTERVAL", 60))
    ENABLE_PROACTIVE: bool = field(default_factory=lambda: _env("ENABLE_PROACTIVE", "true").lower() == "true")
    DIGEST_HOUR: int = field(default_factory=lambda: _env_int("DIGEST_HOUR", 21))

    # Learning
    TRAINING_DATA_PATH: str = field(default_factory=lambda: _env("TRAINING_DATA_PATH", "/data/training_data.jsonl"))
    MAX_TRAINING_PAIRS: int = field(default_factory=lambda: _env_int("MAX_TRAINING_PAIRS", 10000))
    MAX_LESSONS: int = field(default_factory=lambda: _env_int("MAX_LESSONS", 500))
    TRAINING_DATA_CHANNELS: str = field(default_factory=lambda: _env("TRAINING_DATA_CHANNELS", "api"))  # comma-separated: api,discord,telegram,whatsapp,signal

    # Fine-tuning automation
    FINETUNE_MIN_NEW_PAIRS: int = field(default_factory=lambda: _env_int("FINETUNE_MIN_NEW_PAIRS", 50))
    FINETUNE_OUTPUT_DIR: str = field(default_factory=lambda: _env("FINETUNE_OUTPUT_DIR", "/data/finetune"))
    FINETUNE_EVAL_HOLDOUT: int = field(default_factory=lambda: _env_int("FINETUNE_EVAL_HOLDOUT", 10))

    # Reasoning
    ENABLE_PLANNING: bool = field(default_factory=lambda: _env("ENABLE_PLANNING", "true").lower() == "true")
    ENABLE_CRITIQUE: bool = field(default_factory=lambda: _env("ENABLE_CRITIQUE", "true").lower() == "true")
    ENABLE_CUSTOM_TOOLS: bool = field(default_factory=lambda: _env("ENABLE_CUSTOM_TOOLS", "true").lower() == "true")
    ENABLE_EXTENDED_THINKING: bool = field(default_factory=lambda: _env("ENABLE_EXTENDED_THINKING", "true").lower() == "true")

    # Model routing
    VISION_MODEL: str = field(default_factory=lambda: _env("VISION_MODEL", "qwen3.5:9b"))
    FAST_MODEL: str = field(default_factory=lambda: _env("FAST_MODEL", "qwen3.5:4b"))
    HEAVY_MODEL: str = field(default_factory=lambda: _env("HEAVY_MODEL", ""))
    ENABLE_MODEL_ROUTING: bool = field(default_factory=lambda: _env("ENABLE_MODEL_ROUTING", "true").lower() == "true")

    # Critique
    MAX_CRITIQUE_ROUNDS: int = field(default_factory=lambda: _env_int("MAX_CRITIQUE_ROUNDS", 3))

    # Delegation (multi-agent)
    ENABLE_DELEGATION: bool = field(default_factory=lambda: _env("ENABLE_DELEGATION", "true").lower() == "true")
    MAX_DELEGATION_DEPTH: int = field(default_factory=lambda: _env_int("MAX_DELEGATION_DEPTH", 1))

    # Background tasks
    MAX_BACKGROUND_TASKS: int = field(default_factory=lambda: _env_int("MAX_BACKGROUND_TASKS", 5))
    BACKGROUND_TASK_TIMEOUT: int = field(default_factory=lambda: _env_int("BACKGROUND_TASK_TIMEOUT", 300))

    # Auto skill creation
    ENABLE_AUTO_SKILL_CREATION: bool = field(default_factory=lambda: _env("ENABLE_AUTO_SKILL_CREATION", "true").lower() == "true")

    # Skill import/export signing
    REQUIRE_SIGNED_SKILLS: bool = field(default_factory=lambda: _env("REQUIRE_SIGNED_SKILLS", "true").lower() == "true")

    # Curiosity / autonomy
    ENABLE_CURIOSITY: bool = field(default_factory=lambda: _env("ENABLE_CURIOSITY", "true").lower() == "true")

    # Voice (local Whisper speech-to-text)
    ENABLE_VOICE: bool = field(default_factory=lambda: _env("ENABLE_VOICE", "false").lower() == "true")
    WHISPER_MODEL_SIZE: str = field(default_factory=lambda: _env("WHISPER_MODEL_SIZE", "base"))
    VOICE_MAX_DURATION: int = field(default_factory=lambda: _env_int("VOICE_MAX_DURATION", 300))

    # Security
    ENABLE_INJECTION_DETECTION: bool = field(default_factory=lambda: _env("ENABLE_INJECTION_DETECTION", "true").lower() == "true")

    # System access tiers (sandboxed | standard | full | none)
    SYSTEM_ACCESS_LEVEL: str = field(default_factory=lambda: _env("SYSTEM_ACCESS_LEVEL", "sandboxed"))

    # Integrations
    ENABLE_INTEGRATIONS: bool = field(default_factory=lambda: _env("ENABLE_INTEGRATIONS", "true").lower() == "true")

    # Action: Email
    ENABLE_EMAIL_SEND: bool = field(default_factory=lambda: _env("ENABLE_EMAIL_SEND", "false").lower() == "true")
    EMAIL_SMTP_HOST: str = field(default_factory=lambda: _env("EMAIL_SMTP_HOST"))
    EMAIL_SMTP_PORT: int = field(default_factory=lambda: _env_int("EMAIL_SMTP_PORT", 587))
    EMAIL_SMTP_USER: str = field(default_factory=lambda: _env("EMAIL_SMTP_USER"))
    EMAIL_SMTP_PASS: str = field(default_factory=lambda: _env("EMAIL_SMTP_PASS"))
    EMAIL_FROM: str = field(default_factory=lambda: _env("EMAIL_FROM"))
    EMAIL_SMTP_TLS: bool = field(default_factory=lambda: _env("EMAIL_SMTP_TLS", "true").lower() == "true")
    EMAIL_ALLOWED_RECIPIENTS: str = field(default_factory=lambda: _env("EMAIL_ALLOWED_RECIPIENTS"))

    # Action: Calendar
    ENABLE_CALENDAR: bool = field(default_factory=lambda: _env("ENABLE_CALENDAR", "true").lower() == "true")
    CALENDAR_PATH: str = field(default_factory=lambda: _env("CALENDAR_PATH", "/data/calendar.ics"))

    # Action: Webhooks
    ENABLE_WEBHOOKS: bool = field(default_factory=lambda: _env("ENABLE_WEBHOOKS", "false").lower() == "true")
    WEBHOOK_ALLOWED_URLS: str = field(default_factory=lambda: _env("WEBHOOK_ALLOWED_URLS"))

    # Channels
    DISCORD_TOKEN: str = field(default_factory=lambda: _env("DISCORD_TOKEN"))
    DISCORD_CHANNEL_ID: str = field(default_factory=lambda: _env("DISCORD_CHANNEL_ID"))
    DISCORD_ALLOWED_USERS: str = field(default_factory=lambda: _env("DISCORD_ALLOWED_USERS"))
    TELEGRAM_TOKEN: str = field(default_factory=lambda: _env("TELEGRAM_TOKEN"))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    TELEGRAM_ALLOWED_USERS: str = field(default_factory=lambda: _env("TELEGRAM_ALLOWED_USERS"))

    # Channel: WhatsApp (Business API or bridge)
    WHATSAPP_API_URL: str = field(default_factory=lambda: _env("WHATSAPP_API_URL"))
    WHATSAPP_API_TOKEN: str = field(default_factory=lambda: _env("WHATSAPP_API_TOKEN"))
    WHATSAPP_VERIFY_TOKEN: str = field(default_factory=lambda: _env("WHATSAPP_VERIFY_TOKEN"))
    WHATSAPP_PHONE_ID: str = field(default_factory=lambda: _env("WHATSAPP_PHONE_ID"))
    WHATSAPP_CHAT_ID: str = field(default_factory=lambda: _env("WHATSAPP_CHAT_ID"))
    WHATSAPP_ALLOWED_USERS: str = field(default_factory=lambda: _env("WHATSAPP_ALLOWED_USERS"))

    # Channel: Signal (via signal-cli REST API)
    SIGNAL_API_URL: str = field(default_factory=lambda: _env("SIGNAL_API_URL"))
    SIGNAL_PHONE_NUMBER: str = field(default_factory=lambda: _env("SIGNAL_PHONE_NUMBER"))
    SIGNAL_CHAT_ID: str = field(default_factory=lambda: _env("SIGNAL_CHAT_ID"))
    SIGNAL_ALLOWED_USERS: str = field(default_factory=lambda: _env("SIGNAL_ALLOWED_USERS"))
    SIGNAL_POLL_INTERVAL: int = field(default_factory=lambda: _env_int("SIGNAL_POLL_INTERVAL", 2))

    # Auth
    API_KEY: str = field(default_factory=lambda: _env("NOVA_API_KEY"))
    ALLOWED_ORIGINS: str = field(default_factory=lambda: _env("ALLOWED_ORIGINS", "http://localhost:5173"))

    # Server
    HOST: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: _env_int("PORT", 8000))

    # Database
    DB_PATH: str = field(default_factory=lambda: _env("DB_PATH", "/data/nova.db"))
    CHROMADB_PATH: str = field(default_factory=lambda: _env("CHROMADB_PATH", "/data/chromadb"))

    # Sensitive field names — redacted in __repr__/__str__ to prevent secret leakage
    _SENSITIVE_FIELDS = frozenset({
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "EMAIL_SMTP_PASS", "DISCORD_TOKEN", "TELEGRAM_TOKEN",
        "WHATSAPP_API_TOKEN", "WHATSAPP_VERIFY_TOKEN", "API_KEY",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value) -> None:
        """Warn on direct attribute mutation after init. Use config.update() instead."""
        if getattr(self, "_initialized", False) and not name.startswith("_"):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Direct config mutation: %s. Use config.update() for runtime changes.", name
            )
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        fields = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f in self._SENSITIVE_FIELDS and val:
                fields.append(f"{f}='***'")
            else:
                fields.append(f"{f}={val!r}")
        return f"Config({', '.join(fields)})"

    def __str__(self) -> str:
        return self.__repr__()

    def update(self, **kwargs) -> list[str]:
        """Update config values at runtime. Returns validation warnings."""
        for key, value in kwargs.items():
            if not hasattr(self, key) or key.startswith('_'):
                continue
            # Type coerce based on current field type
            current = getattr(self, key)
            if isinstance(current, bool):
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes')
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            object.__setattr__(self, key, value)
        return self.validate()

    def to_dict(self, redact_sensitive: bool = True) -> dict:
        """Export all config values as dict. Redacts sensitive fields by default."""
        result = {}
        for f in self.__dataclass_fields__:
            if f.startswith('_'):
                continue
            val = getattr(self, f)
            if redact_sensitive and f in self._SENSITIVE_FIELDS and val:
                result[f] = "***"
            else:
                result[f] = val
        return result

    def _save_overrides(self, keys: list[str]) -> None:
        """Save changed keys to overrides file."""
        overrides = {}
        if _OVERRIDES_PATH.exists():
            try:
                overrides = json.loads(_OVERRIDES_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        for key in keys:
            if hasattr(self, key) and not key.startswith('_'):
                overrides[key] = getattr(self, key)
        try:
            _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
            _OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
        except OSError:
            pass

    def _load_overrides(self) -> None:
        """Apply saved overrides from file."""
        if not _OVERRIDES_PATH.exists():
            return
        try:
            overrides = json.loads(_OVERRIDES_PATH.read_text())
            self.update(**overrides)
        except (json.JSONDecodeError, OSError):
            pass

    def validate(self) -> list[str]:
        """Validate config values. Returns list of warning messages (empty = valid)."""
        warnings = []

        valid_providers = ("ollama", "openai", "anthropic", "google")
        if self.LLM_PROVIDER not in valid_providers:
            warnings.append(
                f"LLM_PROVIDER must be one of {valid_providers}, got: '{self.LLM_PROVIDER}'"
            )

        # Require API key for cloud providers
        provider_keys = {
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "google": self.GOOGLE_API_KEY,
        }
        if self.LLM_PROVIDER in provider_keys and not provider_keys[self.LLM_PROVIDER]:
            key_name = f"{self.LLM_PROVIDER.upper()}_API_KEY"
            warnings.append(f"{key_name} is required when LLM_PROVIDER={self.LLM_PROVIDER}")

        if not self.OLLAMA_URL.startswith(("http://", "https://")):
            warnings.append(f"OLLAMA_URL must start with http:// or https://, got: {self.OLLAMA_URL}")

        if not (1 <= self.PORT <= 65535):
            warnings.append(f"PORT must be 1-65535, got: {self.PORT}")

        if self.SEARXNG_URL and not self.SEARXNG_URL.startswith(("http://", "https://")):
            warnings.append(f"SEARXNG_URL must start with http:// or https://, got: {self.SEARXNG_URL}")

        if self.MAX_CONTEXT_TOKENS < 1000:
            warnings.append(f"MAX_CONTEXT_TOKENS too low: {self.MAX_CONTEXT_TOKENS} (minimum 1000)")

        if self.SYSTEM_ACCESS_LEVEL.lower() not in ("sandboxed", "standard", "full", "none"):
            warnings.append(
                f"SYSTEM_ACCESS_LEVEL must be sandboxed/standard/full/none, got '{self.SYSTEM_ACCESS_LEVEL}'"
            )

        if not (0 <= self.DIGEST_HOUR <= 23):
            warnings.append(f"DIGEST_HOUR must be 0-23, got: {self.DIGEST_HOUR}")

        if self.HEARTBEAT_INTERVAL < 1:
            warnings.append(f"HEARTBEAT_INTERVAL must be >= 1, got: {self.HEARTBEAT_INTERVAL}")

        return warnings


# ---------------------------------------------------------------------------
# Singleton management — lazy init with proxy for test swapability
# ---------------------------------------------------------------------------

_config_instance: Config | None = None


def get_config() -> Config:
    """Get the Config singleton. Creates on first access."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _config_instance._load_overrides()
    return _config_instance


def reset_config() -> Config:
    """Recreate Config from current env vars. For testing."""
    global _config_instance
    _config_instance = Config()
    _config_instance._load_overrides()
    return _config_instance


class _ConfigProxy:
    """Proxy that delegates to the real Config singleton.

    All modules that do `from app.config import config` get this proxy.
    When tests call reset_config(), every module automatically sees
    the new Config values — no importlib.reload or module-walking needed.
    """

    def __getattr__(self, name: str):
        return getattr(get_config(), name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(get_config(), name, value)

    def __repr__(self) -> str:
        return repr(get_config())

    def __str__(self) -> str:
        return str(get_config())


# Module-level config — a proxy that delegates to the real singleton.
# All existing `from app.config import config` imports work unchanged.
config = _ConfigProxy()
