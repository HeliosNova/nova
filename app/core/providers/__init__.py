"""LLM providers — pluggable backends for Nova's brain."""

__all__ = ["OllamaProvider", "OpenAIProvider", "AnthropicProvider", "GoogleProvider"]

_PROVIDERS = {
    "OllamaProvider": "app.core.providers.ollama",
    "OpenAIProvider": "app.core.providers.openai",
    "AnthropicProvider": "app.core.providers.anthropic",
    "GoogleProvider": "app.core.providers.google",
}

def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib
        module = importlib.import_module(_PROVIDERS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
