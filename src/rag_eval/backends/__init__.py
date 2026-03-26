from .base import BaseBackend


def _lazy_import(name):
    """Return a class that defers the real import until instantiation."""
    def _get_class():
        if name == "OpenAIBackend":
            from .openai_backend import OpenAIBackend
            return OpenAIBackend
        elif name == "AnthropicBackend":
            from .anthropic_backend import AnthropicBackend
            return AnthropicBackend
        elif name == "OllamaBackend":
            from .ollama_backend import OllamaBackend
            return OllamaBackend
        elif name == "GeminiBackend":
            from .gemini_backend import GeminiBackend
            return GeminiBackend
        elif name == "LiteLLMBackend":
            from .litellm_backend import LiteLLMBackend
            return LiteLLMBackend
        raise ImportError(f"Unknown backend: {name}")
    return _get_class


class _LazyBackend:
    """Proxy that resolves to the real backend class on first call."""

    def __init__(self, loader):
        self._loader = loader
        self._cls = None

    def __call__(self, *args, **kwargs):
        if self._cls is None:
            self._cls = self._loader()
        return self._cls(*args, **kwargs)

    def __getattr__(self, item):
        if self._cls is None:
            self._cls = self._loader()
        return getattr(self._cls, item)


OpenAIBackend = _LazyBackend(_lazy_import("OpenAIBackend"))
AnthropicBackend = _LazyBackend(_lazy_import("AnthropicBackend"))
OllamaBackend = _LazyBackend(_lazy_import("OllamaBackend"))
GeminiBackend = _LazyBackend(_lazy_import("GeminiBackend"))
LiteLLMBackend = _LazyBackend(_lazy_import("LiteLLMBackend"))

__all__ = [
    "BaseBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "GeminiBackend",
    "LiteLLMBackend",
]
