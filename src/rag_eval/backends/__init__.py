from .base import BaseBackend


class _LazyMeta(type):
    """Metaclass so isinstance(backend, OpenAIBackend) resolves against the real class."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, cls._resolve())

    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, cls._resolve())


def _lazy_backend(name: str, module: str, cls_name: str) -> type:
    """Return a lazy-loading class proxy that defers the import until first call."""

    def _resolve():
        if _cls_cache[0] is None:
            import importlib
            mod = importlib.import_module(f".{module}", package="rag_eval.backends")
            _cls_cache[0] = getattr(mod, cls_name)
        return _cls_cache[0]

    _cls_cache: list = [None]

    proxy = _LazyMeta(
        name,
        (),
        {
            "_resolve": staticmethod(_resolve),
            "__new__": lambda cls, *a, **kw: _resolve()(*a, **kw),
            "__doc__": f"Lazy proxy for {cls_name}. Import deferred until first instantiation.",
        },
    )
    return proxy


OpenAIBackend = _lazy_backend("OpenAIBackend", "openai_backend", "OpenAIBackend")
AnthropicBackend = _lazy_backend("AnthropicBackend", "anthropic_backend", "AnthropicBackend")
OllamaBackend = _lazy_backend("OllamaBackend", "ollama_backend", "OllamaBackend")
GeminiBackend = _lazy_backend("GeminiBackend", "gemini_backend", "GeminiBackend")
LiteLLMBackend = _lazy_backend("LiteLLMBackend", "litellm_backend", "LiteLLMBackend")

__all__ = [
    "BaseBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "GeminiBackend",
    "LiteLLMBackend",
]
