from abc import ABCMeta

from .base import BaseBackend


class _LazyMeta(ABCMeta):
    """Metaclass so isinstance/issubclass resolve against the real backend class."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, cls._resolve())

    def __subclasscheck__(cls, subclass):
        if subclass is cls:
            return True
        if isinstance(subclass, _LazyMeta):
            subclass = subclass._resolve()
        return issubclass(subclass, cls._resolve())


def _lazy_backend(name: str, module: str, cls_name: str) -> type:
    """Lazy-loading class proxy. The real backend module imports on first use."""

    def _resolve():
        if _cls_cache[0] is None:
            import importlib
            mod = importlib.import_module(f".{module}", package="rag_eval.backends")
            _cls_cache[0] = getattr(mod, cls_name)
        return _cls_cache[0]

    _cls_cache: list = [None]

    proxy = _LazyMeta(
        name,
        (BaseBackend,),
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
