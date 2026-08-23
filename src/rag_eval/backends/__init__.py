"""Lazy-loading backend namespace.

Backend modules are imported only when their class attribute is first
accessed (PEP 562 module ``__getattr__``), so ``import rag_eval.backends``
stays cheap. The heavy SDK imports themselves are additionally guarded
inside each backend module, so importing a backend module is safe even
when its SDK is not installed.
"""
import importlib

from .base import BaseBackend

_BACKEND_MODULES = {
    "OpenAIBackend": "openai_backend",
    "AnthropicBackend": "anthropic_backend",
    "OllamaBackend": "ollama_backend",
    "GeminiBackend": "gemini_backend",
    "LiteLLMBackend": "litellm_backend",
}

__all__ = [
    "BaseBackend",
    *_BACKEND_MODULES.keys(),
]


def __getattr__(name: str):
    module_name = _BACKEND_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", package="rag_eval.backends")
    return getattr(module, name)


def __dir__() -> list:
    return sorted({*__all__, __getattr__.__name__, __dir__.__name__})
