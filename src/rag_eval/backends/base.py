from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """Abstract base for all LLM backends.

    Subclasses must set ``self.model`` in ``__init__`` and implement ``generate``.
    Optionally implement ``embed`` to support SemanticSimilarityMetric.
    """

    model: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return a text response for the given prompt."""
        ...
