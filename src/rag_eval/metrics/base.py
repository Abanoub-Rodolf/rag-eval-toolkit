"""Base metric interface for the RAG Eval Toolkit."""
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Matches the first numeric token in [0, 1] — handles "Score: 0.8", "0.9 (very good)", etc.
_SCORE_RE = re.compile(r'\b(1(?:\.0+)?|0(?:\.\d+)?)\b')


def _parse_score(text: str) -> float:
    """Extract the first [0, 1] float from an LLM response string."""
    m = _SCORE_RE.search(text.strip())
    if m:
        return float(m.group(1))
    return 0.0


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics.

    Subclasses must implement :meth:`score`. LLM-judge subclasses should
    call :meth:`_generate_score` rather than parsing backend responses manually.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Compute a quality score for a single RAG sample.

        Args:
            row: Dict with at minimum ``question``, ``context``, and ``answer``.
            backend: LLM backend exposing a ``generate(prompt) -> str`` method.

        Returns:
            Float in [0.0, 1.0]. Higher is better.
        """
        ...

    def _generate_score(self, backend: Any, prompt: str) -> float:
        """Call backend, robustly parse the response as a [0, 1] score."""
        try:
            response = backend.generate(prompt)
            score = _parse_score(response)
            if score == 0.0 and _SCORE_RE.search(response.strip()) is None:
                logger.warning(
                    "%s: could not extract a score from response: %r",
                    self.__class__.__name__,
                    response[:80],
                )
            return score
        except Exception as exc:
            logger.error("%s: unexpected error during scoring: %s", self.__class__.__name__, exc)
            return 0.0
