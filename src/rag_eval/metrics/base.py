"""Base metric interface for the RAG Eval Toolkit."""
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Matches any standalone number (int or float) for range checking
_NUM_RE = re.compile(r'\b\d+(?:\.\d+)?\b')


def _parse_score(text: str) -> float:
    """Extract the first number in [0.0, 1.0] from an LLM response string.

    Handles common response formats like "Score: 0.8", "0.9 out of 1.0", etc.
    Returns 0.0 if no valid score is found.
    """
    for m in _NUM_RE.finditer(text.strip()):
        try:
            v = float(m.group())
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
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
            if score == 0.0:
                nums = [float(m.group()) for m in _NUM_RE.finditer(response.strip())]
                if nums and all(v < 0.0 or v > 1.0 for v in nums):
                    logger.warning(
                        "%s: all numbers out of [0,1] range in response: %r",
                        self.__class__.__name__,
                        response[:80],
                    )
                elif not nums:
                    logger.warning(
                        "%s: no score found in response: %r",
                        self.__class__.__name__,
                        response[:80],
                    )
            return score
        except Exception as exc:
            logger.error("%s: unexpected error during scoring: %s", self.__class__.__name__, exc)
            return 0.0
