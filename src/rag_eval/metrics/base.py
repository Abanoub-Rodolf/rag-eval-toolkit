"""Base metric interface and shared LLM-judge prompt/parsing utilities.

Every built-in judge prompt asks the model to reason briefly before giving a
labeled score (``SCORE: 0.8``) rather than emitting a bare number. Chain-of-
thought before the verdict is the single best-documented lever for LLM-judge
reliability (see G-Eval, and Ragas/DeepEval's own internal use of reasoning
traces); a bare "output a float" prompt is the weakest version of this
pattern. Parsing prefers the labeled score but falls back to scanning for any
number in [0.0, 1.0], so judges (and tests) that ignore the label still work.
"""
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r'\b\d+(?:\.\d+)?\b')
_LABELED_SCORE_RE = re.compile(r'(?:FINAL_SCORE|SCORE)\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
# Shared by ranked/decomposed judges (context_precision, context_recall):
# lines like "1: yes", "2) no", "3. YES"
_NUMBERED_VERDICT_RE = re.compile(r'^\s*(\d+)\s*[:.)]\s*(yes|no)\b', re.IGNORECASE | re.MULTILINE)


class ScoreParseError(RuntimeError):
    """Raised when a judge response has no parseable score in [0.0, 1.0].

    This is distinct from a legitimate 0.0 verdict. A backend that returns an
    empty string, a refusal, or truncated output should not silently count as
    "completely unfaithful" or "highly toxic" -- that corrupts the average
    with no visible signal. RAGEvaluator catches this, excludes the sample
    from the metric's average, and reports it under ``results["errors"]``.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        preview = response.strip()[:200]
        super().__init__(f"no score in [0.0, 1.0] found in judge response: {preview!r}")


def _parse_score(text: str) -> Optional[float]:
    """Extract a score in [0.0, 1.0] from an LLM judge response.

    Prefers an explicit ``SCORE: 0.8`` / ``FINAL_SCORE: 0.8`` label. Falls
    back to the first standalone number in range, so responses that skip the
    label ("0.9", "0.9 out of 1.0") still parse.

    Returns:
        The score, or None if nothing in [0.0, 1.0] could be found -- callers
        must treat that as a failed judgment, not a score of zero.
    """
    stripped = text.strip()

    labeled = _LABELED_SCORE_RE.search(stripped)
    if labeled:
        v = float(labeled.group(1))
        if 0.0 <= v <= 1.0:
            return v

    for m in _NUM_RE.finditer(stripped):
        v = float(m.group())
        if 0.0 <= v <= 1.0:
            return v
    return None


def _judge_prompt(role: str, instructions: str, fields: dict[str, Any], score_meaning: str) -> str:
    """Build a standard LLM-judge prompt: role, task, XML-delimited inputs, and
    a request for brief reasoning followed by a labeled score.

    Args:
        role: What's being judged, e.g. "RAG output faithfulness".
        instructions: Task-specific evaluation instructions.
        fields: Ordered ``tag_name -> value`` pairs rendered as ``<tag>value</tag>``.
        score_meaning: What 0.0 and 1.0 mean for this metric.
    """
    tags = "\n".join(f"<{name}>{value}</{name}>" for name, value in fields.items())
    return (
        f"You are an impartial judge evaluating {role}.\n\n"
        f"{instructions}\n\n"
        f"{tags}\n\n"
        "Reason briefly (1-2 sentences), then give your verdict on the last "
        "line in exactly this format: SCORE: <float between 0.0 and 1.0>\n\n"
        f"{score_meaning}"
    )


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics.

    Subclasses must implement :meth:`score`. LLM-judge subclasses should
    build prompts with :func:`_judge_prompt` and call :meth:`_generate_score`
    rather than parsing backend responses manually.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def score(self, row: dict[str, Any], backend: Any) -> float:
        """Compute a quality score for a single RAG sample.

        Args:
            row: Dict with at minimum ``question``, ``context``, and ``answer``.
            backend: LLM backend exposing a ``generate(prompt) -> str`` method.

        Returns:
            Float in [0.0, 1.0]. Higher is better.

        Raises:
            ScoreParseError: if the judge's response could not be parsed.
            ValueError: if a field the metric requires (e.g. ``ground_truth``)
                is missing.
        """
        ...

    def _generate_score(self, backend: Any, prompt: str) -> float:
        """Call the backend and parse its response as a score in [0.0, 1.0].

        Backend errors (network, auth, rate limit) propagate as-is. A
        response that doesn't contain a parseable score raises
        :class:`ScoreParseError` rather than being silently treated as 0.0.
        """
        response = backend.generate(prompt)
        score = _parse_score(response)
        if score is None:
            logger.warning(
                "%s: no score found in judge response: %r",
                self.__class__.__name__,
                response[:80],
            )
            raise ScoreParseError(response)
        return score
