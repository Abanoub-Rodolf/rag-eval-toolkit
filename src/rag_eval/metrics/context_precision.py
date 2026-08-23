"""Context precision metric -- signal-to-noise ratio of retrieved context, rank-aware.

Ragas defines context precision as an average-precision computation over
ranked chunks (Es et al., 2023, arXiv:2309.15217): judge each chunk's
relevance in retrieval order, then compute precision at each rank and average
it over the relevant chunks. A retriever that surfaces the right chunk first
scores higher than one that buries it at position 5, even if both retrieve it
somewhere -- rank matters, not just presence.

This implementation follows that definition when ``context`` is a list of
chunks in retrieval order. A plain string has no chunk boundaries or rank
information, so rank-aware precision isn't computable; in that case this
falls back to one holistic judgment of how much of the context is relevant
versus noise. The ranked path also falls back to the holistic judgment if the
judge's per-chunk response can't be parsed.
"""
from typing import Any, Optional

from ..utils.helpers import flatten_context
from .base import _NUMBERED_VERDICT_RE, BaseMetric, ScoreParseError, _judge_prompt


class ContextPrecisionMetric(BaseMetric):
    """Evaluates how precisely the retrieved context matches the information need.

    A score of 1.0 means every retrieved chunk is relevant and relevant chunks
    are ranked first. A score of 0.0 means no retrieved chunk is relevant.
    """

    def __init__(self) -> None:
        super().__init__(name="context_precision")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")

        if isinstance(context, list) and len(context) > 1:
            try:
                return self._score_ranked(question, context, backend)
            except ScoreParseError:
                pass  # judge didn't follow the per-chunk format; fall back below

        flat_context = flatten_context(context)
        prompt = _judge_prompt(
            role="RAG retrieval precision",
            instructions=(
                "Judge how much of the retrieved context is relevant to "
                "answering the question versus irrelevant noise. High "
                "precision means most of the context is on-topic; low "
                "precision means it is padded with unrelated material."
            ),
            fields={"question": question, "context": flat_context},
            score_meaning=(
                "0.0 = no part of the context is relevant\n"
                "1.0 = the context is entirely relevant, no noise"
            ),
        )
        return self._generate_score(backend, prompt)

    def _score_ranked(self, question: str, chunks: list, backend: Any) -> float:
        """Ragas-style average precision over chunks in retrieval rank order."""
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))
        prompt = (
            "You are an impartial judge evaluating RAG retrieval quality.\n\n"
            "Below are context chunks in retrieval rank order (chunk 1 was "
            "retrieved first). For each chunk, judge whether it is relevant to "
            "answering the question.\n\n"
            f"<question>{question}</question>\n"
            f"<chunks>\n{numbered}\n</chunks>\n\n"
            "Respond with exactly one line per chunk, in order, in this "
            "format and nothing else:\n"
            "<chunk number>: <yes or no>"
        )
        response = backend.generate(prompt)
        verdicts = self._parse_verdicts(response, len(chunks))
        if verdicts is None:
            raise ScoreParseError(response)
        return self._average_precision(verdicts)

    @staticmethod
    def _parse_verdicts(response: str, n: int) -> Optional[list]:
        found: dict[int, bool] = {}
        for m in _NUMBERED_VERDICT_RE.finditer(response):
            idx = int(m.group(1))
            if 1 <= idx <= n:
                found[idx] = m.group(2).lower() == "yes"
        if len(found) < n:
            return None
        return [found[i] for i in range(1, n + 1)]

    @staticmethod
    def _average_precision(verdicts: list) -> float:
        relevant_seen = 0
        precision_sum = 0.0
        for k, is_relevant in enumerate(verdicts, start=1):
            if is_relevant:
                relevant_seen += 1
                precision_sum += relevant_seen / k
        if relevant_seen == 0:
            return 0.0
        return precision_sum / relevant_seen
