"""Context recall metric -- coverage of ground-truth information in the retrieved context.

Ragas defines context recall as the fraction of claims in the ground-truth
answer that can be attributed to the retrieved context (Es et al., 2023,
arXiv:2309.15217). This implementation decomposes the ground truth into
individual statements and checks each against the context in one judge call,
then scores the fraction attributed. If the judge's response can't be parsed
into per-statement verdicts, it falls back to a single holistic judgment.
"""
from typing import Any

from ..utils.helpers import flatten_context
from .base import _NUMBERED_VERDICT_RE, BaseMetric, ScoreParseError, _judge_prompt


class ContextRecallMetric(BaseMetric):
    """Evaluates how much of the ground-truth answer is covered by the context.

    A score of 1.0 means every ground-truth statement is attributable to the
    retrieved context. A score of 0.0 means none of it is.

    Note:
        Requires a non-empty ``ground_truth`` field; raises ``ValueError``
        without it, since the metric is undefined without a reference answer.
    """

    def __init__(self) -> None:
        super().__init__(name="context_recall")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")
        ground_truth = row.get("ground_truth", "")

        if not ground_truth:
            raise ValueError(
                "ContextRecallMetric requires a non-empty 'ground_truth' field."
            )

        flat_context = flatten_context(context)

        try:
            return self._score_decomposed(question, flat_context, ground_truth, backend)
        except ScoreParseError:
            pass  # judge didn't follow the per-statement format; fall back below

        prompt = _judge_prompt(
            role="RAG retrieval recall",
            instructions=(
                "Judge what fraction of the ground-truth answer's information "
                "is present in the retrieved context."
            ),
            fields={
                "question": question,
                "context": flat_context,
                "ground_truth": ground_truth,
            },
            score_meaning=(
                "0.0 = none of the ground truth is in the context\n"
                "1.0 = all ground truth information is present in the context"
            ),
        )
        return self._generate_score(backend, prompt)

    @staticmethod
    def _score_decomposed(question: str, context: str, ground_truth: str, backend: Any) -> float:
        prompt = (
            "You are an impartial judge evaluating RAG retrieval quality.\n\n"
            "Break the ground-truth answer down into individual factual "
            "statements, numbered starting at 1. For each statement, judge "
            "whether it can be attributed to (found in or safely inferred "
            "from) the retrieved context.\n\n"
            f"<question>{question}</question>\n"
            f"<context>{context}</context>\n"
            f"<ground_truth>{ground_truth}</ground_truth>\n\n"
            "Respond with exactly one line per statement, in this format and "
            "nothing else:\n"
            "<statement number>: <yes or no>"
        )
        response = backend.generate(prompt)
        matches = sorted(_NUMBERED_VERDICT_RE.finditer(response), key=lambda m: int(m.group(1)))
        if not matches:
            raise ScoreParseError(response)
        verdicts = [m.group(2).lower() == "yes" for m in matches]
        return sum(verdicts) / len(verdicts)
