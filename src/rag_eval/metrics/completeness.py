"""Answer completeness metric -- measures if the answer addresses all parts of the question."""
from typing import Any

from .base import BaseMetric, _judge_prompt


class AnswerCompletenessMetric(BaseMetric):
    """Evaluates if the answer addresses all parts and nuances of the question.

    A score of 1.0 means the answer is complete.
    A score of 0.0 means the answer is missing critical information.
    """

    def __init__(self) -> None:
        super().__init__(name="completeness")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG answer completeness",
            instructions=(
                "Identify every distinct component or sub-question in the "
                "question, then check whether the answer addresses each one. "
                "A fluent answer that fully covers one part of a multi-part "
                "question but ignores the rest is incomplete, not complete."
            ),
            fields={
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            },
            score_meaning=(
                "0.0 = fails to address the question or misses almost all parts\n"
                "1.0 = all parts of the question are fully addressed"
            ),
        )
        return self._generate_score(backend, prompt)
