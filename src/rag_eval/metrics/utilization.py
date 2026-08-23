"""Context utilization metric -- measures how much retrieved context was used in the answer."""
from typing import Any

from .base import BaseMetric, _judge_prompt


class ContextUtilizationMetric(BaseMetric):
    """Evaluates how much of the retrieved context was actually used in the answer.

    A score of 1.0 means the context was effectively used.
    A score of 0.0 means the context was ignored or irrelevant to the answer.
    """

    def __init__(self) -> None:
        super().__init__(name="context_utilization")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG context utilization",
            instructions=(
                "Judge how much of the relevant parts of the context were "
                "actually used to produce the answer, versus the answer being "
                "generated with little regard for what was retrieved."
            ),
            fields={
                "question": row.get("question", ""),
                "context": row.get("context", ""),
                "answer": row.get("answer", ""),
            },
            score_meaning=(
                "0.0 = relevant context was ignored; answer generated without using it\n"
                "1.0 = most or all relevant context was effectively used"
            ),
        )
        return self._generate_score(backend, prompt)
