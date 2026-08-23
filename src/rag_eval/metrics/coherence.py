"""Coherence metric -- measures if the answer is logically structured and clear."""
from typing import Any

from .base import BaseMetric, _judge_prompt


class CoherenceMetric(BaseMetric):
    """Evaluates the logical structure and clarity of the generated answer.

    A score of 1.0 means perfectly coherent and clear.
    A score of 0.0 means disjointed, contradictory, or unintelligible.
    """

    def __init__(self) -> None:
        super().__init__(name="coherence")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG answer coherence",
            instructions=(
                "Judge whether the answer's sentences follow logically from "
                "each other, whether it is internally consistent, and whether "
                "it reads clearly. Judge structure and clarity only -- a short, "
                "well-organized answer should not be marked down for brevity, "
                "and a long answer should not be scored higher just for length."
            ),
            fields={"answer": row.get("answer", "")},
            score_meaning=(
                "0.0 = unintelligible, contradictory, or completely disjointed\n"
                "1.0 = perfectly coherent and logically structured"
            ),
        )
        return self._generate_score(backend, prompt)
