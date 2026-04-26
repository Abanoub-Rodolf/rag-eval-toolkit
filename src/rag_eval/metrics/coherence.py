"""Coherence metric -- measures if the answer is logically structured and clear."""
from typing import Any

from .base import BaseMetric


class CoherenceMetric(BaseMetric):
    """Evaluates the logical structure and clarity of the generated answer.

    A score of 1.0 means perfectly coherent and clear.
    A score of 0.0 means disjointed, contradictory, or unintelligible.
    """

    def __init__(self) -> None:
        super().__init__(name="coherence")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate the coherence and logical structure of the answer: do sentences "
            "follow logically, is there internal consistency, is it easy to read?\n\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = perfectly coherent, logically structured\n"
            "0.0 = unintelligible, contradictory, or completely disjointed"
        )
        return self._generate_score(backend, prompt)
