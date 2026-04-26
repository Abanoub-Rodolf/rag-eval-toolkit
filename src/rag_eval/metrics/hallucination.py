"""Hallucination metric -- measures if the answer contains information not in the context."""
from typing import Any

from .base import BaseMetric


class HallucinationMetric(BaseMetric):
    """Evaluates if the generated answer contains hallucinations.

    A score of 1.0 means the answer is completely grounded (no hallucinations).
    A score of 0.0 means the answer is entirely hallucinated.
    """

    def __init__(self) -> None:
        super().__init__(name="hallucination")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Determine if the answer contains claims NOT supported by the context.\n\n"
            f"<context>{context}</context>\n"
            f"<question>{question}</question>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = no hallucinations; every claim is supported by the context\n"
            "0.0 = entirely hallucinated or contradicts the context"
        )
        return self._generate_score(backend, prompt)
