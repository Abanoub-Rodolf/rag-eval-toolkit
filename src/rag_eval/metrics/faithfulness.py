"""Faithfulness metric -- measures whether the answer is grounded in the context."""
from typing import Any

from .base import BaseMetric


class FaithfulnessMetric(BaseMetric):
    """Evaluates whether the generated answer is faithful to the retrieved context.

    A score of 1.0 means every claim in the answer is supported by the context.
    A score of 0.0 indicates the answer is entirely unsupported.
    """

    def __init__(self) -> None:
        super().__init__(name="faithfulness")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate whether the answer is faithful to the context. Every claim "
            "must be traceable back to the context with no hallucinated additions.\n\n"
            f"<context>{context}</context>\n"
            f"<question>{question}</question>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "0.0 = completely unfaithful / hallucinated\n"
            "1.0 = completely faithful"
        )
        return self._generate_score(backend, prompt)
