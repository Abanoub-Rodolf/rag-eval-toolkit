"""Context precision metric -- measures signal-to-noise ratio of retrieved context."""
from typing import Any

from .base import BaseMetric


class ContextPrecisionMetric(BaseMetric):
    """Evaluates how precisely the retrieved context matches the information need.

    A score of 1.0 means the context contains exactly the information needed.
    A score of 0.0 means the context is entirely useless for answering.
    """

    def __init__(self) -> None:
        super().__init__(name="context_precision")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")

        prompt = (
            "You are an impartial judge evaluating RAG retrieval quality.\n\n"
            "Evaluate how useful the retrieved context is for answering the question. "
            "High precision means most of the context is relevant with little noise.\n\n"
            f"<question>{question}</question>\n"
            f"<context>{context}</context>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "0.0 = completely useless context\n"
            "1.0 = context contains exactly the answer with no irrelevant content"
        )
        return self._generate_score(backend, prompt)
