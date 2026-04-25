"""Groundedness metric -- measures if the answer is grounded in the context (NLI-style)."""
from typing import Any, Dict

from .base import BaseMetric


class GroundednessMetric(BaseMetric):
    """Evaluates the groundedness of the generated answer using Natural Language Inference.

    A score of 1.0 means the answer is fully entailed by the context.
    A score of 0.0 means the answer is not supported or contradicts the context.
    """

    def __init__(self) -> None:
        super().__init__(name="groundedness")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        context = row.get("context", "")
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate groundedness: break the answer into individual claims and "
            "check each one is explicitly stated in or directly inferable from "
            "the context.\n\n"
            f"<context>{context}</context>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = every claim fully supported by the context\n"
            "0.0 = answer not supported by the context at all"
        )
        return self._generate_score(backend, prompt)
