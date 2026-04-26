"""Context utilization metric -- measures how much retrieved context was used in the answer."""
from typing import Any

from .base import BaseMetric


class ContextUtilizationMetric(BaseMetric):
    """Evaluates how much of the retrieved context was actually used in the answer.

    A score of 1.0 means the context was effectively utilized.
    A score of 0.0 means the context was ignored or irrelevant to the answer.
    """

    def __init__(self) -> None:
        super().__init__(name="context_utilization")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate how much of the relevant parts of the context were actually "
            "used to generate the answer.\n\n"
            f"<question>{question}</question>\n"
            f"<context>{context}</context>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = most or all relevant context was effectively used\n"
            "0.0 = relevant context was ignored; answer generated without using context"
        )
        return self._generate_score(backend, prompt)
