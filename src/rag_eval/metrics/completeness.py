"""Answer completeness metric -- measures if the answer addresses all parts of the question."""
from typing import Any, Dict

from .base import BaseMetric


class AnswerCompletenessMetric(BaseMetric):
    """Evaluates if the answer addresses all parts and nuances of the question.

    A score of 1.0 means the answer is complete.
    A score of 0.0 means the answer is missing critical information.
    """

    def __init__(self) -> None:
        super().__init__(name="completeness")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate the completeness of the answer: does it address all components "
            "and sub-questions contained in the question?\n\n"
            f"<question>{question}</question>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = all parts of the question fully addressed\n"
            "0.0 = answer fails to address the question or misses almost all parts"
        )
        return self._generate_score(backend, prompt)
