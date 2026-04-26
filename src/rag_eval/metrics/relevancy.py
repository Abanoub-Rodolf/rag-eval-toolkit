"""Answer relevancy metric -- measures how well the answer addresses the question."""
from typing import Any

from .base import BaseMetric


class AnswerRelevancyMetric(BaseMetric):
    """Evaluates how relevant the generated answer is to the user's question.

    A score of 1.0 means the answer perfectly addresses the question.
    A score of 0.0 means the answer is completely off-topic.
    """

    def __init__(self) -> None:
        super().__init__(name="answer_relevancy")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate how directly and completely the answer addresses the question.\n\n"
            f"<question>{question}</question>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "0.0 = completely irrelevant\n"
            "1.0 = perfectly answers the question"
        )
        return self._generate_score(backend, prompt)
