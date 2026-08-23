"""Answer relevancy metric -- measures how well the answer addresses the question.

Ragas' original definition (Es et al., 2023) generates several candidate
questions from the answer and scores relevancy as embedding similarity
between those and the original question. This implementation uses a direct
LLM judgment instead: one call, no embedding-model dependency, at the cost of
Ragas' noise-averaging across generated questions. For the original method,
compose SemanticSimilarityMetric with your own question-generation step.
"""
from typing import Any

from .base import BaseMetric, _judge_prompt


class AnswerRelevancyMetric(BaseMetric):
    """Evaluates how relevant the generated answer is to the user's question.

    A score of 1.0 means the answer perfectly addresses the question.
    A score of 0.0 means the answer is completely off-topic.
    """

    def __init__(self) -> None:
        super().__init__(name="answer_relevancy")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG answer relevancy",
            instructions=(
                "Judge how directly and completely the answer addresses the "
                "question. An answer that is factually fine but evasive, "
                "off-topic, or answers a different question than the one asked "
                "should score low regardless of its own internal quality."
            ),
            fields={
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            },
            score_meaning=(
                "0.0 = completely irrelevant to the question\n"
                "1.0 = directly and completely answers the question"
            ),
        )
        return self._generate_score(backend, prompt)
