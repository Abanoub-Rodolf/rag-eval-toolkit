"""Conciseness metric -- measures if the answer is direct and free of filler."""
from typing import Any

from .base import BaseMetric, _judge_prompt


class ConcisenessMetric(BaseMetric):
    """Evaluates if the answer is concise and avoids unnecessary verbosity.

    A score of 1.0 means perfectly concise and direct.
    A score of 0.0 means extremely verbose or filled with irrelevant content.
    """

    def __init__(self) -> None:
        super().__init__(name="conciseness")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG answer conciseness",
            instructions=(
                "Judge whether the answer conveys its information without "
                "unnecessary words, hedging, repetition, or filler. A longer "
                "answer is not automatically less concise if every sentence "
                "carries information; score against padding, not length."
            ),
            fields={"answer": row.get("answer", "")},
            score_meaning=(
                "0.0 = extremely wordy, repetitive, or filled with irrelevant content\n"
                "1.0 = perfectly concise, no filler"
            ),
        )
        return self._generate_score(backend, prompt)
