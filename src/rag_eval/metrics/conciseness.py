"""Conciseness metric -- measures if the answer is direct and free of fluff."""
from typing import Any

from .base import BaseMetric


class ConcisenessMetric(BaseMetric):
    """Evaluates if the answer is concise and avoids unnecessary verbosity.

    A score of 1.0 means perfectly concise and direct.
    A score of 0.0 means extremely verbose or filled with irrelevant content.
    """

    def __init__(self) -> None:
        super().__init__(name="conciseness")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        answer = row.get("answer", "")

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate if the answer provides necessary information without "
            "unnecessary words, repetition, or filler content.\n\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = perfectly concise and direct\n"
            "0.0 = extremely wordy, repetitive, or filled with irrelevant content"
        )
        return self._generate_score(backend, prompt)
