"""Hallucination metric -- flags claims that are unsupported by or contradict the context.

Answers the same underlying question as faithfulness, under the name most
CI/guardrail tools (DeepEval, Galileo) use. Scored with a severity lens
instead of a fraction: a single fabricated or contradicting claim should pull
the score down sharply, since one confident lie is worse for downstream trust
than a proportional average suggests.
"""
from typing import Any

from .base import BaseMetric, _judge_prompt


class HallucinationMetric(BaseMetric):
    """Evaluates if the generated answer contains hallucinations.

    A score of 1.0 means the answer is completely grounded (no hallucinations).
    A score of 0.0 means the answer contains a fabricated or contradicting claim.
    """

    def __init__(self) -> None:
        super().__init__(name="hallucination")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG output hallucination risk",
            instructions=(
                "Identify any claim in the answer that is not supported by the "
                "context, or that contradicts it. Weigh a single fabricated or "
                "contradicting claim heavily -- it is a serious hallucination "
                "even if the rest of the answer is accurate, not just a small "
                "deduction proportional to its share of the answer."
            ),
            fields={
                "context": row.get("context", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            },
            score_meaning=(
                "0.0 = contains a fabricated or contradicting claim\n"
                "1.0 = every claim is directly supported by the context"
            ),
        )
        return self._generate_score(backend, prompt)
