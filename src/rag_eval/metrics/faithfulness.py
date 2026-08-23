"""Faithfulness metric -- measures whether the answer is grounded in the context.

Follows Ragas' definition (Es et al., 2023, arXiv:2309.15217): the fraction of
claims in the answer that are supported by the retrieved context. Ragas
computes this with a separate claim-extraction pass before verification; this
implementation asks the judge to do both in one call, trading some precision
for one LLM call per sample instead of two.
"""
from typing import Any

from .base import BaseMetric, _judge_prompt


class FaithfulnessMetric(BaseMetric):
    """Evaluates whether the generated answer is faithful to the retrieved context.

    A score of 1.0 means every claim in the answer is supported by the context.
    A score of 0.0 indicates the answer is entirely unsupported.
    """

    def __init__(self) -> None:
        super().__init__(name="faithfulness")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG output faithfulness",
            instructions=(
                "Break the answer down into its individual factual claims. For "
                "each claim, check whether it is directly stated in or can be "
                "reasonably inferred from the context. A claim that relies on "
                "outside knowledge not present in the context counts against "
                "faithfulness, even if the claim happens to be true."
            ),
            fields={
                "context": row.get("context", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            },
            score_meaning=(
                "0.0 = no claims are supported by the context\n"
                "1.0 = every claim is supported by the context"
            ),
        )
        return self._generate_score(backend, prompt)
