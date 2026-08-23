"""Groundedness metric -- NLI-style entailment check between context and answer.

Same underlying question as faithfulness, framed with the Natural Language
Inference terminology used by TruLens' RAG Triad and Azure AI Foundry's
evaluators. Kept as a separate metric so users coming from those tools find a
familiar name and framing.
"""
from typing import Any

from .base import BaseMetric, _judge_prompt


class GroundednessMetric(BaseMetric):
    """Evaluates the groundedness of the generated answer using Natural Language Inference.

    A score of 1.0 means the answer is fully entailed by the context.
    A score of 0.0 means the answer is contradicted by or unsupported by the context.
    """

    def __init__(self) -> None:
        super().__init__(name="groundedness")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        prompt = _judge_prompt(
            role="RAG output groundedness (NLI entailment)",
            instructions=(
                "Treat the context as the premise and the answer as the "
                "hypothesis. Break the answer into individual claims and check "
                "whether each is entailed by the context (directly stated or a "
                "safe logical inference), contradicted by it, or unsupported "
                "(neither entailed nor contradicted)."
            ),
            fields={
                "context": row.get("context", ""),
                "answer": row.get("answer", ""),
            },
            score_meaning=(
                "0.0 = answer is contradicted by or unsupported by the context\n"
                "1.0 = every claim is entailed by the context"
            ),
        )
        return self._generate_score(backend, prompt)
