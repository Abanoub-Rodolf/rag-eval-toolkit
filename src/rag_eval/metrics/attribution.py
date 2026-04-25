"""Chunk attribution metric -- measures if the answer accurately cites context chunks."""
from typing import Any, Dict, Union

from .base import BaseMetric


class ChunkAttributionMetric(BaseMetric):
    """Evaluates if the answer correctly attributes information to specific context chunks.

    A score of 1.0 means all claims are correctly attributed to source chunks.
    A score of 0.0 means claims are misattributed or not attributed where required.
    """

    def __init__(self) -> None:
        super().__init__(name="chunk_attribution")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        context = row.get("context", "")
        answer = row.get("answer", "")

        # normalise list context to a single string for consistent prompt/cache behaviour
        if isinstance(context, list):
            context = "\n---\n".join(str(c) for c in context)

        prompt = (
            "You are an impartial judge evaluating RAG output quality.\n\n"
            "Evaluate how accurately the answer attributes its claims to the source "
            "context chunks provided. Claims requiring citation should reference the "
            "correct chunk.\n\n"
            f"<context_chunks>{context}</context_chunks>\n"
            f"<answer>{answer}</answer>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "1.0 = all claims correctly and accurately attributed to source chunks\n"
            "0.0 = many claims misattributed or attribution missing entirely"
        )
        return self._generate_score(backend, prompt)
