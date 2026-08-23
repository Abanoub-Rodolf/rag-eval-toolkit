"""Chunk attribution metric -- measures if the answer accurately cites context chunks."""
from typing import Any

from ..utils.helpers import flatten_context
from .base import BaseMetric, _judge_prompt


class ChunkAttributionMetric(BaseMetric):
    """Evaluates if the answer correctly attributes information to specific context chunks.

    A score of 1.0 means all claims are correctly attributed to source chunks.
    A score of 0.0 means claims are misattributed or not attributed where required.
    """

    def __init__(self) -> None:
        super().__init__(name="chunk_attribution")

    def score(self, row: dict[str, Any], backend: Any) -> float:
        context = flatten_context(row.get("context", ""))

        prompt = _judge_prompt(
            role="RAG citation attribution",
            instructions=(
                "Judge how accurately the answer attributes its claims to the "
                "source context chunks provided. Claims requiring citation "
                "should reference the chunk that actually supports them, not "
                "just any chunk."
            ),
            fields={"context_chunks": context, "answer": row.get("answer", "")},
            score_meaning=(
                "0.0 = claims are misattributed or attribution is missing entirely\n"
                "1.0 = all claims are correctly and accurately attributed to source chunks"
            ),
        )
        return self._generate_score(backend, prompt)
