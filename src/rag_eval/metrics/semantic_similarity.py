"""Semantic similarity metric -- embedding cosine similarity between answer and ground truth."""
from typing import Any

import numpy as np

from .base import BaseMetric


class BackendCapabilityError(RuntimeError):
    """Raised when a metric needs a backend capability the configured backend lacks."""


class SemanticSimilarityMetric(BaseMetric):
    """Evaluates the semantic similarity between the answer and the ground truth.

    Does not use an LLM judge; requires the backend to implement
    ``embed(text) -> list[float]``. OpenAIBackend and AnthropicBackend only
    implement chat completion, not embeddings -- use OllamaBackend,
    GeminiBackend, or LiteLLMBackend (with an embedding-capable model)
    instead.
    """

    def __init__(self) -> None:
        super().__init__(name="semantic_similarity")

    def _get_embedding(self, text: str, backend: Any) -> np.ndarray:
        if not hasattr(backend, "embed"):
            raise BackendCapabilityError(
                f"{type(backend).__name__} does not implement embed(). "
                "SemanticSimilarityMetric needs an embedding-capable backend "
                "(OllamaBackend, GeminiBackend, or LiteLLMBackend)."
            )
        embedding = np.array(backend.embed(text))
        if embedding.size == 0:
            # Some backends return [] on embed errors; treat that as a failed
            # judgment instead of a silent 0.0 cosine score (same policy as
            # ScoreParseError for judge text).
            raise BackendCapabilityError(
                f"{type(backend).__name__}.embed() returned an empty vector "
                "(likely an embedding API error); sample excluded from average."
            )
        return embedding

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def score(self, row: dict[str, Any], backend: Any) -> float:
        answer = row.get("answer", "")
        ground_truth = row.get("ground_truth", "")
        if not ground_truth:
            raise ValueError(
                "SemanticSimilarityMetric requires a non-empty 'ground_truth' field."
            )

        emb_answer = self._get_embedding(answer, backend)
        emb_gt = self._get_embedding(ground_truth, backend)
        return self._cosine_similarity(emb_answer, emb_gt)
