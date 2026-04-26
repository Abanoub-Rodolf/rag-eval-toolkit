"""Semantic similarity metric -- measures the similarity between two strings using embeddings."""
import logging
import numpy as np
from typing import Any, Dict, List

from .base import BaseMetric

logger = logging.getLogger(__name__)


class SemanticSimilarityMetric(BaseMetric):
    """Evaluates the semantic similarity between the answer and the ground truth.
    
    This metric does not use an LLM-as-judge; instead, it compares embeddings.
    """

    def __init__(self) -> None:
        super().__init__(name="semantic_similarity")

    def _get_embedding(self, text: str, backend: Any) -> np.ndarray:
        """Get embedding for a text string.

        Requires the backend to expose an ``embed(text) -> list[float]`` method.
        Returns an empty array when embedding is unavailable so cosine similarity
        will return 0.0 via the norm-zero guard in ``_cosine_similarity``.
        """
        if hasattr(backend, "embed"):
            try:
                return np.array(backend.embed(text))
            except Exception as e:
                logger.error("Failed to get embedding: %s", e)
        else:
            logger.warning(
                "SemanticSimilarityMetric: backend does not support embed(). Returning 0.0."
            )
        return np.zeros(0)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        answer = row.get("answer", "")
        ground_truth = row.get("ground_truth", "")

        if not ground_truth:
            logger.warning("SemanticSimilarityMetric: ground_truth is missing. Returning 0.0.")
            return 0.0

        emb_answer = self._get_embedding(answer, backend)
        emb_gt = self._get_embedding(ground_truth, backend)

        return self._cosine_similarity(emb_answer, emb_gt)
