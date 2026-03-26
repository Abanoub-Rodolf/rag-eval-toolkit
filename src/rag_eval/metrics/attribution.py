"""Chunk attribution metric -- measures if the answer accurately cites context chunks."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class ChunkAttributionMetric(BaseMetric):
    """Evaluates if the answer correctly attributes information to specific context chunks.
    
    A score of 1.0 means all claims are correctly attributed to chunks.
    A score of 0.0 means claims are misattributed or not attributed where required.
    """

    def __init__(self) -> None:
        """Initialise ChunkAttributionMetric."""
        super().__init__(name="chunk_attribution")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for chunk attribution.
        
        Args:
            row: Must contain 'context' (list or string) and 'answer' keys.
            backend: LLM backend instance.
            
        Returns:
            Float in [0.0, 1.0].
        """
        context = row.get("context")
        answer = row.get("answer")

        prompt = f"""
Evaluate the attribution of claims in the Answer to the provided Context chunks.

Context Chunks: {context}
Answer: {answer}

Task:
1. Identify claims in the Answer that require citation from the Context.
2. Verify if the Answer correctly attributes these claims to the relevant chunks in the Context.
3. Provide an attribution score from 0.0 to 1.0.
   - 1.0: All claims requiring citation are correctly and accurately attributed to the source chunks.
   - 0.0: Many claims are misattributed, or attribution is missing entirely where needed.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("ChunkAttributionMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("ChunkAttributionMetric: unexpected error during scoring: %s", exc)
            return 0.0
