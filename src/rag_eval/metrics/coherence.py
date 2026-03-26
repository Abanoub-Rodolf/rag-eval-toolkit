"""Coherence metric -- measures if the answer is logically structured and clear."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class CoherenceMetric(BaseMetric):
    """Evaluates the logical structure and clarity of the generated answer.
    
    A score of 1.0 means the answer is perfectly coherent and clear.
    A score of 0.0 means the answer is disjointed, contradictory, or unintelligible.
    """

    def __init__(self) -> None:
        """Initialise CoherenceMetric."""
        super().__init__(name="coherence")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for coherence.
        
        Args:
            row: Must contain 'answer' key.
            backend: LLM backend instance.
            
        Returns:
            Float in [0.0, 1.0].
        """
        answer = row.get("answer")

        prompt = f"""
Evaluate the coherence and logical structure of the following Answer.

Answer: {answer}

Task:
1. Assess if the sentences follow each other logically.
2. Check if there are any internal contradictions.
3. Provide a coherence score from 0.0 to 1.0.
   - 1.0: Perfectly coherent, logically structured, and easy to read.
   - 0.0: Unintelligible, contradictory, or completely disjointed.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("CoherenceMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("CoherenceMetric: unexpected error during scoring: %s", exc)
            return 0.0
