"""Groundedness metric -- measures if the answer is grounded in the context (NLI-style)."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class GroundednessMetric(BaseMetric):
    """Evaluates the groundedness of the generated answer using Natural Language Inference.
    
    A score of 1.0 means the answer is fully entailed by the context.
    A score of 0.0 means the answer is not supported or contradicts the context.
    """

    def __init__(self) -> None:
        """Initialise GroundednessMetric."""
        super().__init__(name="groundedness")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for groundedness.
        
        Args:
            row: Must contain 'context' and 'answer' keys.
            backend: LLM backend instance.
            
        Returns:
            Float in [0.0, 1.0].
        """
        context = row.get("context")
        answer = row.get("answer")

        prompt = f"""
Evaluate the groundedness of the following Answer against the Context.
Groundedness means that every claim in the Answer is supported by the Context.

Context: {context}
Answer: {answer}

Task:
1. Break down the Answer into individual claims.
2. For each claim, check if it is explicitly stated in or directly inferred from the Context.
3. Provide a groundedness score from 0.0 to 1.0.
   - 1.0: Every claim in the Answer is fully supported by the Context.
   - 0.0: The Answer is not supported by the Context at all.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("GroundednessMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("GroundednessMetric: unexpected error during scoring: %s", exc)
            return 0.0
