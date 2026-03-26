"""Hallucination metric -- measures if the answer contains information not in the context."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class HallucinationMetric(BaseMetric):
    """Evaluates if the generated answer contains hallucinations.
    
    A score of 1.0 means the answer is completely grounded in the context (no hallucinations).
    A score of 0.0 means the answer is entirely hallucinated.
    """

    def __init__(self) -> None:
        """Initialise HallucinationMetric."""
        super().__init__(name="hallucination")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for hallucinations.
        
        Args:
            row: Must contain 'question', 'context', and 'answer' keys.
            backend: LLM backend instance.
            
        Returns:
            Float in [0.0, 1.0].
        """
        question = row.get("question")
        context = row.get("context")
        answer = row.get("answer")

        prompt = f"""
Analyze the following Answer against the provided Context and Question.
Determine if the Answer contains any information that is NOT supported by the Context (hallucinations).

Context: {context}
Question: {question}
Answer: {answer}

Task:
1. Identify any claims in the Answer not present in the Context.
2. Provide a score from 0.0 to 1.0.
   - 1.0: No hallucinations; every claim is supported by the context.
   - 0.0: The answer is entirely hallucinated or contradicts the context.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("HallucinationMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("HallucinationMetric: unexpected error during scoring: %s", exc)
            return 0.0
