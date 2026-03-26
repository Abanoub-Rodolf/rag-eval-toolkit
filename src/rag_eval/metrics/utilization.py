"""Context utilization metric -- measures how much retrieved context was used in the answer."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class ContextUtilizationMetric(BaseMetric):
    """Evaluates how much of the retrieved context was actually utilized in the answer.
    
    A score of 1.0 means the context was effectively utilized to answer the question.
    A score of 0.0 means the context was ignored or irrelevant to the answer.
    """

    def __init__(self) -> None:
        """Initialise ContextUtilizationMetric."""
        super().__init__(name="context_utilization")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for context utilization.
        
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
Evaluate how much of the provided Context was utilized to generate the Answer for the Question.

Question: {question}
Context: {context}
Answer: {answer}

Task:
1. Identify parts of the Context that are relevant to the Question.
2. Determine if these relevant parts were used to form the Answer.
3. Provide a utilization score from 0.0 to 1.0.
   - 1.0: Most or all relevant context was effectively used in the answer.
   - 0.0: Relevant context was ignored, or the answer was generated without using the context.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("ContextUtilizationMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("ContextUtilizationMetric: unexpected error during scoring: %s", exc)
            return 0.0
