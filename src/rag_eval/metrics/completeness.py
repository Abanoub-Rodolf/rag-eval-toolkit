"""Answer completeness metric -- measures if the answer addresses all parts of the question."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class AnswerCompletenessMetric(BaseMetric):
    """Evaluates if the answer addresses all parts and nuances of the question.
    
    A score of 1.0 means the answer is complete.
    A score of 0.0 means the answer is completely incomplete or missing critical information.
    """

    def __init__(self) -> None:
        """Initialise AnswerCompletenessMetric."""
        super().__init__(name="completeness")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for completeness.
        
        Args:
            row: Must contain 'question' and 'answer' keys.
            backend: LLM backend instance.
            
        Returns:
            Float in [0.0, 1.0].
        """
        question = row.get("question")
        answer = row.get("answer")

        prompt = f"""
Evaluate the completeness of the following Answer in response to the Question.

Question: {question}
Answer: {answer}

Task:
1. Identify all components or sub-questions within the Question.
2. Determine if the Answer addresses each of these components.
3. Provide a completeness score from 0.0 to 1.0.
   - 1.0: All parts of the question are fully and accurately addressed.
   - 0.0: The answer fails to address the question or misses almost all parts.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("AnswerCompletenessMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("AnswerCompletenessMetric: unexpected error during scoring: %s", exc)
            return 0.0
