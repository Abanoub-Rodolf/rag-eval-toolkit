"""Conciseness metric -- measures if the answer is direct and free of fluff."""
import logging
from typing import Any, Dict

from .base import BaseMetric

logger = logging.getLogger(__name__)


class ConcisenessMetric(BaseMetric):
    """Evaluates if the answer is concise and avoids unnecessary verbosity.
    
    A score of 1.0 means the answer is perfectly concise.
    A score of 0.0 means the answer is extremely verbose or filled with irrelevant fluff.
    """

    def __init__(self) -> None:
        """Initialise ConcisenessMetric."""
        super().__init__(name="conciseness")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        """Score for conciseness.
        
        Args:
            row: Must contain 'answer' key.
            backend: LLM backend instance.
            
        Returns:
            Float in [0.0, 1.0].
        """
        answer = row.get("answer")

        prompt = f"""
Evaluate the conciseness of the following Answer.

Answer: {answer}

Task:
1. Determine if the answer provides the necessary information without unnecessary words, repetition, or fluff.
2. Provide a conciseness score from 0.0 to 1.0.
   - 1.0: Perfectly concise and direct.
   - 0.0: Extremely wordy, repetitive, or contains excessive irrelevant information.

Respond ONLY with the float score.
"""
        try:
            response = backend.generate(prompt)
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            logger.warning("ConcisenessMetric: could not parse backend response as float.")
            return 0.0
        except Exception as exc:
            logger.error("ConcisenessMetric: unexpected error during scoring: %s", exc)
            return 0.0
