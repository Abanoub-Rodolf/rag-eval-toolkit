"""
Example: Creating a custom evaluation metric.

Extend BaseMetric and implement the score() method.
Your metric receives the dataset row and an LLM backend,
and returns a float between 0.0 and 1.0.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from typing import Dict, Any
from rag_eval.metrics.base import BaseMetric


class FactualDensityMetric(BaseMetric):
    """Custom metric: measures how many factual claims per sentence the answer contains."""

    def __init__(self):
        super().__init__(name="factual_density")

    def score(self, row: Dict[str, Any], backend) -> float:
        answer = row.get("answer", "")

        prompt = f"""
Count the factual claims in this answer and rate how information-dense it is.

Answer: {answer}

Score between 0.0 and 1.0:
0.0 = no factual content, purely filler
1.0 = every sentence contains verifiable facts
Only return the number, nothing else.
"""
        response = backend.generate(prompt)
        try:
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            return 0.0


if __name__ == "__main__":
    print("Custom metric defined: FactualDensityMetric")
    print("Use it with RAGEvaluator.add_metric() just like built-in metrics.")
    print()
    print("Example:")
    print("  evaluator.add_metric(FactualDensityMetric())")
