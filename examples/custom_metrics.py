"""Example: build a custom evaluation metric.

Extend BaseMetric and implement score(). The metric receives the dataset row
and an LLM backend, and returns a float in [0.0, 1.0].
"""
from typing import Any, Dict

from rag_eval import BaseMetric


class FactualDensityMetric(BaseMetric):
    """Measures how many factual claims per sentence the answer contains."""

    def __init__(self):
        super().__init__(name="factual_density")

    def score(self, row: Dict[str, Any], backend) -> float:
        answer = row.get("answer", "")
        prompt = (
            "Count the factual claims in this answer and rate how information-dense it is.\n\n"
            f"Answer: {answer}\n\n"
            "Score between 0.0 and 1.0:\n"
            "0.0 = no factual content, purely filler\n"
            "1.0 = every sentence contains verifiable facts\n"
            "Only return the number, nothing else."
        )
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
