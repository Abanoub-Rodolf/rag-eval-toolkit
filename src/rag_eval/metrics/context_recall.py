"""Context recall metric -- measures coverage of ground-truth information in the context."""
from typing import Any, Dict

from .base import BaseMetric


class ContextRecallMetric(BaseMetric):
    """Evaluates how much of the ground-truth answer is covered by the context.

    A score of 1.0 means every piece of ground-truth information appears in
    the retrieved context. A score of 0.0 means none of the required
    information is present.

    Note:
        Requires a ``ground_truth`` key in the input row. If absent, an empty
        string is used which may produce unreliable scores.
    """

    def __init__(self) -> None:
        super().__init__(name="context_recall")

    def score(self, row: Dict[str, Any], backend: Any) -> float:
        question = row.get("question", "")
        context = row.get("context", "")
        ground_truth = row.get("ground_truth", "")

        prompt = (
            "You are an impartial judge evaluating RAG retrieval quality.\n\n"
            "Evaluate what fraction of the ground-truth information is present "
            "in the retrieved context.\n\n"
            f"<question>{question}</question>\n"
            f"<context>{context}</context>\n"
            f"<ground_truth>{ground_truth}</ground_truth>\n\n"
            "Respond ONLY with a single float score from 0.0 to 1.0.\n"
            "0.0 = none of the ground truth is in the context\n"
            "1.0 = all ground truth information is present in the context"
        )
        return self._generate_score(backend, prompt)
