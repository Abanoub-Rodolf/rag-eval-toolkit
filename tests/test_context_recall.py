import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.context_recall import ContextRecallMetric


class TestContextRecallMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ContextRecallMetric()
        self.mock_backend = MagicMock()

    def test_score_full_recall(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG combines retrieval with generation.",
            "ground_truth": "RAG combines retrieval with generation.",
        }
        assert self.metric.score(row, self.mock_backend) == 1.0

    def test_score_partial_recall(self):
        self.mock_backend.generate.return_value = "0.5"
        row = {
            "question": "What is RAG?",
            "context": "RAG retrieves docs.",
            "ground_truth": "RAG retrieves docs and generates grounded answers.",
        }
        assert self.metric.score(row, self.mock_backend) == 0.5

    def test_score_missing_ground_truth_logs_warning(self):
        self.mock_backend.generate.return_value = "0.7"
        row = {"question": "Q", "context": "C"}
        score = self.metric.score(row, self.mock_backend)
        assert 0.0 <= score <= 1.0

    def test_metric_name(self):
        assert self.metric.name == "context_recall"


if __name__ == "__main__":
    unittest.main()
