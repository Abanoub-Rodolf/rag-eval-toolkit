import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.context_recall import ContextRecallMetric


class TestContextRecallMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ContextRecallMetric()
        self.mock_backend = MagicMock()

    def test_score_full_recall_decomposed(self):
        self.mock_backend.generate.return_value = "1: yes\n2: yes"
        row = {
            "question": "What is RAG?",
            "context": "RAG combines retrieval with generation and grounds answers.",
            "ground_truth": "RAG combines retrieval with generation. It grounds answers.",
        }
        assert self.metric.score(row, self.mock_backend) == 1.0

    def test_score_partial_recall_decomposed(self):
        self.mock_backend.generate.return_value = "1: yes\n2: no"
        row = {
            "question": "What is RAG?",
            "context": "RAG retrieves docs.",
            "ground_truth": "RAG retrieves docs. RAG also generates grounded answers.",
        }
        assert self.metric.score(row, self.mock_backend) == 0.5

    def test_falls_back_to_holistic_when_decomposition_unparseable(self):
        self.mock_backend.generate.side_effect = ["not the right format", "0.6"]
        row = {
            "question": "What is RAG?",
            "context": "RAG retrieves docs.",
            "ground_truth": "RAG retrieves docs.",
        }
        assert self.metric.score(row, self.mock_backend) == 0.6
        assert self.mock_backend.generate.call_count == 2

    def test_missing_ground_truth_raises(self):
        row = {"question": "Q", "context": "C"}
        with self.assertRaises(ValueError):
            self.metric.score(row, self.mock_backend)

    def test_list_context_is_flattened(self):
        self.mock_backend.generate.return_value = "1: yes"
        row = {
            "question": "Q",
            "context": ["chunk one", "chunk two"],
            "ground_truth": "One fact.",
        }
        assert self.metric.score(row, self.mock_backend) == 1.0

    def test_metric_name(self):
        assert self.metric.name == "context_recall"

if __name__ == "__main__":
    unittest.main()
