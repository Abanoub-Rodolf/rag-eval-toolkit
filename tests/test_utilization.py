import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.utilization import ContextUtilizationMetric


class TestContextUtilizationMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ContextUtilizationMetric()
        self.mock_backend = MagicMock()

    def test_score_high_utilization(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "Retrieval-Augmented Generation (RAG) is a method."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_low_utilization(self):
        self.mock_backend.generate.return_value = "0.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "I like pizza."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
