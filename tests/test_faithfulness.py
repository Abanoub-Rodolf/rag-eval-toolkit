import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.faithfulness import FaithfulnessMetric


class TestFaithfulnessMetric(unittest.TestCase):
    def setUp(self):
        self.metric = FaithfulnessMetric()
        self.mock_backend = MagicMock()

    def test_score_valid_response(self):
        self.mock_backend.generate.return_value = "0.9"
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "RAG is a technique."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.9)

    def test_score_invalid_response(self):
        self.mock_backend.generate.return_value = "invalid"
        row = {"question": "Q", "context": "C", "answer": "A"}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
