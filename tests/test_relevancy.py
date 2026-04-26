import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.relevancy import AnswerRelevancyMetric


class TestAnswerRelevancyMetric(unittest.TestCase):
    def setUp(self):
        self.metric = AnswerRelevancyMetric()
        self.mock_backend = MagicMock()

    def test_score_valid_response(self):
        self.mock_backend.generate.return_value = "0.95"
        row = {
            "question": "What is RAG?",
            "answer": "RAG is a technique that combines retrieval and generation."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.95)

    def test_score_invalid_response(self):
        self.mock_backend.generate.return_value = "fail"
        row = {"question": "Q", "answer": "A"}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
