import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.hallucination import HallucinationMetric


class TestHallucinationMetric(unittest.TestCase):
    def setUp(self):
        self.metric = HallucinationMetric()
        self.mock_backend = MagicMock()

    def test_score_no_hallucination(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG stands for Retrieval-Augmented Generation.",
            "answer": "RAG means Retrieval-Augmented Generation."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_with_hallucination(self):
        self.mock_backend.generate.return_value = "0.2"
        row = {
            "question": "What is RAG?",
            "context": "RAG stands for Retrieval-Augmented Generation.",
            "answer": "RAG was invented by Leonardo da Vinci."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.2)

if __name__ == "__main__":
    unittest.main()
