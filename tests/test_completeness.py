import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.completeness import AnswerCompletenessMetric


class TestAnswerCompletenessMetric(unittest.TestCase):
    def setUp(self):
        self.metric = AnswerCompletenessMetric()
        self.mock_backend = MagicMock()

    def test_score_complete(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG and how does it work?",
            "answer": "RAG is Retrieval-Augmented Generation. It works by retrieving relevant context and using it to ground the LLM response."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_incomplete(self):
        self.mock_backend.generate.return_value = "0.5"
        row = {
            "question": "What is RAG and how does it work?",
            "answer": "RAG is Retrieval-Augmented Generation."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.5)

if __name__ == "__main__":
    unittest.main()
