import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.toxicity import ToxicityMetric


class TestToxicityMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ToxicityMetric()
        self.mock_backend = MagicMock()

    def test_score_safe(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {"answer": "Hello, how can I help you today?"}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_toxic(self):
        self.mock_backend.generate.return_value = "0.0"
        row = {"answer": "I hate you and I will destroy you."}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
