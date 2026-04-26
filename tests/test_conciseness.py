import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.conciseness import ConcisenessMetric


class TestConcisenessMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ConcisenessMetric()
        self.mock_backend = MagicMock()

    def test_score_concise(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {"answer": "Paris is the capital of France."}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_verbose(self):
        self.mock_backend.generate.return_value = "0.3"
        row = {"answer": "Well, you see, if you look at a map of Europe, and you find France, and then you look for the biggest city, which happens to be Paris, that is the capital."}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.3)

if __name__ == "__main__":
    unittest.main()
