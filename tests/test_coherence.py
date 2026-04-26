import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.coherence import CoherenceMetric


class TestCoherenceMetric(unittest.TestCase):
    def setUp(self):
        self.metric = CoherenceMetric()
        self.mock_backend = MagicMock()

    def test_score_coherent(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {"answer": "First, we gather data. Then, we process it. Finally, we output results."}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_incoherent(self):
        self.mock_backend.generate.return_value = "0.1"
        row = {"answer": "Blue is a color. Therefore, I like pizza. No, I hate pizza."}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.1)

if __name__ == "__main__":
    unittest.main()
