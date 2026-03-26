import unittest
from unittest.mock import MagicMock
from rag_eval.metrics.groundedness import GroundednessMetric

class TestGroundednessMetric(unittest.TestCase):
    def setUp(self):
        self.metric = GroundednessMetric()
        self.mock_backend = MagicMock()

    def test_score_grounded(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {
            "context": "The Eiffel Tower is in Paris.",
            "answer": "The Eiffel Tower is located in the city of Paris."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_not_grounded(self):
        self.mock_backend.generate.return_value = "0.0"
        row = {
            "context": "The Eiffel Tower is in Paris.",
            "answer": "The Eiffel Tower is in London."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
