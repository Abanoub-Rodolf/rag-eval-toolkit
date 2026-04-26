import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.context_precision import ContextPrecisionMetric


class TestContextPrecisionMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ContextPrecisionMetric()
        self.mock_backend = MagicMock()

    def test_score_high_precision(self):
        self.mock_backend.generate.return_value = "0.9"
        row = {
            "question": "What is RAG?",
            "context": "RAG combines retrieval with generation.",
        }
        assert self.metric.score(row, self.mock_backend) == 0.9

    def test_score_low_precision(self):
        self.mock_backend.generate.return_value = "0.1"
        row = {"question": "What is RAG?", "context": "Cooking recipes."}
        assert self.metric.score(row, self.mock_backend) == 0.1

    def test_score_unparseable_response_returns_zero(self):
        self.mock_backend.generate.return_value = "no clue"
        row = {"question": "Q", "context": "C"}
        assert self.metric.score(row, self.mock_backend) == 0.0

    def test_metric_name(self):
        assert self.metric.name == "context_precision"


if __name__ == "__main__":
    unittest.main()
