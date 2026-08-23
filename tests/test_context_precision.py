import unittest
from unittest.mock import MagicMock

import pytest

from rag_eval.metrics.base import ScoreParseError
from rag_eval.metrics.context_precision import ContextPrecisionMetric


class TestContextPrecisionMetricHolistic(unittest.TestCase):
    """String context has no chunk boundaries, so this always uses the
    single-call holistic judgment path."""

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

    def test_score_unparseable_response_raises(self):
        self.mock_backend.generate.return_value = "no clue"
        row = {"question": "Q", "context": "C"}
        with self.assertRaises(ScoreParseError):
            self.metric.score(row, self.mock_backend)

    def test_single_chunk_list_uses_holistic_path(self):
        # a 1-element list has no ranking to speak of; treat like a string
        self.mock_backend.generate.return_value = "0.7"
        row = {"question": "Q", "context": ["only chunk"]}
        assert self.metric.score(row, self.mock_backend) == 0.7

    def test_metric_name(self):
        assert self.metric.name == "context_precision"


class TestContextPrecisionMetricRanked(unittest.TestCase):
    """List context in retrieval order triggers the rank-aware average
    precision path (Ragas' definition)."""

    def setUp(self):
        self.metric = ContextPrecisionMetric()
        self.mock_backend = MagicMock()

    def test_all_relevant_scores_one(self):
        self.mock_backend.generate.return_value = "1: yes\n2: yes\n3: yes"
        row = {"question": "Q", "context": ["a", "b", "c"]}
        assert self.metric.score(row, self.mock_backend) == 1.0

    def test_none_relevant_scores_zero(self):
        self.mock_backend.generate.return_value = "1: no\n2: no\n3: no"
        row = {"question": "Q", "context": ["a", "b", "c"]}
        assert self.metric.score(row, self.mock_backend) == 0.0

    def test_relevant_chunk_ranked_first_scores_higher_than_last(self):
        first_relevant = self.metric._average_precision([True, False, False])
        last_relevant = self.metric._average_precision([False, False, True])
        assert first_relevant > last_relevant
        assert first_relevant == 1.0
        assert last_relevant == pytest.approx(1 / 3)

    def test_average_precision_mixed(self):
        # relevant at rank 1 and 3: precision@1=1/1, precision@3=2/3
        ap = self.metric._average_precision([True, False, True])
        assert ap == pytest.approx((1 / 1 + 2 / 3) / 2)

    def test_falls_back_to_holistic_on_malformed_verdicts(self):
        # first call (ranked) returns garbage, second call (holistic fallback)
        # returns a plain score
        self.mock_backend.generate.side_effect = ["not the right format", "0.42"]
        row = {"question": "Q", "context": ["a", "b"]}
        assert self.metric.score(row, self.mock_backend) == 0.42
        assert self.mock_backend.generate.call_count == 2


if __name__ == "__main__":
    unittest.main()
