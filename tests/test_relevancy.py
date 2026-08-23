import pytest

from rag_eval.metrics.base import ScoreParseError
from rag_eval.metrics.relevancy import AnswerRelevancyMetric


class TestAnswerRelevancyMetric:
    def test_score_valid_response(self, mock_backend):
        mock_backend.generate.return_value = "0.95"
        row = {
            "question": "What is RAG?",
            "answer": "RAG is a technique that combines retrieval and generation."
        }
        score = AnswerRelevancyMetric().score(row, mock_backend)
        assert score == 0.95

    # Reconstructed test: rebuilt from the repo-wide edit pattern
    # (silent 0.0 -> ScoreParseError) applied for the 2.0.0 release.
    def test_score_invalid_response_raises(self, mock_backend):
        mock_backend.generate.return_value = "fail"
        row = {"question": "Q", "answer": "A"}
        with pytest.raises(ScoreParseError):
            AnswerRelevancyMetric().score(row, mock_backend)
