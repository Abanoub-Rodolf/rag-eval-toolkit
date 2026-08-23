import pytest

from rag_eval.metrics.base import ScoreParseError
from rag_eval.metrics.faithfulness import FaithfulnessMetric


class TestFaithfulnessMetric:
    def test_score_valid_response(self, mock_backend):
        mock_backend.generate.return_value = "0.9"
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "RAG is a technique."
        }
        score = FaithfulnessMetric().score(row, mock_backend)
        assert score == 0.9

    def test_score_valid_response_with_reasoning(self, mock_backend):
        mock_backend.generate.return_value = (
            "The answer restates the context accurately.\nSCORE: 0.95"
        )
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "RAG is a technique."
        }
        score = FaithfulnessMetric().score(row, mock_backend)
        assert score == 0.95

    def test_score_invalid_response_raises(self, mock_backend):
        mock_backend.generate.return_value = "invalid"
        row = {"question": "Q", "context": "C", "answer": "A"}
        with pytest.raises(ScoreParseError):
            FaithfulnessMetric().score(row, mock_backend)
