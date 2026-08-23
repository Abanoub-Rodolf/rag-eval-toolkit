import pytest

from rag_eval.metrics.base import ScoreParseError
from rag_eval.metrics.completeness import AnswerCompletenessMetric


class TestAnswerCompletenessMetric:
    def test_score_complete(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG and how does it work?",
            "answer": "RAG is Retrieval-Augmented Generation. It works by retrieving relevant context and using it to ground the LLM response."
        }
        score = AnswerCompletenessMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_incomplete(self, mock_backend):
        mock_backend.generate.return_value = "0.5"
        row = {
            "question": "What is RAG and how does it work?",
            "answer": "RAG is Retrieval-Augmented Generation."
        }
        score = AnswerCompletenessMetric().score(row, mock_backend)
        assert score == 0.5

    # Reconstructed test: the original uncommitted version of this file was
    # lost to an editor overwrite; rebuilt from the repo-wide edit pattern
    # (silent 0.0 -> ScoreParseError) applied for the 2.0.0 release.
    def test_score_invalid_response_raises(self, mock_backend):
        mock_backend.generate.return_value = "invalid"
        row = {"question": "Q", "answer": "A"}
        with pytest.raises(ScoreParseError):
            AnswerCompletenessMetric().score(row, mock_backend)
