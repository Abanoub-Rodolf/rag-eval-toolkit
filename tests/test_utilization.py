from rag_eval.metrics.utilization import ContextUtilizationMetric


class TestContextUtilizationMetric:
    def test_score_high_utilization(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "Retrieval-Augmented Generation (RAG) is a method."
        }
        score = ContextUtilizationMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_low_utilization(self, mock_backend):
        mock_backend.generate.return_value = "0.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG is Retrieval-Augmented Generation.",
            "answer": "I like pizza."
        }
        score = ContextUtilizationMetric().score(row, mock_backend)
        assert score == 0.0
