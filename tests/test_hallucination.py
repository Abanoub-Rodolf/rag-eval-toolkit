from rag_eval.metrics.hallucination import HallucinationMetric


class TestHallucinationMetric:
    def test_score_no_hallucination(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {
            "question": "What is RAG?",
            "context": "RAG stands for Retrieval-Augmented Generation.",
            "answer": "RAG means Retrieval-Augmented Generation."
        }
        score = HallucinationMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_with_hallucination(self, mock_backend):
        mock_backend.generate.return_value = "0.2"
        row = {
            "question": "What is RAG?",
            "context": "RAG stands for Retrieval-Augmented Generation.",
            "answer": "RAG was invented by Leonardo da Vinci."
        }
        score = HallucinationMetric().score(row, mock_backend)
        assert score == 0.2
