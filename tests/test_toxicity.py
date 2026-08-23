from rag_eval.metrics.toxicity import ToxicityMetric


class TestToxicityMetric:
    def test_score_safe(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {"answer": "Hello, how can I help you today?"}
        score = ToxicityMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_toxic(self, mock_backend):
        mock_backend.generate.return_value = "0.0"
        row = {"answer": "I hate you and I will destroy you."}
        score = ToxicityMetric().score(row, mock_backend)
        assert score == 0.0
