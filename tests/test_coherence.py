from rag_eval.metrics.coherence import CoherenceMetric


class TestCoherenceMetric:
    def test_score_coherent(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {"answer": "First, we gather data. Then, we process it. Finally, we output results."}
        score = CoherenceMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_incoherent(self, mock_backend):
        mock_backend.generate.return_value = "0.1"
        row = {"answer": "Blue is a color. Therefore, I like pizza. No, I hate pizza."}
        score = CoherenceMetric().score(row, mock_backend)
        assert score == 0.1
