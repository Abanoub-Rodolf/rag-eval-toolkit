from rag_eval.metrics.conciseness import ConcisenessMetric


class TestConcisenessMetric:
    def test_score_concise(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {"answer": "Paris is the capital of France."}
        score = ConcisenessMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_verbose(self, mock_backend):
        mock_backend.generate.return_value = "0.3"
        row = {"answer": "Well, you see, if you look at a map of Europe, and you find France, and then you look for the biggest city, which happens to be Paris, that is the capital."}
        score = ConcisenessMetric().score(row, mock_backend)
        assert score == 0.3
