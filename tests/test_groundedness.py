from rag_eval.metrics.groundedness import GroundednessMetric


class TestGroundednessMetric:
    def test_score_grounded(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {
            "context": "The Eiffel Tower is in Paris.",
            "answer": "The Eiffel Tower is located in the city of Paris."
        }
        score = GroundednessMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_not_grounded(self, mock_backend):
        mock_backend.generate.return_value = "0.0"
        row = {
            "context": "The Eiffel Tower is in Paris.",
            "answer": "The Eiffel Tower is in London."
        }
        score = GroundednessMetric().score(row, mock_backend)
        assert score == 0.0
