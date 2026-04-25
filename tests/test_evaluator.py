import unittest
import pytest
from unittest.mock import MagicMock
from rag_eval.evaluator import RAGEvaluator

class TestRAGEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_backend = MagicMock()
        self.mock_backend.generate.return_value = "0.85"
        self.evaluator = RAGEvaluator(backend=self.mock_backend)

    def test_evaluate_returns_results(self):
        mock_metric = MagicMock()
        mock_metric.name = "faithfulness"
        mock_metric.score.return_value = 0.9
        self.evaluator.add_metric(mock_metric)
        dataset = [{"question": "What is RAG?", "context": "RAG stands for Retrieval-Augmented Generation.", "answer": "RAG is a technique."}]
        results = self.evaluator.evaluate(dataset)
        # Check that averages are calculated correctly
        self.assertIn("averages", results)
        self.assertIn("faithfulness", results["averages"])
        self.assertEqual(results["averages"]["faithfulness"], 0.9)

    def test_evaluate_empty_dataset(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate([])

    def test_evaluate_no_metrics_raises(self):
        dataset = [{"question": "q", "context": "c", "answer": "a"}]
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(dataset)

    def test_evaluate_per_sample_scores(self):
        mock_metric = MagicMock()
        mock_metric.name = "coherence"
        mock_metric.score.side_effect = [0.6, 0.8]
        self.evaluator.add_metric(mock_metric)
        dataset = [
            {"question": "q1", "context": "c1", "answer": "a1"},
            {"question": "q2", "context": "c2", "answer": "a2"},
        ]
        results = self.evaluator.evaluate(dataset)
        assert results["per_sample"]["coherence"] == [0.6, 0.8]
        assert results["averages"]["coherence"] == pytest.approx(0.7, abs=1e-4)

    def test_on_progress_fires_once_per_score(self):
        m1 = MagicMock()
        m1.name = "faithfulness"
        m1.score.return_value = 0.9
        m2 = MagicMock()
        m2.name = "coherence"
        m2.score.return_value = 0.7
        self.evaluator.add_metric(m1)
        self.evaluator.add_metric(m2)
        dataset = [
            {"question": "q1", "context": "c1", "answer": "a1"},
            {"question": "q2", "context": "c2", "answer": "a2"},
        ]
        calls = []
        self.evaluator.evaluate(dataset, on_progress=lambda: calls.append(1))
        # 2 samples × 2 metrics = 4 progress events
        assert len(calls) == 4

    def test_on_progress_fires_on_cache_hit(self):
        """Cache hits must still fire on_progress so the bar reaches 100%."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            backend = MagicMock()
            backend.model = "test-model"  # must be a string for JSON serialization in cache key
            ev = RAGEvaluator(backend=backend, cache_path=db)
            m = MagicMock()
            m.name = "faithfulness"
            m.score.return_value = 0.8
            ev.add_metric(m)
            row = {"question": "q", "context": "c", "answer": "a"}

            # first call — populates cache
            calls = []
            ev.evaluate([row], on_progress=lambda: calls.append(1))
            assert len(calls) == 1

            # second call — cache hit path
            calls2 = []
            ev.evaluate([row], on_progress=lambda: calls2.append(1))
            assert len(calls2) == 1
        finally:
            os.unlink(db)


if __name__ == "__main__":
    unittest.main()
