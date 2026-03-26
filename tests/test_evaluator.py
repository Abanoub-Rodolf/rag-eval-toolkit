import unittest
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

if __name__ == "__main__":
    unittest.main()
