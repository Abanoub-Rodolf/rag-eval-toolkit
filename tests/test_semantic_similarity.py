import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.semantic_similarity import SemanticSimilarityMetric


class TestSemanticSimilarityMetric(unittest.TestCase):
    def setUp(self):
        self.metric = SemanticSimilarityMetric()
        self.mock_backend = MagicMock()

    def test_score_identical_embeddings(self):
        self.mock_backend.embed.side_effect = lambda x: [1.0, 0.0] if x == "a" else [1.0, 0.0]
        row = {"answer": "a", "ground_truth": "a"}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_different_embeddings(self):
        # Orthogonal vectors
        self.mock_backend.embed.side_effect = lambda x: [1.0, 0.0] if x == "a" else [0.0, 1.0]
        row = {"answer": "a", "ground_truth": "b"}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

    def test_score_missing_ground_truth(self):
        row = {"answer": "a"}
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
