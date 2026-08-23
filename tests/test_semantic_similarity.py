import unittest
from unittest.mock import MagicMock

from rag_eval.metrics.semantic_similarity import BackendCapabilityError, SemanticSimilarityMetric


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

    def test_score_missing_ground_truth_raises(self):
        row = {"answer": "a"}
        with self.assertRaises(ValueError):
            self.metric.score(row, self.mock_backend)

    def test_backend_empty_embedding_raises_capability_error(self):
        # An embed() that fails must not degrade into a silent 0.0 cosine score
        self.mock_backend.embed.return_value = []
        row = {"answer": "a", "ground_truth": "b"}
        with self.assertRaises(BackendCapabilityError):
            self.metric.score(row, self.mock_backend)

    def test_backend_without_embed_raises_capability_error(self):
        backend = MagicMock(spec=[])  # no attributes at all, including embed
        row = {"answer": "a", "ground_truth": "b"}
        with self.assertRaises(BackendCapabilityError):
            self.metric.score(row, backend)

if __name__ == "__main__":
    unittest.main()
