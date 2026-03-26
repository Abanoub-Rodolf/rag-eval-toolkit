import unittest
from unittest.mock import MagicMock
from rag_eval.metrics.attribution import ChunkAttributionMetric

class TestChunkAttributionMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ChunkAttributionMetric()
        self.mock_backend = MagicMock()

    def test_score_correct_attribution(self):
        self.mock_backend.generate.return_value = "1.0"
        row = {
            "context": ["Chunk 1: RAG is cool.", "Chunk 2: Python is great."],
            "answer": "RAG is cool (Chunk 1) and Python is great (Chunk 2)."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 1.0)

    def test_score_incorrect_attribution(self):
        self.mock_backend.generate.return_value = "0.0"
        row = {
            "context": ["Chunk 1: RAG is cool.", "Chunk 2: Python is great."],
            "answer": "RAG is cool (Chunk 2)."
        }
        score = self.metric.score(row, self.mock_backend)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
