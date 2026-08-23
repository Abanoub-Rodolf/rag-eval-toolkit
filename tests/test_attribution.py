from rag_eval.metrics.attribution import ChunkAttributionMetric


class TestChunkAttributionMetric:
    def test_score_correct_attribution(self, mock_backend):
        mock_backend.generate.return_value = "1.0"
        row = {
            "context": ["Chunk 1: RAG is cool.", "Chunk 2: Python is great."],
            "answer": "RAG is cool (Chunk 1) and Python is great (Chunk 2)."
        }
        score = ChunkAttributionMetric().score(row, mock_backend)
        assert score == 1.0

    def test_score_incorrect_attribution(self, mock_backend):
        mock_backend.generate.return_value = "0.0"
        row = {
            "context": ["Chunk 1: RAG is cool.", "Chunk 2: Python is great."],
            "answer": "RAG is cool (Chunk 2)."
        }
        score = ChunkAttributionMetric().score(row, mock_backend)
        assert score == 0.0
