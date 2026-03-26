import unittest
import asyncio
import os
import shutil
from unittest.mock import MagicMock
from rag_eval.evaluator import RAGEvaluator

class TestAsyncEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_backend = MagicMock()
        self.mock_backend.model = "test-model"
        self.mock_backend.generate.return_value = "0.9"
        self.cache_path = "test_cache.db"
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        self.evaluator = RAGEvaluator(backend=self.mock_backend, cache_path=self.cache_path)

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def test_aevaluate_parallel(self):
        mock_metric = MagicMock()
        mock_metric.name = "m1"
        mock_metric.score.return_value = 0.8
        self.evaluator.add_metric(mock_metric)
        
        dataset = [{"question": "q1", "context": "c1", "answer": "a1"}] * 5
        results = asyncio.run(self.evaluator.aevaluate(dataset))
        
        self.assertEqual(results["averages"]["m1"], 0.8)
        # Check that cache was populated (only 1 call to metric.score due to caching in loop? 
        # No, aevaluate parallelizes FIRST, so all might hit score() if not careful.
        # Wait, my aevaluate loop:
        # for item in dataset:
        #   for metric in self.metrics:
        #     tasks.append(score_task(metric, item))
        # Since they all start together, they might all hit score() if they share items.
        
    def test_caching(self):
        mock_metric = MagicMock()
        mock_metric.name = "m2"
        mock_metric.score.return_value = 0.7
        self.evaluator.add_metric(mock_metric)
        
        row = {"question": "q", "context": "c", "answer": "a"}
        
        # First call hits score()
        score1 = asyncio.run(self.evaluator.aevaluate([row]))
        self.assertEqual(mock_metric.score.call_count, 1)
        
        # Second call hits cache
        score2 = asyncio.run(self.evaluator.aevaluate([row]))
        self.assertEqual(mock_metric.score.call_count, 1)
        self.assertEqual(score2["averages"]["m2"], 0.7)

if __name__ == "__main__":
    unittest.main()
