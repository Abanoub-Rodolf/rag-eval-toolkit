"""RAG Eval Toolkit — Core evaluator module."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Orchestrates RAG pipeline evaluation across one or more metrics.

    Args:
        backend: An LLM backend instance used to power LLM-as-judge evaluation.
        cache_path: Optional path for SQLite cache database.
    """

    def __init__(self, backend: Any, cache_path: Optional[str] = None) -> None:
        self.backend = backend
        self.metrics: List[Any] = []
        from rag_eval.utils.cache import EvaluationCache
        self.cache = EvaluationCache(db_path=cache_path) if cache_path else None

    def add_metric(self, metric: Any) -> None:
        """Register a metric to be computed during evaluation.

        Args:
            metric: A metric instance implementing the BaseMetric interface.
        """
        self.metrics.append(metric)

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run all registered metrics over the provided dataset (synchronous).

        Safe to call from regular Python scripts. If you are already inside an
        async event loop (Jupyter, FastAPI, etc.), use ``await aevaluate()`` instead.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.aevaluate(dataset)).result()
        return asyncio.run(self.aevaluate(dataset))

    async def aevaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run all registered metrics asynchronously over the dataset.

        Args:
            dataset: List of dicts containing 'question', 'context', 'answer'.

        Returns:
            Dictionary with averages and per-sample scores.
        """
        import asyncio
        if not dataset:
            raise ValueError("Dataset must not be empty.")
        if not self.metrics:
            raise ValueError("No metrics registered.")

        # In a real implementation, metrics would also need to be async-aware.
        # For v1.0, we'll wrap synchronous scoring in threads to achieve parallelism
        # if the backend/metrics aren't native async.
        
        per_metric_scores: Dict[str, List[float]] = {m.name: [] for m in self.metrics}
        
        # Define a helper for concurrent execution
        async def score_task(metric, item):
            # Check cache first
            model_name = getattr(self.backend, "model", "default")
            if self.cache:
                cached_score = self.cache.get(metric.name, model_name, item)
                if cached_score is not None:
                    return cached_score
            
            loop = asyncio.get_running_loop()
            score = await loop.run_in_executor(None, metric.score, item, self.backend)
            
            # Store in cache
            if self.cache:
                self.cache.set(metric.name, model_name, item, score)
            return score

        tasks = []
        for item in dataset:
            for metric in self.metrics:
                tasks.append(score_task(metric, item))
        
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Re-map results to per-metric lists
        idx = 0
        for item in dataset:
            for metric in self.metrics:
                res = raw_results[idx]
                if isinstance(res, Exception):
                    logger.error("Error in async scoring for %s: %s", metric.name, res)
                    per_metric_scores[metric.name].append(0.0)
                else:
                    per_metric_scores[metric.name].append(max(0.0, min(1.0, float(res))))
                idx += 1

        averages = {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in per_metric_scores.items()
            if scores
        }

        return {"averages": averages, "per_sample": per_metric_scores}

    def generate_report(
        self, results: Dict[str, Any], output: str = "eval_report.html"
    ) -> str:
        """Generate an HTML evaluation report.

        Args:
            results: Output from evaluate(), containing averages and per-sample scores.
            output: File path for the generated HTML report.

        Returns:
            Path to the generated report file.
        """
        try:
            from rag_eval.report.generator import generate_html_report
            generate_html_report(results, output)
            logger.info("Report saved to %s", output)
        except ImportError:
            logger.warning("Report generator not available. Install with: pip install rag-eval-toolkit[report]")
        return output
