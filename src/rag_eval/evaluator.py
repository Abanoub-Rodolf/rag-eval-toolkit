"""Core RAG evaluation orchestrator."""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from rag_eval.backends.base import BaseBackend
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 20


class RAGEvaluator:
    """Orchestrates RAG pipeline evaluation across one or more metrics.

    Args:
        backend: An LLM backend instance used to power LLM-as-judge evaluation.
        cache_path: Optional path for SQLite cache database. Pass None to disable.
        max_concurrency: Max parallel scoring tasks (default 20).
    """

    def __init__(
        self,
        backend: Any,
        cache_path: Optional[str] = None,
        max_concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        self.backend = backend
        self.metrics: List[Any] = []
        self._max_concurrency = max_concurrency
        from rag_eval.utils.cache import EvaluationCache
        self.cache = EvaluationCache(db_path=cache_path) if cache_path else None

    def add_metric(self, metric: Any) -> None:
        """Register a metric to be computed during evaluation."""
        self.metrics.append(metric)

    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run all registered metrics over the dataset (synchronous).

        Safe to call from regular Python scripts. If already inside an async
        event loop (Jupyter, FastAPI, etc.), use ``await aevaluate()`` instead.

        Args:
            dataset: List of dicts containing ``question``, ``context``, ``answer``.
            on_progress: Optional zero-argument callable invoked after each
                score completes. Use to drive a progress bar.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.aevaluate(dataset, on_progress=on_progress)).result()
        return asyncio.run(self.aevaluate(dataset, on_progress=on_progress))

    async def aevaluate(
        self,
        dataset: List[Dict[str, Any]],
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run all registered metrics asynchronously over the dataset.

        Args:
            dataset: List of dicts containing ``question``, ``context``, ``answer``.
            on_progress: Optional zero-argument callable invoked after each
                score completes. Fires ``len(dataset) * len(metrics)`` times total.

        Returns:
            Dict with ``averages`` (per-metric mean) and ``per_sample`` (all scores).
        """
        if not dataset:
            raise ValueError("Dataset must not be empty.")
        if not self.metrics:
            raise ValueError("No metrics registered.")

        # re-create semaphore inside the running loop
        sem = asyncio.Semaphore(self._max_concurrency)

        per_metric_scores: Dict[str, List[float]] = {m.name: [] for m in self.metrics}

        async def score_task(metric: Any, item: Dict[str, Any]) -> float:
            try:
                model_name = getattr(self.backend, "model", "default")
                if self.cache:
                    cached = self.cache.get(metric.name, model_name, item)
                    if cached is not None:
                        return cached

                async with sem:
                    loop = asyncio.get_running_loop()
                    score = await loop.run_in_executor(None, metric.score, item, self.backend)

                if self.cache:
                    self.cache.set(metric.name, model_name, item, score)
                return score
            finally:
                if on_progress is not None:
                    try:
                        on_progress()
                    except Exception:
                        logger.exception("on_progress callback raised")

        tasks = [
            score_task(metric, item)
            for item in dataset
            for metric in self.metrics
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        idx = 0
        for item in dataset:
            for metric in self.metrics:
                res = raw_results[idx]
                if isinstance(res, Exception):
                    logger.error("Error scoring %s: %s", metric.name, res)
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
            results: Output from evaluate().
            output: Destination file path for the HTML report.

        Returns:
            Path to the generated report.
        """
        from rag_eval.report.generator import generate_html_report
        generate_html_report(results, output)
        logger.info("Report saved to %s", output)
        return output
