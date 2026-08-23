"""Judge test-retest reliability measurement.

This measures judge *self-consistency*, not judge-human agreement: run the
same dataset through the same judge multiple times and see how much the
scores move around. It does not tell you whether the judge is right, only
whether it is stable. A metric with high run-to-run variance is not
trustworthy for the second thing until it's fixed for the first -- see
README's Limitations section for why this toolkit doesn't ship a
judge-human agreement number.
"""
import statistics
from typing import Any

from .evaluator import RAGEvaluator


def measure_reliability(
    evaluator: RAGEvaluator,
    dataset: list[dict[str, Any]],
    runs: int = 3,
) -> dict[str, Any]:
    """Run ``evaluator`` over ``dataset`` ``runs`` times and report per-metric
    score variance across runs.

    Caching is bypassed for the duration of this call (a cache hit would
    just return the first run's score every time, hiding variance) and
    restored afterward.

    Args:
        evaluator: A configured RAGEvaluator with metrics already added.
        dataset: The dataset to re-evaluate ``runs`` times.
        runs: Number of repeated passes. Must be at least 2.

    Returns:
        Dict with ``runs`` and a ``metrics`` mapping of metric name to
        ``{mean_stdev, max_stdev, mean_range, samples_measured}``. A sample
        contributes only if it scored successfully (no error) in at least 2
        of the runs; ``samples_measured`` reports how many did.
        ``mean_stdev``/``max_stdev``/``mean_range`` are None if fewer than 2
        runs produced a usable score for every sample.
    """
    if runs < 2:
        raise ValueError("runs must be >= 2 to measure variance")

    original_cache = evaluator.cache
    evaluator.cache = None
    try:
        all_runs = [evaluator.evaluate(dataset) for _ in range(runs)]
    finally:
        evaluator.cache = original_cache

    metric_names = list(all_runs[0]["per_sample"].keys())
    per_metric: dict[str, Any] = {}

    for name in metric_names:
        n_samples = len(all_runs[0]["per_sample"][name])
        sample_stdevs = []
        sample_ranges = []
        for i in range(n_samples):
            values = [
                r["per_sample"][name][i]
                for r in all_runs
                if r["per_sample"][name][i] is not None
            ]
            if len(values) >= 2:
                sample_stdevs.append(statistics.stdev(values))
                sample_ranges.append(max(values) - min(values))

        per_metric[name] = {
            "samples_measured": len(sample_stdevs),
            "mean_stdev": round(statistics.mean(sample_stdevs), 4) if sample_stdevs else None,
            "max_stdev": round(max(sample_stdevs), 4) if sample_stdevs else None,
            "mean_range": round(statistics.mean(sample_ranges), 4) if sample_ranges else None,
        }

    return {"runs": runs, "metrics": per_metric}
