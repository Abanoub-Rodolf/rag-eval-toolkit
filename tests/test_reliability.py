import statistics
from unittest.mock import MagicMock

import pytest

from rag_eval.evaluator import RAGEvaluator
from rag_eval.reliability import measure_reliability


def _evaluator():
    backend = MagicMock()
    backend.model = "test-model"
    return RAGEvaluator(backend=backend)


def _dataset(n=2):
    return [{"question": f"q{i}", "context": f"c{i}", "answer": f"a{i}"} for i in range(n)]


class TestMeasureReliability:
    def test_runs_below_two_raises(self):
        ev = _evaluator()
        with pytest.raises(ValueError):
            measure_reliability(ev, _dataset(), runs=1)

    def test_constant_scores_have_zero_stdev(self):
        ev = _evaluator()
        metric = MagicMock()
        metric.name = "stable"
        metric.score.return_value = 0.7
        ev.add_metric(metric)

        report = measure_reliability(ev, _dataset(2), runs=3)

        stats = report["metrics"]["stable"]
        assert stats["mean_stdev"] == 0.0
        assert stats["max_stdev"] == 0.0
        assert stats["mean_range"] == 0.0
        assert stats["samples_measured"] == 2

    def test_varying_scores_compute_correct_stdev(self):
        # Runs execute sequentially (each `evaluate()` call fully completes
        # before the next starts), but scoring *within* a run is concurrent
        # across dataset items, so the mock can't rely on a flat call-order
        # list. Key scores per-question instead, consumed in call order for
        # that question only -- that ordering is deterministic across runs.
        ev = _evaluator()
        metric = MagicMock()
        metric.name = "noisy"
        scores_by_question = {"q0": [0.5, 0.7, 0.9], "q1": [0.6, 0.6, 0.6]}
        call_index = {"q0": 0, "q1": 0}

        def score_fn(row, backend):
            q = row["question"]
            v = scores_by_question[q][call_index[q]]
            call_index[q] += 1
            return v

        metric.score.side_effect = score_fn
        ev.add_metric(metric)

        report = measure_reliability(ev, _dataset(2), runs=3)

        stats = report["metrics"]["noisy"]
        sample0_vals = [0.5, 0.7, 0.9]
        sample1_vals = [0.6, 0.6, 0.6]
        expected_mean_stdev = statistics.mean(
            [statistics.stdev(sample0_vals), statistics.stdev(sample1_vals)]
        )
        assert stats["mean_stdev"] == pytest.approx(round(expected_mean_stdev, 4), abs=1e-4)
        assert stats["max_stdev"] == pytest.approx(round(statistics.stdev(sample0_vals), 4), abs=1e-4)
        assert stats["samples_measured"] == 2

    def test_disables_cache_during_measurement_and_restores_it(self):
        ev = _evaluator()
        ev.cache = "sentinel-cache"
        metric = MagicMock()
        metric.name = "m"
        metric.score.return_value = 0.5
        ev.add_metric(metric)

        measure_reliability(ev, _dataset(1), runs=2)

        assert ev.cache == "sentinel-cache"

    def test_errored_samples_excluded_from_variance(self):
        ev = _evaluator()
        metric = MagicMock()
        metric.name = "flaky"
        # 1 sample, 3 runs = 3 calls; the middle run fails for that sample.
        metric.score.side_effect = [0.5, RuntimeError("boom"), 0.9]
        ev.add_metric(metric)

        report = measure_reliability(ev, _dataset(1), runs=3)

        stats = report["metrics"]["flaky"]
        # only 2 of 3 runs produced a usable score for the one sample, but
        # 2 is still enough to compute a stdev over.
        assert stats["samples_measured"] == 1
        assert stats["mean_stdev"] == pytest.approx(statistics.stdev([0.5, 0.9]), abs=1e-4)

if __name__ == "__main__":
    pytest.main([__file__])
