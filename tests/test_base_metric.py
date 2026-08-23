from unittest.mock import MagicMock

import pytest

from rag_eval.metrics.base import BaseMetric, ScoreParseError, _judge_prompt, _parse_score


class TestParseScore:
    def test_plain_float(self):
        assert _parse_score("0.85") == 0.85

    def test_integer_one(self):
        assert _parse_score("1") == 1.0

    def test_integer_zero(self):
        assert _parse_score("0") == 0.0

    def test_prefixed_score(self):
        # real LLM responses often look like this
        assert _parse_score("Score: 0.9") == 0.9

    def test_explanatory_text(self):
        assert _parse_score("I'd give this 0.8 out of 1.0") == 0.8

    def test_no_score_returns_none(self):
        assert _parse_score("I cannot determine a score here.") is None

    def test_empty_string_returns_none(self):
        assert _parse_score("") is None

    def test_out_of_range_value_ignored(self):
        # 1.5 is out of range; no in-range token present, so None
        assert _parse_score("1.5") is None

    def test_one_point_zero(self):
        assert _parse_score("1.0") == 1.0

    def test_whitespace(self):
        assert _parse_score("   0.75   ") == 0.75

    def test_prefers_labeled_score_over_reasoning_numbers(self):
        # reasoning may mention other numbers (e.g. "2 of 3 claims"); the
        # labeled SCORE line must win.
        text = "The answer makes 3 claims, 2 are supported.\nSCORE: 0.67"
        assert _parse_score(text) == 0.67

    def test_final_score_label(self):
        assert _parse_score("Reasoning here.\nFINAL_SCORE: 0.4") == 0.4

    def test_labeled_score_out_of_range_falls_back_to_scan(self):
        # if the label itself is bogus, fall back to scanning for any in-range number
        text = "SCORE: 5.0\nActually more like 0.3."
        assert _parse_score(text) == 0.3


class TestGenerateScore:
    def test_returns_parsed_score(self):
        class DummyMetric(BaseMetric):
            def score(self, row, backend):
                return self._generate_score(backend, "prompt")

        backend = MagicMock()
        backend.generate.return_value = "SCORE: 0.6"
        metric = DummyMetric(name="dummy")
        assert metric.score({}, backend) == 0.6

    def test_raises_score_parse_error_on_unparseable_response(self):
        class DummyMetric(BaseMetric):
            def score(self, row, backend):
                return self._generate_score(backend, "prompt")

        backend = MagicMock()
        backend.generate.return_value = "I refuse to answer."
        metric = DummyMetric(name="dummy")
        with pytest.raises(ScoreParseError):
            metric.score({}, backend)

    def test_backend_exception_propagates(self):
        class DummyMetric(BaseMetric):
            def score(self, row, backend):
                return self._generate_score(backend, "prompt")

        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("network error")
        metric = DummyMetric(name="dummy")
        with pytest.raises(RuntimeError, match="network error"):
            metric.score({}, backend)


class TestJudgePrompt:
    def test_includes_role_instructions_and_fields(self):
        prompt = _judge_prompt(
            role="test role",
            instructions="do the thing",
            fields={"answer": "hello"},
            score_meaning="0.0 = bad\n1.0 = good",
        )
        assert "test role" in prompt
        assert "do the thing" in prompt
        assert "<answer>hello</answer>" in prompt
        assert "SCORE:" in prompt
