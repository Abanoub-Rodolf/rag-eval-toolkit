import pytest
from rag_eval.metrics.base import _parse_score, BaseMetric


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

    def test_no_score_returns_zero(self):
        assert _parse_score("I cannot determine a score here.") == 0.0

    def test_empty_string(self):
        assert _parse_score("") == 0.0

    def test_out_of_range_value_ignored(self):
        # 1.5 is out of range; should return 0.0 (no [0,1] token found)
        assert _parse_score("1.5") == 0.0

    def test_one_point_zero(self):
        assert _parse_score("1.0") == 1.0

    def test_whitespace(self):
        assert _parse_score("   0.75   ") == 0.75
