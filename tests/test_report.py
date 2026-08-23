import os

import pytest

from rag_eval.report.generator import HTMLReportGenerator, generate_html_report


@pytest.fixture
def sample_results():
    return {
        "averages": {"faithfulness": 0.85, "coherence": 0.72},
        "per_sample": {
            "faithfulness": [0.9, 0.8],
            "coherence": [0.7, 0.74],
        },
    }


class TestGenerateHtmlReport:
    def test_creates_file(self, sample_results, tmp_path):
        out = str(tmp_path / "report.html")
        generate_html_report(sample_results, out)
        assert os.path.exists(out)

    def test_contains_metric_names(self, sample_results, tmp_path):
        out = str(tmp_path / "report.html")
        generate_html_report(sample_results, out)
        content = open(out).read()
        assert "faithfulness" in content
        assert "coherence" in content

    def test_contains_scores(self, sample_results, tmp_path):
        out = str(tmp_path / "report.html")
        generate_html_report(sample_results, out)
        content = open(out).read()
        assert "0.8500" in content
        assert "0.7200" in content

    def test_xss_script_tag_escaped(self, tmp_path):
        """User data with </pre><script> must not appear unescaped in output."""
        results = {
            "averages": {"faithfulness": 0.5},
            "per_sample": {"faithfulness": [0.5]},
            "_raw_question": '</pre><script>alert(1)</script>',
        }
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "<script>alert(1)</script>" not in content
        assert "&lt;script&gt;" in content

    def test_xss_angle_brackets_escaped(self, tmp_path):
        results = {
            "averages": {"m": 0.5},
            "per_sample": {"m": [0.5]},
            "data": "<img src=x onerror=alert(1)>",
        }
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "<img src=x" not in content

    def test_empty_averages(self, tmp_path):
        out = str(tmp_path / "report.html")
        generate_html_report({"averages": {}, "per_sample": {}}, out)
        content = open(out).read()
        assert "<!DOCTYPE html>" in content

    def test_html_report_generator_class(self, sample_results, tmp_path):
        out = str(tmp_path / "report.html")
        with pytest.warns(DeprecationWarning):
            gen = HTMLReportGenerator()
        gen.generate(sample_results, out)
        assert os.path.exists(out)

    def test_xss_metric_name_in_summary_table(self, tmp_path):
        results = {
            "averages": {"<script>alert(1)</script>": 0.5},
            "per_sample": {"<script>alert(1)</script>": [0.5]},
        }
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "<script>alert(1)</script>" not in content
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content

    def test_xss_metric_name_in_per_sample_header(self, tmp_path):
        results = {
            "averages": {"normal": 0.5},
            "per_sample": {"<img onerror=x>": [0.5]},
        }
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "<img onerror=x>" not in content
        assert "&lt;img onerror=x&gt;" in content

    def test_score_bar_clamps_overflow(self, tmp_path):
        results = {"averages": {"m": 1.5}, "per_sample": {"m": [1.5]}}
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "width:150px" not in content
        assert "width:100px" in content

    def test_none_per_sample_score_renders_as_error_marker(self, tmp_path):
        """A None entry (metric errored on that sample) must render, not crash."""
        results = {
            "averages": {"faithfulness": 0.9},
            "per_sample": {"faithfulness": [0.9, None]},
        }
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "<em>error</em>" in content
        assert "0.9000" in content

    def test_score_bar_clamps_negative(self, tmp_path):
        results = {"averages": {"m": -0.3}, "per_sample": {"m": [-0.3]}}
        out = str(tmp_path / "report.html")
        generate_html_report(results, out)
        content = open(out).read()
        assert "width:-30px" not in content
        assert "width:0px" in content


class TestHTMLReportGeneratorDeprecation:
    def test_generate_emits_deprecation_warning(self, sample_results, tmp_path):
        import warnings

        from rag_eval.report.generator import HTMLReportGenerator

        out = str(tmp_path / "report.html")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HTMLReportGenerator().generate(sample_results, out)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
