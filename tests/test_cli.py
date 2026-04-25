import json
import os
import pytest
import yaml
from click.testing import CliRunner
from unittest.mock import MagicMock, patch

from rag_eval.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def dataset_file(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"question":"What is RAG?","context":"RAG is Retrieval-Augmented Generation.","answer":"It augments LLMs with retrieval."}\n')
    return str(f)


def _mock_backend():
    b = MagicMock()
    b.model = "test-model"
    b.generate.return_value = "0.85"
    return b


class TestMetricsCommand:
    def test_lists_all_metrics(self, runner):
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        for name in ("faithfulness", "hallucination", "toxicity", "coherence", "groundedness"):
            assert name in result.output

    def test_shows_13_metrics(self, runner):
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        # 13 metric rows should be present
        assert result.output.count("\n") >= 13


class TestInitCommand:
    def test_creates_config_file(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert os.path.exists("eval_config.yaml")

    def test_config_is_valid_yaml(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(main, ["init"])
            with open("eval_config.yaml") as f:
                cfg = yaml.safe_load(f)
            assert "dataset" in cfg
            assert "backend" in cfg
            assert "metrics" in cfg
            assert isinstance(cfg["metrics"], list)


class TestRunCommand:
    def test_missing_dataset_exits_1(self, runner):
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 1

    def test_missing_dataset_shows_error(self, runner):
        result = runner.invoke(main, ["run"])
        assert "error" in result.output.lower() or "dataset" in result.output.lower()

    def test_bad_backend_init_exits_1(self, runner, dataset_file):
        bad_cls = MagicMock(side_effect=ValueError("OPENAI_API_KEY not set"))
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": bad_cls}):
            result = runner.invoke(main, ["run", "--dataset", dataset_file, "--backend", "openai"])
        assert result.exit_code == 1

    def test_bad_backend_shows_error(self, runner, dataset_file):
        bad_cls = MagicMock(side_effect=ValueError("OPENAI_API_KEY not set"))
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": bad_cls}):
            result = runner.invoke(main, ["run", "--dataset", dataset_file, "--backend", "openai"])
        assert "error" in result.output.lower()

    def test_all_unknown_metrics_exits_1(self, runner, dataset_file, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"dataset: {dataset_file}\nbackend: openai\nmetrics:\n  - not_a_real_metric\n")
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            result = runner.invoke(main, ["run", "--config", str(cfg)])
        assert result.exit_code == 1

    def test_run_success_exits_0(self, runner, dataset_file, tmp_path):
        out = str(tmp_path / "results.json")
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            result = runner.invoke(main, ["run", "--dataset", dataset_file, "--backend", "openai", "--output", out])
        assert result.exit_code == 0

    def test_run_writes_output_file(self, runner, dataset_file, tmp_path):
        out = str(tmp_path / "results.json")
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            runner.invoke(main, ["run", "--dataset", dataset_file, "--backend", "openai", "--output", out])
        assert os.path.exists(out)
        data = json.loads(open(out).read())
        assert "averages" in data
        assert "per_sample" in data

    def test_run_with_yaml_config(self, runner, tmp_path):
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"question":"Q","context":"C","answer":"A"}\n')
        out = str(tmp_path / "out.json")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"dataset: {str(ds)}\nbackend: openai\n"
            f"metrics:\n  - faithfulness\noutput: {out}\n"
        )
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            result = runner.invoke(main, ["run", "--config", str(cfg)])
        assert result.exit_code == 0
        assert os.path.exists(out)

    def test_run_unknown_metric_warns_but_valid_proceed(self, runner, tmp_path):
        """One bad metric + one good metric: warn about bad, succeed."""
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"question":"Q","context":"C","answer":"A"}\n')
        out = str(tmp_path / "out.json")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"dataset: {str(ds)}\nbackend: openai\n"
            f"metrics:\n  - faithfulness\n  - not_real\noutput: {out}\n"
        )
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            result = runner.invoke(main, ["run", "--config", str(cfg)])
        assert result.exit_code == 0
        assert "warning" in result.output.lower() or "skipped" in result.output.lower()

    def test_run_bad_dataset_file_exits_1(self, runner, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            result = runner.invoke(main, ["run", "--dataset", str(bad_file), "--backend", "openai"])
        assert result.exit_code == 1

    def test_cache_path_flag(self, runner, dataset_file, tmp_path):
        out = str(tmp_path / "results.json")
        cache = str(tmp_path / "cache.db")
        with patch.dict("rag_eval.cli.BACKEND_MAP", {"openai": MagicMock(return_value=_mock_backend())}):
            result = runner.invoke(main, [
                "run", "--dataset", dataset_file, "--backend", "openai",
                "--output", out, "--cache-path", cache,
            ])
        assert result.exit_code == 0
        assert os.path.exists(cache)
