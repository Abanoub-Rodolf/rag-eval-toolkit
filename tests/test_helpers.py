import csv

from rag_eval.utils.helpers import flatten_context, format_score, save_results_to_csv


class TestFormatScore:
    def test_formats_fraction_as_percentage(self):
        assert format_score(0.856) == "85.60%"

    def test_handles_zero_and_one(self):
        assert format_score(0.0) == "0.00%"
        assert format_score(1.0) == "100.00%"


class TestSaveResultsToCsv:
    def test_writes_header_and_rows(self, tmp_path):
        out = tmp_path / "results.csv"
        rows = [{"name": "faithfulness", "score": 0.9}, {"name": "toxicity", "score": 0.1}]
        save_results_to_csv(rows, str(out))
        with open(out, newline="") as f:
            data = list(csv.DictReader(f))
        assert len(data) == 2
        assert data[0]["name"] == "faithfulness"

    def test_empty_results_writes_nothing(self, tmp_path):
        out = tmp_path / "empty.csv"
        save_results_to_csv([], str(out))
        assert not out.exists()


class TestFlattenContext:
    def test_joins_list_chunks_with_separator(self):
        assert flatten_context(["a", "b"]) == "a\n---\nb"

    def test_passes_string_through(self):
        assert flatten_context("solo chunk") == "solo chunk"

    def test_stringifies_non_string_chunks(self):
        assert flatten_context([1, 2]) == "1\n---\n2"

    def test_matches_cache_key_flattening(self):
        # cache._key uses the same helper: separator drift would silently miss cache hits
        from rag_eval.utils.cache import EvaluationCache

        cache = EvaluationCache(db_path=":memory:")
        try:
            row = {"question": "q", "context": ["c1", "c2"], "answer": "a", "ground_truth": "g"}
            cache.set("m", "model", row, 0.5)
            flat_row = {"question": "q", "context": "c1\n---\nc2", "answer": "a", "ground_truth": "g"}
            assert cache.get("m", "model", flat_row) == 0.5
        finally:
            cache.close()
