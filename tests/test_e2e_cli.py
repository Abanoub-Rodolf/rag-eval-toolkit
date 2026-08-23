"""End-to-end tests: real HTTP over localhost through the full stack.

Unit tests mock requests/SDK objects; these bind a real socket and exercise
click -> OllamaBackend -> RAGEvaluator -> cache -> report with real JSON on
both sides, so wire-format mistakes unit mocks cannot see still surface.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from rag_eval.backends.ollama_backend import OllamaBackend
from rag_eval.cli import main
from rag_eval.evaluator import RAGEvaluator
from rag_eval.metrics.faithfulness import FaithfulnessMetric
from rag_eval.metrics.semantic_similarity import SemanticSimilarityMetric
from rag_eval.reliability import measure_reliability


class FakeOllama:
    """Minimal stand-in for a real Ollama server.

    Serves POST /api/generate and POST /api/embeddings, counts requests so
    tests can prove cache hits, and can be toggled to fail embedding calls.
    """

    def __init__(self):
        self.generate_requests = 0
        self.embed_requests = 0
        self.fail_embeds = False
        self.score_text = "The answer matches the context.\nSCORE: 0.9"
        self.prompts_seen = []

    def handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/api/generate":
                    outer.generate_requests += 1
                    outer.prompts_seen.append(body.get("prompt", ""))
                    self._json({"response": outer.score_text})
                elif self.path == "/api/embeddings":
                    outer.embed_requests += 1
                    if outer.fail_embeds:
                        self._json({"error": "embedding model not loaded"}, status=500)
                    else:
                        self._json({"embedding": [0.5, 0.5, 0.0]})
                else:
                    self._json({"error": "not found"}, status=404)

            def _json(self, payload, status=200):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        return Handler


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    server.base_url = f"http://127.0.0.1:{httpd.server_port}"
    yield server
    httpd.shutdown()
    httpd.server_close()


DATASET = [
    {
        "question": "What is RAG?",
        "context": "RAG is Retrieval-Augmented Generation.",
        "answer": "RAG combines retrieval with generation.",
        "ground_truth": "Retrieval-augmented generation.",
    },
    {
        "question": "Capital of France?",
        "context": "Paris is the capital of France.",
        "answer": "Paris.",
        "ground_truth": "Paris.",
    },
]


class TestPublicApiEndToEnd:
    def test_full_evaluation_over_real_http(self, fake_ollama, tmp_path):
        backend = OllamaBackend(base_url=fake_ollama.base_url)
        evaluator = RAGEvaluator(backend)
        evaluator.add_metric(FaithfulnessMetric())
        evaluator.add_metric(SemanticSimilarityMetric())

        results = evaluator.evaluate(DATASET)

        assert set(results["averages"]) == {"faithfulness", "semantic_similarity"}
        # both metrics scored both samples successfully
        assert results["errors"] == {"faithfulness": 0, "semantic_similarity": 0}
        assert all(s is not None for scores in results["per_sample"].values() for s in scores)
        # only the judge metric hits /api/generate (2 rows); embeddings go to
        # /api/embeddings and are counted separately
        assert len(fake_ollama.prompts_seen) == 2
        assert fake_ollama.embed_requests == 4
        assert any("faithful" in p.lower() for p in fake_ollama.prompts_seen)

    def test_failed_embedding_excluded_not_zeroed(self, fake_ollama):
        fake_ollama.fail_embeds = True
        backend = OllamaBackend(base_url=fake_ollama.base_url)
        evaluator = RAGEvaluator(backend)
        evaluator.add_metric(SemanticSimilarityMetric())

        results = evaluator.evaluate(DATASET)

        assert results["errors"]["semantic_similarity"] == 2
        assert "semantic_similarity" not in results["averages"]
        assert all(s is None for s in results["per_sample"]["semantic_similarity"])

    def test_cache_prevents_second_wire_roundtrip(self, fake_ollama, tmp_path):
        cache_path = str(tmp_path / "cache.db")
        backend = OllamaBackend(base_url=fake_ollama.base_url)

        first = RAGEvaluator(backend, cache_path=cache_path)
        first.add_metric(FaithfulnessMetric())
        first_results = first.evaluate(DATASET)
        assert fake_ollama.generate_requests == 2

        second_backend = OllamaBackend(base_url=fake_ollama.base_url)
        second = RAGEvaluator(second_backend, cache_path=cache_path)
        second.add_metric(FaithfulnessMetric())
        cached_results = second.evaluate(DATASET)
        # zero new judge calls: every score came from SQLite
        assert fake_ollama.generate_requests == 2
        assert cached_results["averages"] == first_results["averages"]

    def test_report_written_from_live_results(self, fake_ollama, tmp_path):
        out = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--dataset", str(_write_dataset(tmp_path)),
                "--backend", "ollama",
                "--base-url", fake_ollama.base_url,
                "--output", str(out),
                "--report",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out.read_text())
        assert data["averages"]["faithfulness"] == 0.9
        html = out.with_suffix(".html")
        assert html.exists()
        assert "faithfulness" in html.read_text()


class TestCliEndToEnd:
    def test_run_command_over_real_http(self, fake_ollama, tmp_path):
        out = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--dataset", str(_write_dataset(tmp_path)),
                "--backend", "ollama",
                "--base-url", fake_ollama.base_url,
                "--output", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Evaluation Summary" in result.output
        assert json.loads(out.read_text())["errors"]["faithfulness"] == 0

    def test_reliability_command_over_real_http(self, fake_ollama, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "reliability",
                "--dataset", str(_write_dataset(tmp_path)),
                "--backend", "ollama",
                "--base-url", fake_ollama.base_url,
                "--runs", "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Judge Reliability" in result.output
        # deterministic stub judge => zero variance
        assert "0.0000" in result.output


class TestReliabilityEndToEnd:
    def test_measure_reliability_two_runs(self, fake_ollama):
        backend = OllamaBackend(base_url=fake_ollama.base_url)
        evaluator = RAGEvaluator(backend)
        evaluator.add_metric(FaithfulnessMetric())

        report = measure_reliability(evaluator, DATASET, runs=2)

        assert report["runs"] == 2
        stats = report["metrics"]["faithfulness"]
        assert stats["samples_measured"] == 2
        assert stats["mean_stdev"] == 0.0
        # 2 runs x 2 samples = 4 judge calls total (no cache attached)
        assert fake_ollama.generate_requests == 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path):
    ds = tmp_path / "data.jsonl"
    lines = [json.dumps(row) for row in DATASET]
    ds.write_text("\n".join(lines) + "\n")
    return str(ds)
