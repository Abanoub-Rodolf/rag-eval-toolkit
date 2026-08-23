# Changelog

All notable changes to rag-eval-toolkit are documented here.

## [3.0.0] - 2026-08-22

### Changed (breaking)
- **Embedding failures now raise.** `LiteLLMBackend.embed()`,
  `GeminiBackend.embed()`, and `OllamaBackend.embed()` previously returned
  `[]` on API errors, which `SemanticSimilarityMetric` silently scored as a
  0.0 cosine (a failed embed was indistinguishable from "completely
  dissimilar"). All backends now raise like `generate()` does; failed samples
  are excluded from averages and counted under `results["errors"]`.
  (`SemanticSimilarityMetric` also rejects empty vectors defensively.)

### Deprecated
- `HTMLReportGenerator` is deprecated; call `generate_html_report()`
  directly. Instantiation emits `DeprecationWarning`.

### Changed
- Backend SDK imports are deferred to backend construction: importing
  `rag_eval` loads no LLM SDKs. The lazy-import proxies in
  `rag_eval.backends` were replaced with a standard module-level
  `__getattr__` (PEP 562).

## [2.0.0] - 2026-07

Judge quality and error-handling release, driven by a review against
Ragas/DeepEval's published metric definitions and current LLM-as-judge
reliability research.

### Changed (breaking)
- **Failed judgments no longer score 0.0.** A judge response with no
  parseable score (empty, refusal, truncated, wrong format) used to be
  silently recorded as the worst possible score, biasing averages downward
  with no visible signal. `BaseMetric.score()` now raises `ScoreParseError`
  (backend call failures propagate as-is too). `RAGEvaluator` catches this,
  excludes the sample from `results["averages"]`, and reports the count
  under the new `results["errors"]` key. `results["per_sample"]` now
  contains `None` at the index of any sample that failed to score, instead
  of `0.0`.
- `ContextRecallMetric` and `SemanticSimilarityMetric` now raise `ValueError`
  when `ground_truth` is missing, instead of proceeding with an empty string
  or silently returning 0.0.
- `SemanticSimilarityMetric` raises `BackendCapabilityError` when the backend
  has no `embed()` method, instead of silently scoring every sample 0.0.
  `OpenAIBackend` and `AnthropicBackend` do not implement `embed()`; use
  `OllamaBackend`, `GeminiBackend`, or `LiteLLMBackend`.
- Default judge models refreshed: OpenAI `gpt-4o` -> `gpt-5.1` (retired),
  Anthropic `claude-sonnet-4-6` -> `claude-sonnet-5`, Gemini `gemini-1.5-flash`
  -> `gemini-2.5-flash` (deprecated generation).

### Added
- Every judge prompt now asks for brief reasoning before a labeled
  `SCORE: 0.x` verdict instead of a bare number. Chain-of-thought before
  scoring is the best-documented lever for LLM-judge reliability; parsing
  still falls back to scanning for a bare float for judges that ignore the
  label.
- `ContextPrecisionMetric` now computes Ragas-style rank-aware average
  precision when `context` is passed as a list of chunks in retrieval order,
  instead of a single holistic float. Falls back to holistic scoring for
  plain-string context or if the judge doesn't follow the per-chunk format.
- `ContextRecallMetric` now decomposes `ground_truth` into individual
  statements and checks each against the context, instead of asking for one
  holistic float. Falls back to holistic scoring if decomposition parsing
  fails.
- `rag_eval.reliability.measure_reliability()` and `rag-eval reliability`
  CLI command: run a dataset through the same judge N times and report
  per-metric score variance across runs (test-retest self-consistency, not
  judge-human agreement -- see README Limitations).
- CLI `run` now prints a warning per metric with excluded samples instead of
  silently folding errors into the score.

### Notes for upgraders from 1.0
- If you call `metric.score()` directly (not through `RAGEvaluator`), wrap it
  or catch `ScoreParseError` / `ValueError` -- it can now raise where it used
  to return `0.0`.
- If you consume `results["per_sample"]` directly, handle `None` entries.

## [1.0.0] - 2026-04

First production release.

### Added
- 13 evaluation metrics: faithfulness, answer_relevancy, context_precision, context_recall, hallucination, toxicity, coherence, conciseness, completeness, chunk_attribution, context_utilization, groundedness, semantic_similarity
- 5 lazy-loaded backends: OpenAI, Anthropic, Ollama, Google Gemini, LiteLLM
- Async evaluation with configurable concurrency (asyncio + thread pool)
- SQLite-backed result cache (opt-in, keyed on metric/model/input hash)
- CLI commands: `init`, `metrics`, `run` with YAML config support
- HTML report generation with XSS-safe rendering (stdlib only, no jinja2)
- Top-level package exports for backends, metrics, evaluator, and dataset loaders
- 117 tests covering metrics, backends, CLI, cache, loaders, and report generation
- Type hints across the public API
- Custom metric support via `BaseMetric` subclassing

### Public API
```python
from rag_eval import (
    RAGEvaluator,
    OpenAIBackend, AnthropicBackend, OllamaBackend, GeminiBackend, LiteLLMBackend,
    FaithfulnessMetric, AnswerRelevancyMetric,  # ... 13 metrics
    BaseMetric, BaseBackend,
    load_dataset,
)
```

### Notes for upgraders from v0.1
- Score parser now uses a shared `_parse_score()` regex across all metrics, handling common LLM response formats ("Score: 0.8", "0.85 (very confident)", etc.)
- Backends are lazy-loaded; importing the package no longer triggers SDK imports
- Cache is now opt-in by default (was previously always-on at `.rag_eval_cache.db`)
- Default models updated: OpenAI → `gpt-4o`, Anthropic → `claude-sonnet-4-6`, LiteLLM → `gpt-4o`
- CLI exits with non-zero codes on errors (was silently logging and continuing)
- Gemini backend migrated from `google-generativeai` to `google-genai` SDK
