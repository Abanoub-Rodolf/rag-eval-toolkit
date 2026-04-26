# Changelog

All notable changes to rag-eval-toolkit are documented here.

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
- Score parser now uses a shared `_parse_score()` regex across all metrics with robust handling of LLM response formats ("Score: 0.8", "0.85 (very confident)", etc.)
- Backends are lazy-loaded; importing the package no longer triggers SDK imports
- Cache is now opt-in by default (was previously always-on at `.rag_eval_cache.db`)
- Default models updated: OpenAI → `gpt-4o`, Anthropic → `claude-sonnet-4-6`, LiteLLM → `gpt-4o`
- CLI exits with non-zero codes on errors (was silently logging and continuing)
- Gemini backend migrated from `google-generativeai` to `google-genai` SDK
