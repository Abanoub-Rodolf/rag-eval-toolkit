# rag-eval-toolkit

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python toolkit for evaluating RAG (Retrieval-Augmented Generation) pipelines with LLM-as-judge metrics: faithfulness, hallucination, groundedness, relevancy, and more. Local-first via Ollama, with async execution and result caching.

## Why this exists

Ragas and DeepEval cover more ground and are the more mature choice for most teams. This toolkit exists for a narrower case: when you want to read and edit the exact judge prompt driving a score, without going through a metrics registry or a hosted dashboard. Every prompt lives in a single readable file under `src/rag_eval/metrics/`, asks the judge to reason before scoring, and is easy to fork.

It's also local-first: Ollama is a first-class backend, not an afterthought, so you can run a full evaluation without an API key or per-token cost.

If you need the full "RAG Triad" with production tracing, or 50+ metrics with a pytest-native CI story, use TruLens/Phoenix or DeepEval instead. See [Limitations](#limitations) before trusting these scores for anything high-stakes.

## Installation

```bash
# Core (no LLM backends, just the framework)
pip install rag-eval-toolkit

# With a specific backend
pip install rag-eval-toolkit[openai]
pip install rag-eval-toolkit[anthropic]
pip install rag-eval-toolkit[ollama]

# Everything
pip install rag-eval-toolkit[all]
```

Or from source:

```bash
git clone https://gitlab.com/abanoub.rodolf/rag-eval-toolkit.git
cd rag-eval-toolkit
pip install -e ".[all]"
```

## Quick start

### Python API

```python
from rag_eval import RAGEvaluator
from rag_eval.backends import OllamaBackend
from rag_eval.metrics import FaithfulnessMetric, HallucinationMetric, GroundednessMetric

backend = OllamaBackend(model="llama3")
evaluator = RAGEvaluator(backend=backend)

evaluator.add_metric(FaithfulnessMetric())
evaluator.add_metric(HallucinationMetric())
evaluator.add_metric(GroundednessMetric())

dataset = [
    {
        "question": "What is RAG?",
        "context": "RAG stands for Retrieval-Augmented Generation. It retrieves relevant documents and uses them to ground LLM responses.",
        "answer": "RAG is a technique that retrieves documents to ground AI responses.",
    }
]

results = evaluator.evaluate(dataset)
print(results["averages"])
# {'faithfulness': 0.92, 'hallucination': 0.95, 'groundedness': 0.88}
```

A sample that fails to score (backend error, or a judge response with no
parseable score) is excluded from `averages` rather than counted as a 0.0.
Check `results["errors"]` for per-metric failure counts and
`results["per_sample"]` for which samples came back `None`.

### CLI

```bash
# Generate a config template
rag-eval init

# Run evaluation from config
rag-eval run --config eval_config.yaml

# Run with CLI flags
rag-eval run --dataset data.jsonl --backend ollama --model llama3

# Point a self-hosted backend at a non-default server
rag-eval run --dataset data.jsonl --backend ollama --base-url http://192.168.1.20:11434

# List available metrics
rag-eval metrics

# Measure judge test-retest reliability (self-consistency, not accuracy)
rag-eval reliability --dataset data.jsonl --backend ollama --model llama3 --runs 5
```

## Metrics

13 evaluation metrics, each returning a float from 0.0 (worst) to 1.0 (best).

### RAG quality (LLM-as-judge)

| Metric | What it measures | Required fields |
|--------|-----------------|-----------------|
| `faithfulness` | Fraction of answer claims supported by the context | question, context, answer |
| `hallucination` | Presence of claims not in (or contradicting) the context | question, context, answer |
| `groundedness` | NLI-style entailment check, context as premise | context, answer |
| `context_precision` | Retrieved context relevance, rank-aware if context is a chunk list | question, context |
| `context_recall` | Ground-truth coverage in the context, statement-level | question, context, ground_truth |
| `context_utilization` | How much of the retrieved context was actually used | question, context, answer |
| `answer_relevancy` | Does the answer address the question? | question, answer |
| `completeness` | Does the answer cover every part of the question? | question, answer |
| `chunk_attribution` | Are claims attributed to the right source chunk? | context, answer |

`faithfulness`, `hallucination`, and `groundedness` ask a version of the same
underlying question under the terminology different frameworks use for it
(Ragas, DeepEval/Galileo, and TruLens/Azure AI, respectively). Keep the one
whose name matches what you're used to; running all three on the same sample
is redundant.

### Answer quality (LLM-as-judge)

| Metric | What it measures | Required fields |
|--------|-----------------|-----------------|
| `coherence` | Is the answer logically structured? | answer |
| `conciseness` | Is the answer free of filler? | answer |
| `toxicity` | Coarse safety/bias screen, not a moderation classifier | answer |

### Classical (no LLM needed)

| Metric | What it measures | Required fields |
|--------|-----------------|-----------------|
| `semantic_similarity` | Embedding cosine similarity | answer, ground_truth |

`semantic_similarity` requires a backend with `embed()`. `OpenAIBackend` and
`AnthropicBackend` don't implement it; use `OllamaBackend`, `GeminiBackend`,
or `LiteLLMBackend`.

## Backends

| Backend | Provider | Default model | Install |
|---------|----------|----------------|---------|
| `OpenAIBackend` | OpenAI | `gpt-5.1` | `pip install rag-eval-toolkit[openai]` |
| `AnthropicBackend` | Anthropic (Claude) | `claude-sonnet-5` | `pip install rag-eval-toolkit[anthropic]` |
| `OllamaBackend` | Local models (llama3, mistral) | `llama3` | `pip install rag-eval-toolkit[ollama]` |
| `GeminiBackend` | Google Gemini | `gemini-2.5-flash` | `pip install rag-eval-toolkit[gemini]` |
| `LiteLLMBackend` | 100+ providers via LiteLLM | `gpt-5.1` | `pip install rag-eval-toolkit[litellm]` |

## Dataset format

Each item in your dataset is a dictionary. Required keys depend on which metrics you use (see table above). A typical item:

```python
{
    "question": "What is retrieval-augmented generation?",
    "context": "RAG combines retrieval with LLM generation...",
    "answer": "RAG is a technique that...",
    "ground_truth": "RAG retrieves documents and generates grounded answers."  # optional
}
```

`context` can also be a list of chunk strings in retrieval order. Passing a
list unlocks rank-aware scoring for `context_precision`; a plain string is
scored holistically since there's no chunk boundary or rank to weight.

Supported formats: JSON, JSONL, CSV.

## Features

- **Async execution**: Metrics are scored in parallel using asyncio + thread pool, capped by `max_concurrency` (default 20)
- **Result caching**: SQLite-backed cache keyed on (metric, model, input hash), opt-in via `cache_path`
- **YAML config**: Reproducible evaluation runs via `eval_config.yaml`
- **HTML reports**: Visual report generation, zero extra deps
- **Rich CLI output**: Colored tables, progress bars, clear error messages
- **Honest failure modes**: a judge response with no parseable score raises rather than silently scoring 0.0; `RAGEvaluator` tracks and reports these separately from real low scores

## Custom metrics

```python
from rag_eval.metrics.base import BaseMetric

class MyMetric(BaseMetric):
    def __init__(self):
        super().__init__(name="my_metric")

    def score(self, row, backend) -> float:
        prompt = f"Evaluate this answer: {row['answer']}\nScore 0.0-1.0:"
        response = backend.generate(prompt)
        try:
            return max(0.0, min(1.0, float(response.strip())))
        except ValueError:
            return 0.0
```

## Limitations

- **Judge-to-human agreement has not been measured for this toolkit.** The
  Ragas paper reports 95%/78%/70% agreement with human annotators for
  faithfulness/answer relevancy/context relevance on their own WikiEval
  benchmark; that number is theirs, not ours. Nothing here has been
  validated against a human-labeled dataset. Treat scores as a fast,
  cheap signal for regression detection, not as ground truth. If you need a
  validated signal, budget for a small human-labeled sample and check your
  own judge-human agreement before relying on it, or use `rag-eval
  reliability` to at least confirm the judge is self-consistent first.
- **LLM judges have well-documented biases**: verbosity bias (rating longer
  answers higher regardless of quality) and self-preference bias (rating a
  model's own outputs more favorably) are the most relevant ones for
  single-response scoring like this toolkit does. Position bias mostly
  affects pairwise comparison setups, which this toolkit doesn't do.
- Several metrics (`faithfulness`, `answer_relevancy`, `context_precision`,
  `context_recall`) implement a single-call simplification of the
  Ragas/DeepEval definitions rather than a full claim-extraction pipeline.
  Where the simplification is significant, it's documented in the metric's
  module docstring.
- `toxicity` is a single LLM judge call, not a dedicated moderation
  classifier. Don't use it as the only safety gate in front of production
  traffic.

## Architecture

```
src/rag_eval/
    __init__.py          # Package root, version
    evaluator.py          # Core orchestrator (sync + async)
    reliability.py        # Judge test-retest measurement
    cli.py                # Click CLI (init, run, metrics, reliability)
    backends/              # LLM provider integrations
        base.py            # BaseBackend abstract class
        openai_backend.py
        anthropic_backend.py
        ollama_backend.py
        gemini_backend.py
        litellm_backend.py
    metrics/                # Evaluation metrics
        base.py             # BaseMetric, shared judge-prompt builder and score parser
        faithfulness.py
        relevancy.py
        hallucination.py
        groundedness.py
        context_precision.py
        context_recall.py
        ... (13 total)
    report/                # HTML report generation
    utils/                 # Caching, dataset loaders, helpers
```

## Contributing

PRs welcome. Open an issue first to discuss major changes. Run `PYTHONPATH=src pytest tests/ -v` before submitting.

## Author

**Abanoub Rodolf Boctor** - [GitLab](https://gitlab.com/abanoub.rodolf) - [LinkedIn](https://linkedin.com/in/abanoubrodolf)

## License

MIT
