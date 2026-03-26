# rag-eval-toolkit

[![CI](https://github.com/rodolfboctor/rag-eval-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rodolfboctor/rag-eval-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source Python toolkit for evaluating RAG (Retrieval-Augmented Generation) pipelines. Uses LLM-as-judge to score faithfulness, hallucination, groundedness, relevancy, and more. Supports local models via Ollama, async execution, and result caching.

## Why this exists

Most RAG eval tools are either expensive to run (DeepEval, Ragas with GPT-4), opaque about how scores are computed, or painful to set up. This toolkit is transparent (every LLM judge prompt is readable in the source), local-first (Ollama support out of the box), and CLI-driven for reproducible evaluations.

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
git clone https://github.com/rodolfboctor/rag-eval-toolkit
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

### CLI

```bash
# Generate a config template
rag-eval init

# Run evaluation from config
rag-eval run --config eval_config.yaml

# Run with CLI flags
rag-eval run --dataset data.jsonl --backend ollama --model llama3

# List available metrics
rag-eval metrics
```

## Metrics

13 evaluation metrics, each returning a float from 0.0 (worst) to 1.0 (best).

### RAG quality (LLM-as-judge)

| Metric | What it measures | Required fields |
|--------|-----------------|-----------------|
| `faithfulness` | Is the answer grounded in the context? | question, context, answer |
| `hallucination` | Does the answer contain claims not in the context? | question, context, answer |
| `groundedness` | NLI-style entailment check | context, answer |
| `context_precision` | Is the retrieved context relevant to the question? | question, context |
| `context_recall` | Does the context cover the ground truth? | question, context, ground_truth |
| `context_utilization` | How much of the context was actually used? | question, context, answer |
| `answer_relevancy` | Does the answer address the question? | question, answer |
| `completeness` | Does the answer cover all parts of the question? | question, answer |
| `chunk_attribution` | Are claims correctly attributed to source chunks? | context, answer |

### Answer quality (LLM-as-judge)

| Metric | What it measures | Required fields |
|--------|-----------------|-----------------|
| `coherence` | Is the answer logically structured? | answer |
| `conciseness` | Is the answer appropriately brief? | answer |
| `toxicity` | Safety and bias check | answer |

### Classical (no LLM needed)

| Metric | What it measures | Required fields |
|--------|-----------------|-----------------|
| `semantic_similarity` | Embedding cosine similarity | answer, ground_truth |

## Backends

| Backend | Provider | Install |
|---------|----------|---------|
| `OpenAIBackend` | OpenAI (GPT-4, etc.) | `pip install rag-eval-toolkit[openai]` |
| `AnthropicBackend` | Anthropic (Claude) | `pip install rag-eval-toolkit[anthropic]` |
| `OllamaBackend` | Local models (llama3, mistral) | `pip install rag-eval-toolkit[ollama]` |
| `GeminiBackend` | Google Gemini | `pip install rag-eval-toolkit[gemini]` |
| `LiteLLMBackend` | 100+ providers via LiteLLM | `pip install rag-eval-toolkit[litellm]` |

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

Supported formats: JSON, JSONL, CSV.

## Features

- **Async execution**: Metrics are scored in parallel using asyncio + thread pool
- **Result caching**: SQLite-backed cache keyed on (metric, model, input hash). Skip re-evaluation of identical inputs.
- **YAML config**: Reproducible evaluation runs via `eval_config.yaml`
- **HTML reports**: Visual report generation (requires `jinja2`)
- **Rich CLI output**: Colored tables, progress bars, clear error messages

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

## Architecture

```
src/rag_eval/
    __init__.py          # Package root, version
    evaluator.py         # Core orchestrator (sync + async)
    cli.py               # Click CLI (init, run, metrics)
    backends/            # LLM provider integrations
        base.py          # BaseBackend abstract class
        openai_backend.py
        anthropic_backend.py
        ollama_backend.py
        gemini_backend.py
        litellm_backend.py
    metrics/             # Evaluation metrics
        base.py          # BaseMetric abstract class
        faithfulness.py
        relevancy.py
        hallucination.py
        groundedness.py
        context_precision.py
        context_recall.py
        ... (13 total)
    report/              # HTML report generation
    utils/               # Caching, dataset loaders, helpers
```

## Contributing

PRs welcome. Open an issue first to discuss major changes. Run `PYTHONPATH=src pytest tests/ -v` before submitting.

## Author

**Abanoub Rodolf Boctor** - [GitHub](https://github.com/rodolfboctor) - [LinkedIn](https://linkedin.com/in/abanoubrodolf)

## License

MIT
