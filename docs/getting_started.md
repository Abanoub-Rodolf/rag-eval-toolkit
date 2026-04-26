# Getting started with rag-eval-toolkit

## Installation

```bash
pip install rag-eval-toolkit[all]
```

Or from source:

```bash
git clone https://gitlab.com/abanoub.rodolf/rag-eval-toolkit.git
cd rag-eval-toolkit
pip install -e ".[all]"
```

## Quick start (Python)

```python
from rag_eval import RAGEvaluator
from rag_eval.backends import OllamaBackend
from rag_eval.metrics import FaithfulnessMetric, HallucinationMetric

backend = OllamaBackend(model="llama3")
evaluator = RAGEvaluator(backend=backend)
evaluator.add_metric(FaithfulnessMetric())
evaluator.add_metric(HallucinationMetric())

dataset = [
    {
        "question": "What is RAG?",
        "context": "RAG stands for Retrieval-Augmented Generation...",
        "answer": "RAG is a technique that combines retrieval with generation.",
    }
]

results = evaluator.evaluate(dataset)
print(results["averages"])
```

## Quick start (CLI)

```bash
# Create a config template
rag-eval init

# Edit eval_config.yaml with your dataset path and preferences

# Run evaluation
rag-eval run --config eval_config.yaml

# Or use CLI flags directly
rag-eval run --dataset data.jsonl --backend ollama --model llama3
```

## Available metrics

| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Is the answer grounded in the context? |
| `hallucination` | Does the answer make unsupported claims? |
| `groundedness` | NLI-style entailment check |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Is the retrieved context relevant? |
| `context_recall` | Does the context cover the ground truth? |
| `context_utilization` | How much context was actually used? |
| `completeness` | Does the answer cover all parts of the question? |
| `chunk_attribution` | Are claims attributed to correct chunks? |
| `coherence` | Is the answer logically structured? |
| `conciseness` | Is the answer appropriately brief? |
| `toxicity` | Safety and bias check |
| `semantic_similarity` | Embedding cosine similarity (no LLM) |

## Backends

| Backend | Env var | Install extra |
|---------|---------|---------------|
| OpenAI | `OPENAI_API_KEY` | `[openai]` |
| Anthropic | `ANTHROPIC_API_KEY` | `[anthropic]` |
| Ollama | (none, runs locally) | `[ollama]` |
| Google Gemini | `GOOGLE_API_KEY` | `[gemini]` |
| LiteLLM | (varies by provider) | `[litellm]` |

## Dataset format

JSON, JSONL, or CSV. Each row needs at minimum:

```json
{
    "question": "Your question",
    "context": "Retrieved context",
    "answer": "Generated answer"
}
```

Some metrics also use `ground_truth` (optional).

## Custom metrics

Inherit from `BaseMetric` and implement `score()`. See `examples/custom_metrics.py`.

## Caching

Pass `cache_path` to avoid re-evaluating identical inputs:

```python
evaluator = RAGEvaluator(backend=backend, cache_path=".rag_eval_cache.db")
```

The cache is SQLite-backed and keys on (metric name, model name, input hash).
