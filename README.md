# RAG Eval Toolkit (v1.0.0)

Production-grade toolkit for evaluating RAG (Retrieval-Augmented Generation) pipelines.

## Key Features

- **Multi-Metric Suite:** 10+ research-backed metrics (Faithfulness, Relevancy, Hallucination, Groundedness, etc.).
- **Async Engine:** Parallel evaluation for massive datasets.
- **Local-First:** Native support for **Ollama** and LiteLLM.
- **Smart Caching:** SQLite-backed hashing to save tokens and time.
- **CLI-First DX:** `rag-eval run --config eval.yaml` for reproducible workflows.
- **HTML Reports:** Beautiful, interactive visual results.

## Installation

```bash
pip install rag-eval-toolkit[all]
```

## Quick Start

### 1. Initialize Config
```bash
rag-eval init
```

### 2. Run Evaluation
```bash
rag-eval run --dataset my_data.jsonl --backend openai --model gpt-4
```

## Metrics

| Metric | Goal | Method |
| :--- | :--- | :--- |
| `faithfulness` | Grounding in context | LLM-as-judge |
| `hallucination` | Detect false claims | LLM-as-judge |
| `groundedness` | NLI-style entailment | LLM-as-judge |
| `context_utilization` | Context usage | LLM-as-judge |
| `answer_relevancy` | Query alignment | LLM-as-judge |
| `semantic_similarity` | Semantic overlap | Embeddings (No LLM) |
| `toxicity` | Safety & bias check | LLM-as-judge |

## Supported Backends

- **OpenAI** (`gpt-4`, `gpt-3.5-turbo`)
- **Anthropic** (`claude-3-opus`, `claude-3-sonnet`)
- **Google Gemini** (`gemini-1.5-flash`)
- **Ollama** (Local models: `llama3`, `mistral`)
- **LiteLLM** (Universal proxy for 100+ providers)

## Development by ThynkQ

Built for the community by [ThynkQ](https://thynkq.com).
