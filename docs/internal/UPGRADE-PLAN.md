# RAG Evaluation Toolkit v1.0 — Upgrade Plan

## 1. Metric Roadmap (Prioritized)

| Metric | Target | Description | Implementation |
| :--- | :--- | :--- | :--- |
| **Hallucination** | Answer -> Context | Binary detect + reasoning. | NLI judge prompt. |
| **Context Precision** | Context -> Ground Truth | Retrieval quality. | LLM-based relevancy check per chunk. |
| **Context Recall** | Context -> Ground Truth | Retrieval coverage. | LLM-based coverage check. |
| **Groundedness** | Answer -> Context | Citation-level verification. | Evidence extraction + matching. |
| **Semantic Sim** | Answer -> Ground Truth | No-LLM baseline. | `sentence-transformers` embeddings. |
| **Toxicity / Safety** | Answer | Guardrails. | Pre-defined safety rubric judge. |
| **BLEU / ROUGE** | Answer -> Ground Truth | Classical NLP metrics. | `nltk` or `evaluate`. |

## 2. Backend Support
- **LiteLLM (High Priority):** Universal proxy for 100+ providers.
- **Ollama (High Priority):** Local-first development.
- **Google Gemini:** Direct SDK support.
- **Together AI / Groq:** Fast inference providers.

## 3. Core Features (Execution Engine)
- **Async Engine:** `asyncio` for parallel evaluation.
- **Caching:** SQLite-backed hash cache for (prompt, model) pairs.
- **Dataset Loaders:** CSV, JSONL, HuggingFace.
- **YAML Config:** `rag-eval run --config eval.yaml`
- **Tracing:** Capture LLM judge reasoning string and raw prompt.

## 4. CLI Revamp
- `rag-eval run --config config.yaml` (main entry point)
- `rag-eval init` (interactive scaffolding)
- `rag-eval metrics --list` (catalog of available metrics)
- `rag-eval compare run1.json run2.json` (diff reports)
- `rag-eval datasets` (fetch sample RAG datasets)

## 5. Quality & Packaging
- **Type Safety:** Strict `mypy` enforcement.
- **Linting:** `ruff` (replaces black, isort, flake8).
- **CI/CD:** GitHub Actions (Python 3.9-3.12).
- **Packaging:** Pure `pyproject.toml` (remove `setup.py`).
- **Optional Dependencies:** `rag-eval-toolkit[openai]`, `[ollama]`, etc.

## 6. Success Criteria
- [ ] 10+ metrics implemented.
- [ ] 3+ new backends.
- [ ] Async execution support.
- [ ] Config-based CLI runs.
- [ ] 80%+ test coverage.
- [ ] All tests passing on Python 3.9-3.12.
