# RAG Evaluation Toolkit v1.0 — Research Summary (2025-2026)

This document summarizes the current landscape of RAG evaluation tools and identifies the gaps that `rag-eval-toolkit` v1.0 aims to fill.

## 1. Competitor Analysis

| Framework | Focus | Strengths | Weaknesses (Pain Points) |
| :--- | :--- | :--- | :--- |
| **Ragas** | RAG Metrics | Industry standard "RAG Triad", reference-free. | Maintenance lag, opaque metrics (black box), poor DX, "dead repo" perception. |
| **DeepEval** | Unit Testing / CI | Pytest-native, 50+ metrics, great DX/CLI. | Expensive (heavy LLM use), ecosystem lock-in (Confident AI), high latency. |
| **TruLens** | Observability | GPA framework for agents, deep tracing. | Stability issues (OTEL), dashboard bugs, "out-of-context" failures. |
| **Phoenix** | Monitoring | Open-source, OTel-based, vendor-neutral. | Steeper learning curve, focused more on observability than pure eval. |
| **promptfoo** | Security | CLI-first, red-teaming, YAML configs. | Focused on safety/security over detailed RAG metrics. |

## 2. Market Gaps & Opportunities

### **A. Transparency & Debuggability**
Most tools provide a score (0.0-1.0) without explaining *why*.
*   **v1.0 Goal:** Include "Evaluation Traces" — log the exact prompt sent to the LLM judge and its raw reasoning/explanation string alongside the score.

### **B. Local-First & Cost Efficiency**
DeepEval and Ragas are expensive to run at scale with GPT-4o/5.
*   **v1.0 Goal:** First-class support for **Ollama** and local embedding models. Use **Caching** (hash-based) to prevent re-evaluating identical inputs.

### **C. CLI-First Developer Experience**
Ragas and TruLens can be cumbersome to set up in a script.
*   **v1.0 Goal:** A robust CLI with `rag-eval init`, `rag-eval run --config eval.yaml`, and `rag-eval compare`.

### **D. Metric Quality & "RAG Triad"**
Current v0.1 only has basic Faithfulness and Relevancy.
*   **v1.0 Goal:** Complete the "RAG Triad" (Context Precision/Recall) and add Groundedness with citation tracking.

## 3. Targeted v1.0 Feature Set

### **New Metrics**
- **Hallucination Detection:** Binary with evidence extraction.
- **Groundedness:** Verified citations against context.
- **Context Precision/Recall:** For retrieval evaluation.
- **Toxicity/Safety:** For production guardrails.
- **Latency/Cost:** Business metrics for optimization.

### **Core Features**
- **Async & Batch:** Parallel evaluation for speed.
- **Caching:** SQLite-backed or file-based cache.
- **YAML Config:** Reproducible evaluation runs.
- **Comparison Mode:** Side-by-side pipeline benchmarking.
- **LiteLLM Support:** One backend to rule them all (100+ providers).

## 4. Academic & Industry Standards
- **RAGAS Paper:** Defined Faithfulness, Answer Relevancy, Context Precision/Recall.
- **ARES:** Automated Readability Index for RAG.
- **NLI (Natural Language Inference):** The gold standard for Groundedness/Entailment.

---
*Research conducted: March 25, 2026*
