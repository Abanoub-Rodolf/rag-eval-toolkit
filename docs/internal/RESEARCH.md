# RAG Evaluation Toolkit v1.0 - Research Summary (2025-2026)

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
*   **v1.0 Goal:** Include "Evaluation Traces": log the exact prompt sent to the LLM judge and its raw reasoning/explanation string alongside the score.

### **B. Local-First & Cost Efficiency**
DeepEval and Ragas are expensive to run at scale with GPT-4o/5.
*   **v1.0 Goal:** First-class support for **Ollama** and local embedding models. Use **Caching** (hash-based) to prevent re-evaluating identical inputs.

### **C. CLI-First Developer Experience**
Ragas and TruLens can be cumbersome to set up in a script.
*   **v1.0 Goal:** A dependable CLI with `rag-eval init`, `rag-eval run --config eval.yaml`, and `rag-eval compare`.

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

## 5. Addendum: v2.0 research pass (2026-07-27)

Revisited before the v2.0 release. Findings that changed the implementation:

- **Ragas context precision is rank-aware** (average precision over chunks
  in retrieval order, from the Ragas paper, arXiv:2309.15217), not a single
  holistic float. `ContextPrecisionMetric` now computes this when `context`
  is a list of chunks.
- **Ragas context recall decomposes the ground truth into statements** and
  checks each against the context. `ContextRecallMetric` now does this
  instead of asking for one number.
- **DeepEval's published recommended thresholds** (2026): faithfulness 0.75,
  answer relevancy 0.8, context precision 0.7, context recall 0.8. Not wired
  into this toolkit as defaults (thresholds are dataset-dependent), but
  useful as a sanity check when interpreting scores.
- **LLM-judge bias literature** is converging on three well-characterized
  biases: position bias (pairwise comparisons only, doesn't apply to this
  toolkit's pointwise scoring), verbosity bias, and self-preference bias.
  Mitigation via chain-of-thought reasoning before the verdict is
  well-supported; mitigation via randomized ordering only applies to
  pairwise setups.
- **Ragas' own validation numbers**: 95%/78%/70% agreement with human
  annotators for faithfulness/answer relevancy/context relevance on their
  WikiEval benchmark (50 Wikipedia pages, human-annotated). This toolkit has
  no equivalent number; see README Limitations.
- **Benchmarks confirmed to still exist and be relevant**: RAGTruth (ACL
  2024, ~18k span-annotated hallucination examples), HaluEval, LLM-AggreFact,
  ExpertQA (arXiv:2309.07852, 32-field expert-curated long-form QA), WikiEval
  (Ragas' own benchmark, arXiv:2309.15217). None of these are bundled with
  this toolkit; they're reference points for anyone building a validation
  set against it.
- Chain-of-thought before a numeric verdict (reason briefly, then emit a
  labeled `SCORE:` line) is the single best-supported lever for LLM-judge
  reliability in the current literature; the v1.0 "output only a float"
  prompts were the weakest version of an LLM-judge prompt. All 13 judge
  prompts were rewritten around this.
