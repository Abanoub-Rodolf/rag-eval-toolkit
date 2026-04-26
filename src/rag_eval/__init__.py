"""rag-eval-toolkit: production-grade framework for evaluating RAG pipelines."""

__version__ = "1.0.0"

from rag_eval.evaluator import RAGEvaluator
from rag_eval.backends import (
    BaseBackend,
    OpenAIBackend,
    AnthropicBackend,
    OllamaBackend,
    GeminiBackend,
    LiteLLMBackend,
)
from rag_eval.metrics import (
    BaseMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    HallucinationMetric,
    ToxicityMetric,
    CoherenceMetric,
    ConcisenessMetric,
    AnswerCompletenessMetric,
    ChunkAttributionMetric,
    ContextUtilizationMetric,
    GroundednessMetric,
    SemanticSimilarityMetric,
)
from rag_eval.utils import load_dataset, format_score, save_results_to_csv

__all__ = [
    "RAGEvaluator",
    "BaseBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "GeminiBackend",
    "LiteLLMBackend",
    "BaseMetric",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "HallucinationMetric",
    "ToxicityMetric",
    "CoherenceMetric",
    "ConcisenessMetric",
    "AnswerCompletenessMetric",
    "ChunkAttributionMetric",
    "ContextUtilizationMetric",
    "GroundednessMetric",
    "SemanticSimilarityMetric",
    "load_dataset",
    "format_score",
    "save_results_to_csv",
]
