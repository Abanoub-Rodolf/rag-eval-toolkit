"""rag-eval-toolkit: RAG evaluation with readable LLM-judge prompts."""

__version__ = "3.0.0"

from rag_eval.backends import (
    AnthropicBackend,
    BaseBackend,
    GeminiBackend,
    LiteLLMBackend,
    OllamaBackend,
    OpenAIBackend,
)
from rag_eval.evaluator import RAGEvaluator
from rag_eval.metrics import (
    AnswerCompletenessMetric,
    AnswerRelevancyMetric,
    BaseMetric,
    ChunkAttributionMetric,
    CoherenceMetric,
    ConcisenessMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextUtilizationMetric,
    FaithfulnessMetric,
    GroundednessMetric,
    HallucinationMetric,
    SemanticSimilarityMetric,
    ToxicityMetric,
)
from rag_eval.reliability import measure_reliability
from rag_eval.utils import format_score, load_dataset, save_results_to_csv

__all__ = [
    "RAGEvaluator",
    "measure_reliability",
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
