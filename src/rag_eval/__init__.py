"""
rag-eval-toolkit: A production-grade framework for evaluating Retrieval-Augmented Generation pipelines.
"""

__version__ = "1.0.0"

from rag_eval.evaluator import RAGEvaluator
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

__all__ = [
    "RAGEvaluator",
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
]
