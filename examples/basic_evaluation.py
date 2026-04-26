"""Basic example: evaluate a RAG pipeline using rag-eval-toolkit.

Usage:
    pip install -e ".[anthropic]"
    export ANTHROPIC_API_KEY=your_key_here
    python examples/basic_evaluation.py
"""
from rag_eval import RAGEvaluator, AnthropicBackend
from rag_eval import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric


SAMPLE_DATASET = [
    {
        "question": "What is retrieval-augmented generation?",
        "context": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with language model generation. It retrieves "
            "relevant documents from a knowledge base and uses them to ground "
            "the model's response."
        ),
        "answer": "RAG is a technique that retrieves relevant documents and uses them to generate more accurate, grounded responses.",
        "ground_truth": "Retrieval-Augmented Generation combines retrieval and generation to produce factually grounded answers.",
    },
    {
        "question": "What programming language is commonly used for ML?",
        "context": "Python is the dominant language in machine learning and AI research. Libraries like PyTorch, TensorFlow, and scikit-learn are all Python-based.",
        "answer": "Python is the most commonly used language for machine learning.",
        "ground_truth": "Python is the dominant language in ML, supported by libraries like PyTorch and TensorFlow.",
    },
]


def main():
    print("=== RAG Evaluation Toolkit - Basic Example ===\n")

    backend = AnthropicBackend()
    evaluator = RAGEvaluator(backend=backend)

    evaluator.add_metric(FaithfulnessMetric())
    evaluator.add_metric(AnswerRelevancyMetric())
    evaluator.add_metric(HallucinationMetric())

    print("Evaluating dataset...")
    results = evaluator.evaluate(SAMPLE_DATASET)

    print("\n=== Results ===")
    print(f"Averages: {results['averages']}")
    for metric_name, scores in results["per_sample"].items():
        print(f"  {metric_name}: {scores}")


if __name__ == "__main__":
    main()
