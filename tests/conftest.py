import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_backend():
    b = MagicMock()
    b.model = "test-model"
    b.generate.return_value = "0.8"
    return b


@pytest.fixture
def sample_row():
    return {
        "question": "What is RAG?",
        "context": "RAG stands for Retrieval-Augmented Generation.",
        "answer": "RAG is a technique that combines retrieval and generation.",
        "ground_truth": "Retrieval-Augmented Generation combines retrieval and generation.",
    }
