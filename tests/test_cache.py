import os
import pytest
from rag_eval.utils.cache import EvaluationCache

DB = "test_cache_unit.db"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists(DB):
        os.remove(DB)


def test_miss_returns_none():
    c = EvaluationCache(DB)
    assert c.get("faithfulness", "gpt-4o", {"question": "q"}) is None


def test_set_then_get():
    c = EvaluationCache(DB)
    row = {"question": "q", "context": "c", "answer": "a"}
    c.set("faithfulness", "gpt-4o", row, 0.9)
    assert c.get("faithfulness", "gpt-4o", row) == pytest.approx(0.9)


def test_different_metrics_dont_collide():
    c = EvaluationCache(DB)
    row = {"question": "q", "context": "c", "answer": "a"}
    c.set("faithfulness", "m", row, 0.9)
    c.set("hallucination", "m", row, 0.4)
    assert c.get("faithfulness", "m", row) == pytest.approx(0.9)
    assert c.get("hallucination", "m", row) == pytest.approx(0.4)


def test_list_context_same_as_joined():
    c = EvaluationCache(DB)
    row_list = {"question": "q", "context": ["chunk a", "chunk b"], "answer": "a"}
    row_str = {"question": "q", "context": "chunk a\n---\nchunk b", "answer": "a"}
    c.set("chunk_attribution", "m", row_list, 0.7)
    assert c.get("chunk_attribution", "m", row_str) == pytest.approx(0.7)


def test_overwrite():
    c = EvaluationCache(DB)
    row = {"question": "q", "context": "c", "answer": "a"}
    c.set("faithfulness", "m", row, 0.5)
    c.set("faithfulness", "m", row, 0.9)
    assert c.get("faithfulness", "m", row) == pytest.approx(0.9)
