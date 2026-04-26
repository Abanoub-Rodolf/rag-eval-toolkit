import json
import os
import tempfile
import pytest
from rag_eval.utils.loaders import load_dataset


def _write(suffix, content, mode="w"):
    f = tempfile.NamedTemporaryFile(suffix=suffix, mode=mode, delete=False)
    f.write(content)
    f.close()
    return f.name


def test_load_json_list():
    path = _write(".json", json.dumps([{"q": "a"}]))
    try:
        data = load_dataset(path)
        assert data == [{"q": "a"}]
    finally:
        os.unlink(path)


def test_load_json_single_object():
    path = _write(".json", json.dumps({"q": "a"}))
    try:
        data = load_dataset(path)
        assert data == [{"q": "a"}]
    finally:
        os.unlink(path)


def test_load_jsonl():
    lines = "\n".join(json.dumps({"i": i}) for i in range(3))
    path = _write(".jsonl", lines)
    try:
        data = load_dataset(path)
        assert len(data) == 3
        assert data[1] == {"i": 1}
    finally:
        os.unlink(path)


def test_load_csv():
    path = _write(".csv", "question,answer\nWhat is RAG?,A technique\n")
    try:
        data = load_dataset(path)
        assert data[0]["question"] == "What is RAG?"
    finally:
        os.unlink(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("/tmp/nonexistent_rag_eval_file.json")


def test_unsupported_extension_raises():
    path = _write(".txt", "hello")
    try:
        with pytest.raises(ValueError):
            load_dataset(path)
    finally:
        os.unlink(path)


def test_jsonl_with_utf8_bom():
    content = "﻿" + json.dumps({"q": "ok"}) + "\n"
    path = _write(".jsonl", content)
    try:
        data = load_dataset(path)
        assert data == [{"q": "ok"}]
    finally:
        os.unlink(path)


def test_csv_with_utf8_bom_preserves_keys():
    content = "﻿question,answer\nQ?,A\n"
    path = _write(".csv", content)
    try:
        data = load_dataset(path)
        assert "question" in data[0]
        assert "﻿question" not in data[0]
    finally:
        os.unlink(path)


def test_json_with_utf8_bom():
    path = _write(".json", "﻿" + json.dumps([{"q": "ok"}]))
    try:
        data = load_dataset(path)
        assert data == [{"q": "ok"}]
    finally:
        os.unlink(path)


def test_jsonl_skips_comments_and_blank_lines():
    content = "# header comment\n" + json.dumps({"i": 1}) + "\n\n  \n# mid\n" + json.dumps({"i": 2}) + "\n"
    path = _write(".jsonl", content)
    try:
        data = load_dataset(path)
        assert data == [{"i": 1}, {"i": 2}]
    finally:
        os.unlink(path)
