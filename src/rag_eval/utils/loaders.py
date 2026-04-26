"""Dataset loading utilities."""
import csv
import json
import os
from typing import Any, Dict, List


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a dataset from CSV, JSON, or JSONL."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8-sig") as f:
            rows = []
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                rows.append(json.loads(stripped))
            return rows

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    raise ValueError(f"Unsupported dataset format: {path}. Use .csv, .json, or .jsonl")
