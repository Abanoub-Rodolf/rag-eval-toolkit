"""Dataset loading utilities."""
import csv
import json
import os
from typing import Any, Dict, List

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load dataset from CSV or JSONL file.
    
    Args:
        path: Path to dataset file.
        
    Returns:
        List of dictionaries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
            
    raise ValueError(f"Unsupported dataset format: {path}. Use .csv, .json, or .jsonl")
