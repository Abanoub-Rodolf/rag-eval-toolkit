"""Small formatting and export helpers."""
import csv
from typing import Any


def format_score(score: float) -> str:
    return f"{score:.2%}"


def save_results_to_csv(results: list[dict[str, Any]], output_path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
