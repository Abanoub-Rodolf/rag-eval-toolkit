"""Small formatting and export helpers."""
import csv
from typing import Any


def format_score(score: float) -> str:
    return f"{score:.2%}"


def flatten_context(context: Any) -> str:
    """Join a list of context chunks into one string; pass strings through.

    Canonical implementation -- cache keys, judge prompts, and ranked metrics
    must agree on the separator or cache hits silently miss.
    """
    if isinstance(context, list):
        return "\n---\n".join(str(c) for c in context)
    return str(context)


def save_results_to_csv(results: list[dict[str, Any]], output_path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
