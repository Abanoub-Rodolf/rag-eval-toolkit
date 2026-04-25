"""Cache utility for LLM responses."""
import hashlib
import json
import sqlite3
from typing import Any, Optional

class EvaluationCache:
    """SQLite-based cache for evaluation results.
    
    Args:
        db_path: Path to SQLite database file (default: ".rag_eval_cache.db").
    """

    def __init__(self, db_path: str = ".rag_eval_cache.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _generate_key(self, metric_name: str, model_name: str, row: dict) -> str:
        """Generate a unique hash key for a given input."""
        # Use only relevant keys for hashing
        hash_input = {
            "metric": metric_name,
            "model": model_name,
            "data": {k: row.get(k) for k in ["question", "context", "answer", "ground_truth"]}
        }
        serialized = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self, metric_name: str, model_name: str, row: dict) -> Optional[float]:
        """Retrieve cached score if it exists."""
        key = self._generate_key(metric_name, model_name, row)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            return float(result[0]) if result else None

    def set(self, metric_name: str, model_name: str, row: dict, score: float) -> None:
        """Store score in cache."""
        key = self._generate_key(metric_name, model_name, row)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, str(score))
            )
