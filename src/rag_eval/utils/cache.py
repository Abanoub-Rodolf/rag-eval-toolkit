"""SQLite-backed cache for evaluation results."""
import hashlib
import json
import sqlite3
import threading
from typing import Any, Optional

from .helpers import flatten_context


class EvaluationCache:
    """Persistent cache for per-sample metric scores.

    Uses a single SQLite connection held for the lifetime of the instance.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = ".rag_eval_cache.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self._conn.commit()

    def _key(self, metric_name: str, model_name: str, row: dict[str, Any]) -> str:
        context = flatten_context(row.get("context", ""))
        payload = {
            "metric": metric_name,
            "model": model_name,
            "data": {
                "question": row.get("question"),
                "context": context,
                "answer": row.get("answer"),
                "ground_truth": row.get("ground_truth"),
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, metric_name: str, model_name: str, row: dict[str, Any]) -> Optional[float]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value FROM cache WHERE key = ?", (self._key(metric_name, model_name, row),)
            )
            result = cursor.fetchone()
            return float(result[0]) if result else None

    def set(self, metric_name: str, model_name: str, row: dict[str, Any], score: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (self._key(metric_name, model_name, row), str(score)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        # threading module may be torn down at interpreter shutdown, skip the lock
        try:
            self._conn.close()
        except Exception:
            pass
