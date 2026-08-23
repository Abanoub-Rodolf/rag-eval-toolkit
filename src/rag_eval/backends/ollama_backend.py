"""Ollama backend for local LLM-as-judge evaluation."""
import logging

import requests

from .base import BaseBackend

logger = logging.getLogger(__name__)


class OllamaBackend(BaseBackend):
    """Local Ollama server.

    Defaults to llama3 on http://localhost:11434.
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as exc:
            logger.error("Ollama API call failed: %s", exc)
            raise RuntimeError(f"Ollama backend error: {exc}") from exc

    def embed(self, text: str) -> list:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as exc:
            logger.error("Ollama embedding call failed: %s", exc)
            raise RuntimeError(f"Ollama backend error: {exc}") from exc
