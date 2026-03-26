"""Ollama backend for local LLM-as-judge evaluation."""
import json
import logging
import requests
from typing import Any, Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)


class OllamaBackend(BaseBackend):
    """Backend for Ollama (local LLMs).
    
    Args:
        model: Name of the model to use (default: "llama3").
        base_url: Base URL for Ollama API (default: "http://localhost:11434").
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        """Generate response from local Ollama model.
        
        Args:
            prompt: Prompt string.
            
        Returns:
            Model response text.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error("Ollama API call failed: %s", e)
            raise RuntimeError(f"Ollama backend error: {e}") from e

    def embed(self, text: str) -> list:
        """Get embeddings from Ollama.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the embedding.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except Exception as e:
            logger.error("Ollama embedding call failed: %s", e)
            return []
