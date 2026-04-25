"""Google Gemini backend for evaluation."""
import logging
import os
from typing import Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)


class GeminiBackend(BaseBackend):
    """Backend for Google Gemini.

    Args:
        model: Gemini model name (default: "gemini-1.5-flash").
        api_key: Google AI API key. Falls back to ``GOOGLE_API_KEY`` env var.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI SDK not installed. Run: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var.")

        # use a per-instance client to avoid mutating module-level global state
        self._client = genai.Client(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Gemini backend error: {exc}") from exc

    def embed(self, text: str) -> list:
        try:
            result = self._client.models.embed_content(
                model="models/text-embedding-004",
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as exc:
            logger.error("Gemini embedding call failed: %s", exc)
            return []
