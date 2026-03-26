"""Google Gemini backend for evaluation."""
import logging
import os
from typing import Any, Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)


class GeminiBackend(BaseBackend):
    """Backend for Google Gemini.
    
    Args:
        model: Name of the Gemini model (default: "gemini-1.5-flash").
        api_key: Google AI API key. If not provided, reads from GOOGLE_API_KEY env var.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> None:
        try:
            import google.generativeai as genai
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var.")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("Google Generative AI SDK not installed. Run: pip install google-generativeai")

    def generate(self, prompt: str) -> str:
        """Generate response from Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            raise RuntimeError(f"Gemini backend error: {e}") from e

    def embed(self, text: str) -> list:
        """Get embeddings from Gemini."""
        import google.generativeai as genai
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            logger.error("Gemini embedding call failed: %s", e)
            return []
