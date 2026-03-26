"""LiteLLM backend for universal provider support."""
import logging
from typing import Any, Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)


class LiteLLMBackend(BaseBackend):
    """Universal backend using LiteLLM to support 100+ LLM providers.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "anthropic/claude-3-opus", "gemini/gemini-pro").
        api_key: Optional API key.
    """

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None) -> None:
        try:
            import litellm
            self.model = model
            self.api_key = api_key
            # LiteLLM automatically handles keys from env vars if not provided
        except ImportError:
            raise ImportError("LiteLLM not installed. Run: pip install litellm")

    def generate(self, prompt: str) -> str:
        """Generate response from model via LiteLLM."""
        import litellm
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LiteLLM API call failed: %s", e)
            raise RuntimeError(f"LiteLLM backend error: {e}") from e

    def embed(self, text: str) -> list:
        """Get embeddings via LiteLLM."""
        import litellm
        try:
            response = litellm.embedding(
                model=self.model,
                input=[text],
                api_key=self.api_key
            )
            return response.data[0]["embedding"]
        except Exception as e:
            logger.error("LiteLLM embedding call failed: %s", e)
            return []
