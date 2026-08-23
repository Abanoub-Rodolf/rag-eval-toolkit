"""LiteLLM backend for universal provider support."""
import logging
from typing import Optional

from .base import BaseBackend

logger = logging.getLogger(__name__)


class LiteLLMBackend(BaseBackend):
    """Universal backend using LiteLLM to support 100+ LLM providers.

    Args:
        model: Model identifier (e.g. "gpt-5.1", "anthropic/claude-sonnet-5",
            "gemini/gemini-2.5-flash"). LiteLLM resolves keys from env vars if
            api_key is not passed.
    """

    def __init__(self, model: str = "gpt-5.1", api_key: Optional[str] = None) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError("LiteLLM not installed. Run: pip install litellm") from exc
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        import litellm
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as exc:
            logger.error("LiteLLM API call failed: %s", exc)
            raise RuntimeError(f"LiteLLM backend error: {exc}") from exc

    def embed(self, text: str) -> list:
        import litellm
        try:
            response = litellm.embedding(
                model=self.model,
                input=[text],
                api_key=self.api_key,
            )
            return response.data[0]["embedding"]
        except Exception as exc:
            logger.error("LiteLLM embedding call failed: %s", exc)
            raise RuntimeError(f"LiteLLM backend error: {exc}") from exc
