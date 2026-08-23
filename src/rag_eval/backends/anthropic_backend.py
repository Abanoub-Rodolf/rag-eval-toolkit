import os

from .base import BaseBackend


class AnthropicBackend(BaseBackend):
    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Please install anthropic: pip install anthropic") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
