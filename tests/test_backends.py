"""Unit tests for all five LLM backends."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class TestOpenAIBackend:
    def _make_response(self, text):
        resp = MagicMock()
        resp.choices[0].message.content = text
        return resp

    def test_generate_returns_text(self):
        from rag_eval.backends.openai_backend import OpenAIBackend
        with patch("rag_eval.backends.openai_backend.OpenAI") as mock_cls, \
             patch("os.environ.get", side_effect=lambda k, d=None: "sk-test" if k == "OPENAI_API_KEY" else d):
            mock_cls.return_value.chat.completions.create.return_value = self._make_response("0.7")
            backend = OpenAIBackend(model="gpt-4o")
            assert backend.generate("prompt") == "0.7"

    def test_missing_api_key_raises(self):
        from rag_eval.backends.openai_backend import OpenAIBackend
        with patch("rag_eval.backends.openai_backend.OpenAI"), \
             patch("os.environ.get", return_value=None):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIBackend()

    def test_missing_package_raises(self):
        import rag_eval.backends.openai_backend as mod
        original = mod.OpenAI
        try:
            mod.OpenAI = None
            with pytest.raises(ImportError, match="openai"):
                from rag_eval.backends.openai_backend import OpenAIBackend
                OpenAIBackend()
        finally:
            mod.OpenAI = original

    def test_model_attribute_set(self):
        from rag_eval.backends.openai_backend import OpenAIBackend
        with patch("rag_eval.backends.openai_backend.OpenAI"), \
             patch("os.environ.get", side_effect=lambda k, d=None: "sk-test" if k == "OPENAI_API_KEY" else d):
            backend = OpenAIBackend(model="gpt-3.5-turbo")
            assert backend.model == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class TestAnthropicBackend:
    def _make_response(self, text):
        resp = MagicMock()
        resp.content[0].text = text
        return resp

    def test_generate_returns_text(self):
        from rag_eval.backends.anthropic_backend import AnthropicBackend
        with patch("rag_eval.backends.anthropic_backend.Anthropic") as mock_cls, \
             patch("os.environ.get", side_effect=lambda k, d=None: "sk-ant-test" if k == "ANTHROPIC_API_KEY" else d):
            mock_cls.return_value.messages.create.return_value = self._make_response("0.6")
            backend = AnthropicBackend(model="claude-sonnet-4-6")
            assert backend.generate("prompt") == "0.6"

    def test_missing_api_key_raises(self):
        from rag_eval.backends.anthropic_backend import AnthropicBackend
        with patch("rag_eval.backends.anthropic_backend.Anthropic"), \
             patch("os.environ.get", return_value=None):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicBackend()

    def test_missing_package_raises(self):
        import rag_eval.backends.anthropic_backend as mod
        original = mod.Anthropic
        try:
            mod.Anthropic = None
            with pytest.raises(ImportError, match="anthropic"):
                from rag_eval.backends.anthropic_backend import AnthropicBackend
                AnthropicBackend()
        finally:
            mod.Anthropic = original

    def test_default_model(self):
        from rag_eval.backends.anthropic_backend import AnthropicBackend
        with patch("rag_eval.backends.anthropic_backend.Anthropic"), \
             patch("os.environ.get", side_effect=lambda k, d=None: "sk-ant" if k == "ANTHROPIC_API_KEY" else d):
            backend = AnthropicBackend()
            assert backend.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class TestOllamaBackend:
    def test_generate_returns_text(self):
        from rag_eval.backends.ollama_backend import OllamaBackend
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "  0.8  "}
        mock_resp.raise_for_status.return_value = None
        with patch("rag_eval.backends.ollama_backend.requests.post", return_value=mock_resp):
            backend = OllamaBackend(model="llama3")
            assert backend.generate("test prompt") == "0.8"

    def test_generate_raises_on_http_error(self):
        from rag_eval.backends.ollama_backend import OllamaBackend
        with patch("rag_eval.backends.ollama_backend.requests.post", side_effect=Exception("connection refused")):
            backend = OllamaBackend()
            with pytest.raises(RuntimeError, match="Ollama"):
                backend.generate("prompt")

    def test_embed_returns_list(self):
        from rag_eval.backends.ollama_backend import OllamaBackend
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_resp.raise_for_status.return_value = None
        with patch("rag_eval.backends.ollama_backend.requests.post", return_value=mock_resp):
            backend = OllamaBackend()
            emb = backend.embed("text")
            assert emb == [0.1, 0.2, 0.3]

    def test_embed_returns_empty_on_error(self):
        from rag_eval.backends.ollama_backend import OllamaBackend
        with patch("rag_eval.backends.ollama_backend.requests.post", side_effect=Exception("timeout")):
            backend = OllamaBackend()
            assert backend.embed("text") == []

    def test_default_model(self):
        from rag_eval.backends.ollama_backend import OllamaBackend
        backend = OllamaBackend()
        assert backend.model == "llama3"


# ---------------------------------------------------------------------------
# LiteLLM backend
# ---------------------------------------------------------------------------

class TestLiteLLMBackend:
    def _make_response(self, text):
        resp = MagicMock()
        resp.choices[0].message.content = f"  {text}  "
        return resp

    def test_generate_returns_text(self):
        from rag_eval.backends.litellm_backend import LiteLLMBackend
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = self._make_response("0.9")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = LiteLLMBackend(model="gpt-4")
            result = backend.generate("prompt")
            assert result == "0.9"

    def test_generate_raises_on_error(self):
        from rag_eval.backends.litellm_backend import LiteLLMBackend
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = Exception("API error")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = LiteLLMBackend(model="gpt-4")
            with pytest.raises(RuntimeError, match="LiteLLM"):
                backend.generate("prompt")

    def test_embed_returns_list(self):
        from rag_eval.backends.litellm_backend import LiteLLMBackend
        mock_litellm = MagicMock()
        mock_litellm.embedding.return_value.data = [{"embedding": [0.1, 0.2]}]
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = LiteLLMBackend(model="text-embedding-ada-002")
            assert backend.embed("text") == [0.1, 0.2]

    def test_embed_returns_empty_on_error(self):
        from rag_eval.backends.litellm_backend import LiteLLMBackend
        mock_litellm = MagicMock()
        mock_litellm.embedding.side_effect = Exception("err")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = LiteLLMBackend(model="gpt-4")
            assert backend.embed("text") == []


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

class TestGeminiBackend:
    def _make_genai_module(self, text="0.75", embed_values=None):
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        # generate_content
        mock_response = MagicMock()
        mock_response.text = f" {text} "
        mock_client.models.generate_content.return_value = mock_response

        # embed_content
        mock_embed_response = MagicMock()
        mock_embed_response.embedding.values = embed_values or [0.1, 0.2, 0.3]
        mock_client.models.embed_content.return_value = mock_embed_response

        return mock_genai, mock_client

    def test_generate_returns_text(self):
        from rag_eval.backends.gemini_backend import GeminiBackend
        mock_genai, _ = self._make_genai_module("0.75")
        fake_google = MagicMock()
        fake_google.genai = mock_genai
        with patch.dict("sys.modules", {"google": fake_google, "google.genai": mock_genai}), \
             patch("rag_eval.backends.gemini_backend.GeminiBackend.__init__",
                   lambda self, model="gemini-1.5-flash", api_key=None: self.__dict__.update(
                       {"_client": mock_genai.Client(), "model": model}
                   )):
            backend = GeminiBackend(api_key="test-key")
            result = backend.generate("prompt")
            assert result == "0.75"

    def test_missing_api_key_raises(self):
        from rag_eval.backends.gemini_backend import GeminiBackend
        mock_genai, _ = self._make_genai_module()
        fake_google = MagicMock()
        fake_google.genai = mock_genai
        with patch.dict("sys.modules", {"google": fake_google, "google.genai": mock_genai}), \
             patch.dict("os.environ", {}, clear=True):
            import os
            original = os.environ.get
            with patch("os.environ.get", side_effect=lambda k, d=None: None):
                with pytest.raises((ValueError, Exception)):
                    GeminiBackend()

    def test_embed_returns_values(self):
        from rag_eval.backends.gemini_backend import GeminiBackend
        mock_genai, mock_client = self._make_genai_module(embed_values=[0.5, 0.6])
        with patch("rag_eval.backends.gemini_backend.GeminiBackend.__init__",
                   lambda self, model="gemini-1.5-flash", api_key=None: self.__dict__.update(
                       {"_client": mock_client, "model": model}
                   )):
            backend = GeminiBackend(api_key="test-key")
            emb = backend.embed("text")
            assert emb == [0.5, 0.6]

    def test_embed_returns_empty_on_error(self):
        from rag_eval.backends.gemini_backend import GeminiBackend
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("API error")
        with patch("rag_eval.backends.gemini_backend.GeminiBackend.__init__",
                   lambda self, model="gemini-1.5-flash", api_key=None: self.__dict__.update(
                       {"_client": mock_client, "model": model}
                   )):
            backend = GeminiBackend(api_key="test-key")
            assert backend.embed("text") == []
