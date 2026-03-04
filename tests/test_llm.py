"""Tests for LLM provider factory and base class."""

import pytest
import os
from beyondml.llm.base import LLMProvider
from beyondml.llm import get_llm_provider


class TestLLMBase:
    def test_cannot_instantiate_base(self):
        """LLMProvider is abstract and should not be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()


class TestLLMFactory:
    def test_default_is_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        provider = get_llm_provider()
        from beyondml.llm.ollama_provider import OllamaProvider
        assert isinstance(provider, OllamaProvider)

    def test_explicit_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        provider = get_llm_provider()
        from beyondml.llm.ollama_provider import OllamaProvider
        assert isinstance(provider, OllamaProvider)

    def test_groq_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        # GroqProvider __init__ may need an API key, so we set a dummy
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        provider = get_llm_provider()
        from beyondml.llm.groq_provider import GroqProvider
        assert isinstance(provider, GroqProvider)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "nonexistent")
        with pytest.raises(ValueError, match="Unknown LLM Provider"):
            get_llm_provider()
