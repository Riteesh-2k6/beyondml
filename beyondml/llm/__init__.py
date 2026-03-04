from .base import LLMProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
import os

def get_llm_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "groq":
        return GroqProvider()
    elif provider == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
