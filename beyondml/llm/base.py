"""
Abstract LLM Provider interface.

All agents call LLM through this interface. To add a local model backend
(Ollama, llama.cpp, vLLM), create a new class inheriting LLMProvider and
implement chat() and chat_stream(). No agent code changes needed.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, AsyncGenerator, Optional


class LLMProvider(ABC):
    """Base class for all LLM providers (Groq, Ollama, local, etc.)."""

    DEFAULT_TIMEOUT = 60  # seconds

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.3,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Synchronous chat completion.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            json_mode: If True, force JSON output format.
            temperature: Sampling temperature.
            timeout: Max seconds to wait. Defaults to DEFAULT_TIMEOUT.

        Returns:
            The assistant's reply as a string.
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat completion — yields token deltas.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.

        Yields:
            Token strings as they arrive.
        """
        ...
        # Need yield to make it an async generator
        yield ""  # pragma: no cover
