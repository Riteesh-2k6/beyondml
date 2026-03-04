"""
Groq LLM Provider — uses the Groq Python SDK.

Reads GROQ_API_KEY and GROQ_MODEL from environment / .env file.
Supports both synchronous chat and async streaming.
"""

import os
import asyncio
from typing import List, Dict, AsyncGenerator, Optional
from groq import Groq
from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq cloud LLM provider (fast inference on LPU)."""

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Pass it directly or set the environment variable."
            )

        self._client = Groq(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.3,
        timeout: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        effective_timeout = timeout or self.DEFAULT_TIMEOUT
        kwargs["timeout"] = effective_timeout
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Groq. Runs the sync stream in a thread."""

        def _sync_stream():
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stream=True,
            )

        stream = await asyncio.to_thread(_sync_stream)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
