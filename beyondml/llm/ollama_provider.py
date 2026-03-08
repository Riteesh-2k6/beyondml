"""
Ollama LLM Provider — local inference.

Reads OLLAMA_HOST and OLLAMA_MODEL from environment / .env file.
Supports both synchronous chat and async streaming via aiohttp and requests.
"""

import os
import json
import asyncio
import requests
import aiohttp
from typing import List, Dict, AsyncGenerator, Optional
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider (privacy-focused, local execution)."""

    def __init__(self, host: str = None, model: str = None):
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self._url = f"{self._host.rstrip('/')}/api/chat"

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    def test_connection(self) -> bool:
        """Check if Ollama is running and has the model."""
        try:
            # Check server version/tags as a lightweight health check
            tags_url = f"{self._host.rstrip('/')}/api/tags"
            response = requests.get(tags_url, timeout=2)
            if response.status_code != 200:
                return False
            
            # Check if our specific model exists in tags
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            return self._model in models or f"{self._model}:latest" in models
        except Exception:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.3,
        timeout: Optional[int] = None,
    ) -> str:
        """Synchronous chat completion."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        if json_mode:
            payload["format"] = "json"

        effective_timeout = timeout or self.DEFAULT_TIMEOUT
        response = requests.post(self._url, json=payload, timeout=effective_timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama asynchronously."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self._url, json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
