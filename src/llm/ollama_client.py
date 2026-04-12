"""
Ollama Cloud API Client for LLM-based Recommendation.

Supports various open-source models for recommendation tasks.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
import time


@dataclass
class OllamaConfig:
    """Configuration for Ollama Cloud API."""
    api_key: str
    base_url: str = "https://ollama.com/api"
    default_model: str = "qwen3:8b-cloud"
    timeout: int = 60
    max_retries: int = 3


class OllamaClient:
    """Client for Ollama Cloud API."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        tools: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (default: config.default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            tools: Optional tool definitions for function calling

        Returns:
            Response dict with 'message' containing assistant response
        """
        model = model or self.config.default_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if tools:
            payload["tools"] = tools

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.config.base_url}/chat",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream chat response token by token."""
        model = model or self.config.default_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        response = self.session.post(
            f"{self.config.base_url}/chat",
            json=payload,
            stream=True,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]

    def embed(
        self,
        texts: List[str],
        model: str = "nomic-embed-text:cloud",
    ) -> List[List[float]]:
        """
        Get embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model name

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            payload = {
                "model": model,
                "input": text,
            }
            response = self.session.post(
                f"{self.config.base_url}/embed",
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            result = response.json()
            embeddings.append(result.get("embeddings", [[]])[0])

        return embeddings

    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Simple text generation."""
        model = model or self.config.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        response = self.session.post(
            f"{self.config.base_url}/generate",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")


# Singleton client instance
_client: Optional[OllamaClient] = None


def get_client(api_key: str = None) -> OllamaClient:
    """Get or create Ollama client."""
    global _client
    if _client is None:
        api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY not set")
        config = OllamaConfig(api_key=api_key)
        _client = OllamaClient(config)
    return _client


# Available models for different tasks
MODELS = {
    # General recommendation
    "general": [
        "qwen3:8b-cloud",
        "mistral:7b-cloud",
        "llama3.3:70b-cloud",
    ],
    # Thinking/reasoning models
    "thinking": [
        "qwen3-next:80b-cloud",
        "deepseek-v3.1:671b-cloud",
        "kimi-k2-thinking:cloud",
    ],
    # Tool use models
    "tools": [
        "qwen3-vl:8b-cloud",
        "ministral-3:8b-cloud",
        "devstral-small-2:24b-cloud",
    ],
    # Embedding models
    "embedding": [
        "nomic-embed-text:cloud",
        "mxbai-embed-large:cloud",
    ],
}
