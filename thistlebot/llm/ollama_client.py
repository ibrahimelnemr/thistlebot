from __future__ import annotations

import json
from typing import Iterable

import httpx

from .base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/api"):
            cleaned = cleaned[: -len("/api")]
        self.base_url = cleaned
        self.timeout = timeout

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            response = httpx.get(url, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        payload = response.json()
        models = payload.get("models", [])
        return [item.get("name", "") for item in models if item.get("name")]

    def chat(self, messages: list[dict[str, str]], model: str, stream: bool = False) -> str | Iterable[str]:
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}

        if not stream:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            return message.get("content", "")

        def stream_chunks() -> Iterable[str]:
            in_thinking = False
            with httpx.stream("POST", url, json=payload, timeout=self.timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = data.get("message", {})
                    thinking_chunk = message.get("thinking")
                    if thinking_chunk:
                        if not in_thinking:
                            yield "<think>"
                            in_thinking = True
                        yield thinking_chunk

                    content_chunk = message.get("content")
                    if content_chunk:
                        if in_thinking:
                            yield "</think>"
                            in_thinking = False
                        yield content_chunk

                    if data.get("done") and in_thinking:
                        yield "</think>"
                        in_thinking = False

                if in_thinking:
                    yield "</think>"

        return stream_chunks()
