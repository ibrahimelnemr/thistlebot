from __future__ import annotations

import json
from typing import Iterable

import httpx

from .base import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.default_headers = default_headers or {}

    def _headers(self) -> dict[str, str]:
        headers = dict(self.default_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        try:
            response = httpx.get(url, headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError:
            return []

        payload = response.json() if response.content else {}
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return [item.get("id", "") for item in data if isinstance(item, dict) and item.get("id")]

    def chat(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        tools: list[dict] | None = None,
    ) -> str | dict | Iterable[str]:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools

        if not stream:
            response = httpx.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            data = response.json() if response.content else {}
            choices = data.get("choices", []) if isinstance(data, dict) else []
            first_choice = choices[0] if choices else {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            if isinstance(message, dict) and message.get("tool_calls"):
                return {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": message.get("tool_calls", []),
                }
            return str(message.get("content", ""))

        def stream_chunks() -> Iterable[str]:
            in_thinking = False
            with httpx.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", []) if isinstance(data, dict) else []
                    first_choice = choices[0] if choices else {}
                    delta = first_choice.get("delta", {}) if isinstance(first_choice, dict) else {}
                    if not isinstance(delta, dict):
                        continue

                    reasoning = self._extract_reasoning_text(delta)
                    if reasoning:
                        if not in_thinking:
                            yield "<think>"
                            in_thinking = True
                        yield reasoning

                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        if in_thinking:
                            yield "</think>"
                            in_thinking = False
                        yield content

                    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
                    if finish_reason and in_thinking:
                        yield "</think>"
                        in_thinking = False

                if in_thinking:
                    yield "</think>"

        return stream_chunks()

    def _extract_reasoning_text(self, delta: dict) -> str:
        candidates = [
            delta.get("reasoning"),
            delta.get("reasoning_content"),
            delta.get("reasoning_text"),
        ]
        parts: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                parts.append(candidate)
            elif isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item:
                        parts.append(item)
                    elif isinstance(item, dict):
                        text_value = item.get("text")
                        if isinstance(text_value, str) and text_value:
                            parts.append(text_value)
        return "".join(parts)