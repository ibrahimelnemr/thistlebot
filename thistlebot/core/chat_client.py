from __future__ import annotations

from typing import Iterable

import httpx


def stream_chat(
    gateway_url: str,
    messages: list[dict[str, str]],
    model: str,
    session_id: str,
) -> Iterable[str]:
    url = f"{gateway_url.rstrip('/')}/chat/stream"
    payload = {"session_id": session_id, "messages": messages, "model": model}

    with httpx.stream("POST", url, json=payload, timeout=None) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data = line.replace("data: ", "", 1)
                if data.strip() == "[DONE]":
                    break
                yield data
