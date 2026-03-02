from __future__ import annotations

import json
from typing import Callable
from typing import Iterable

import httpx


EVENT_PREFIX = "[[THISTLEBOT_EVENT]]"


def stream_chat(
    gateway_url: str,
    messages: list[dict[str, str]],
    model: str,
    session_id: str,
    on_event: Callable[[dict], None] | None = None,
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
                if data.startswith(EVENT_PREFIX):
                    if on_event is not None:
                        raw = data[len(EVENT_PREFIX) :]
                        try:
                            on_event(json.loads(raw))
                        except json.JSONDecodeError:
                            on_event({"event": "tool_event_parse_error", "raw": raw})
                    continue
                yield data
