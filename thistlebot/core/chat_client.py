from __future__ import annotations

import json
from typing import Callable
from typing import Iterable

import httpx


EVENT_PREFIX = "[[THISTLEBOT_EVENT]]"


class ChatStreamError(RuntimeError):
    pass


def stream_chat(
    gateway_url: str,
    messages: list[dict[str, str]],
    model: str,
    session_id: str,
    on_event: Callable[[dict], None] | None = None,
) -> Iterable[str]:
    url = f"{gateway_url.rstrip('/')}/chat/stream"
    payload = {"session_id": session_id, "messages": messages, "model": model}

    try:
        with httpx.stream("POST", url, json=payload, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line.replace("data: ", "", 1)
                    if data.strip() == "[DONE]":
                        break
                    if data.strip() == "[STREAM_ERROR]":
                        raise ChatStreamError(
                            "Gateway reported a stream_error event while generating SSE response "
                            f"(url={url}, model={model}, session={session_id})"
                        )
                    if data.startswith(EVENT_PREFIX):
                        if on_event is not None:
                            raw = data[len(EVENT_PREFIX) :]
                            try:
                                on_event(json.loads(raw))
                            except json.JSONDecodeError:
                                on_event({"event": "tool_event_parse_error", "raw": raw})
                        continue
                    yield data
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip() if exc.response is not None else ""
        preview = response_text[:280] + ("..." if len(response_text) > 280 else "")
        detail = f"Gateway stream request failed with {exc.response.status_code if exc.response is not None else 'unknown status'}"
        if preview:
            detail = f"{detail}: {preview}"
        raise ChatStreamError(f"{detail} (url={url}, model={model}, session={session_id})") from exc
    except httpx.RemoteProtocolError as exc:
        raise ChatStreamError(
            "Gateway stream terminated early while reading chunked response "
            f"(url={url}, model={model}, session={session_id}). "
            "This usually means an exception occurred inside the gateway stream generator; "
            "check gateway stderr and ~/.thistlebot/logs/gateway.log"
        ) from exc
    except httpx.HTTPError as exc:
        raise ChatStreamError(
            f"Gateway stream request failed ({type(exc).__name__}) "
            f"(url={url}, model={model}, session={session_id}): {exc}"
        ) from exc
