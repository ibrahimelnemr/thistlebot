from __future__ import annotations

import json
from typing import Any

from ..llm.base import BaseLLMClient
from .tools.registry import ToolRegistry, normalize_tool_args


def run_tool_agent(
    *,
    client: BaseLLMClient,
    registry: ToolRegistry,
    model: str,
    messages: list[dict],
    max_iterations: int = 8,
    return_events: bool = False,
) -> str | tuple[str, list[dict[str, Any]]]:
    history: list[dict] = list(messages)
    tools = registry.to_model_tools()
    events: list[dict[str, Any]] = []

    for _ in range(max_iterations):
        response = client.chat(history, model=model, stream=False, tools=tools)
        assistant_message = _normalize_assistant_message(response)
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
            content = str(assistant_message.get("content") or "").strip()
            if return_events:
                return content, events
            return content

        history.append(
            {
                "role": "assistant",
                "content": str(assistant_message.get("content") or ""),
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            tool_name = _tool_name(call)
            tool_call_id = str(call.get("id") or "")
            raw_args = _tool_args(call)
            args = normalize_tool_args(raw_args)
            events.append({"event": "tool_call", "tool": tool_name, "args": args})

            result = registry.invoke(tool_name, args)
            payload = result.as_payload()
            events.append(
                {
                    "event": "tool_result",
                    "tool": tool_name,
                    "ok": result.ok,
                    "error": result.error,
                    "content": result.content,
                    "truncated": result.truncated,
                }
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "tool_name": tool_name,
                    "content": json.dumps(payload, default=str),
                }
            )

    final_text = "I reached the tool-iteration limit before producing a final response."
    if return_events:
        return final_text, events
    return final_text


def _normalize_assistant_message(response: str | dict | Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    return {"role": "assistant", "content": str(response or "")}


def _tool_name(tool_call: dict[str, Any]) -> str:
    function_data = tool_call.get("function")
    if isinstance(function_data, dict):
        return str(function_data.get("name") or "")
    return str(tool_call.get("name") or "")


def _tool_args(tool_call: dict[str, Any]) -> dict[str, Any] | str:
    function_data = tool_call.get("function")
    if isinstance(function_data, dict):
        return function_data.get("arguments") or {}
    return tool_call.get("arguments") or {}
