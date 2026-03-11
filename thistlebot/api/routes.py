from __future__ import annotations

import json
import logging
from typing import Iterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.agent_runtime import run_tool_agent
from ..core.session import SessionStore
from ..core.tools.registry import build_tool_registry
from ..integrations.mcp.registry import MCPRegistry, build_mcp_registry
from ..llm.base import BaseLLMClient
from ..llm.factory import get_default_model


EVENT_PREFIX = "[[THISTLEBOT_EVENT]]"
LOGGER = logging.getLogger(__name__)


def build_router(client: BaseLLMClient, sessions: SessionStore, config: dict) -> APIRouter:
    router = APIRouter()
    mcp_registry: MCPRegistry = build_mcp_registry(config)
    tool_registry = build_tool_registry(config, mcp_registry)
    tools_cfg = config.get("tools", {})
    runtime_cfg = tools_cfg.get("runtime", {})
    tool_loop_enabled = bool(runtime_cfg.get("enabled", True))
    max_iterations = int(runtime_cfg.get("max_iterations", 8))
    openrouter_stream_with_tools_default = bool(runtime_cfg.get("openrouter_stream_with_tools", False))

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @router.get("/models")
    def list_models() -> dict:
        return {"models": client.list_models()}

    @router.post("/chat")
    def chat(payload: dict) -> dict:
        session_id = payload.get("session_id", "default")
        messages = payload.get("messages", [])
        model = payload.get("model") or get_default_model(config)

        if tool_loop_enabled and tool_registry.list_tool_names():
            content = run_tool_agent(
                client=client,
                registry=tool_registry,
                model=model,
                messages=messages,
                max_iterations=max_iterations,
            )
        else:
            content = client.chat(messages, model=model, stream=False)
            if isinstance(content, dict):
                content = str(content.get("content") or "")

        message = {"role": "assistant", "content": content}
        user_content = messages[-1]["content"] if messages else ""
        sessions.append_message(session_id, {"role": "user", "content": user_content})
        sessions.append_message(session_id, message)
        return {"session_id": session_id, "message": message}

    @router.post("/chat/stream")
    def chat_stream(payload: dict) -> StreamingResponse:
        session_id = payload.get("session_id", "default")
        messages = payload.get("messages", [])
        model = payload.get("model") or get_default_model(config)
        provider = str(config.get("llm", {}).get("provider") or "").strip()
        payload_stream_with_tools = payload.get("stream_with_tools")
        if payload_stream_with_tools is None:
            stream_with_tools = openrouter_stream_with_tools_default
        else:
            stream_with_tools = bool(payload_stream_with_tools)
        use_tool_loop_stream = tool_loop_enabled and tool_registry.list_tool_names() and (provider != "openrouter" or stream_with_tools)

        def event_stream() -> Iterable[str]:
            user_content = messages[-1]["content"] if messages else ""
            sessions.append_message(session_id, {"role": "user", "content": user_content})
            assistant_content = ""
            stream_failed = False

            try:
                if use_tool_loop_stream:
                    assistant_content, events = run_tool_agent(
                        client=client,
                        registry=tool_registry,
                        model=model,
                        messages=messages,
                        max_iterations=max_iterations,
                        return_events=True,
                    )
                    for event in events:
                        yield f"data: {EVENT_PREFIX}{json.dumps(event)}\n\n"
                    if assistant_content:
                        yield f"data: {assistant_content}\n\n"
                else:
                    chunks = client.chat(messages, model=model, stream=True)
                    if not isinstance(chunks, str):
                        for chunk in chunks:
                            assistant_content += chunk
                            yield f"data: {chunk}\n\n"
            except Exception as exc:
                stream_failed = True
                LOGGER.exception("chat stream failed for session_id=%s model=%s", session_id, model)
                event = {
                    "event": "stream_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                yield f"data: {EVENT_PREFIX}{json.dumps(event)}\n\n"
                yield "data: [STREAM_ERROR]\n\n"

            sessions.append_message(session_id, {"role": "assistant", "content": assistant_content})
            if not stream_failed:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/session/reset")
    def reset_session(payload: dict) -> dict:
        session_id = payload.get("session_id", "default")
        sessions.reset_session(session_id)
        return {"session_id": session_id, "reset": True}

    return router
