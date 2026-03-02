from __future__ import annotations

import json
from typing import Iterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.agent_runtime import run_tool_agent
from ..core.session import SessionStore
from ..core.tools.registry import build_tool_registry
from ..integrations.mcp.registry import MCPRegistry, build_mcp_registry
from ..llm.base import BaseLLMClient


EVENT_PREFIX = "[[THISTLEBOT_EVENT]]"


def build_router(client: BaseLLMClient, sessions: SessionStore, config: dict) -> APIRouter:
    router = APIRouter()
    mcp_registry: MCPRegistry = build_mcp_registry(config)
    tool_registry = build_tool_registry(config, mcp_registry)
    tools_cfg = config.get("tools", {})
    runtime_cfg = tools_cfg.get("runtime", {})
    tool_loop_enabled = bool(runtime_cfg.get("enabled", True))
    max_iterations = int(runtime_cfg.get("max_iterations", 8))

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
        model = payload.get("model") or config.get("ollama", {}).get("model", "llama3")

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
        model = payload.get("model") or config.get("ollama", {}).get("model", "llama3")

        def event_stream() -> Iterable[str]:
            user_content = messages[-1]["content"] if messages else ""
            sessions.append_message(session_id, {"role": "user", "content": user_content})
            assistant_content = ""

            if tool_loop_enabled and tool_registry.list_tool_names():
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

            sessions.append_message(session_id, {"role": "assistant", "content": assistant_content})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/session/reset")
    def reset_session(payload: dict) -> dict:
        session_id = payload.get("session_id", "default")
        sessions.reset_session(session_id)
        return {"session_id": session_id, "reset": True}

    return router
