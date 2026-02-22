from __future__ import annotations

from typing import Iterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.session import SessionStore
from ..llm.base import BaseLLMClient


def build_router(client: BaseLLMClient, sessions: SessionStore, config: dict) -> APIRouter:
    router = APIRouter()

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

        content = client.chat(messages, model=model, stream=False)
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
            chunks = client.chat(messages, model=model, stream=True)
            assistant_content = ""
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
