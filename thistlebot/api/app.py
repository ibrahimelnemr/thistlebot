from __future__ import annotations

from fastapi import FastAPI

from ..core.session import SessionStore
from ..llm.ollama_client import OllamaClient
from .routes import build_router


def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="Thistlebot Gateway")

    ollama_cfg = config.get("ollama", {})
    client = OllamaClient(ollama_cfg.get("base_url", "http://127.0.0.1:11434"))
    session_store = SessionStore()

    app.include_router(build_router(client, session_store, config))
    return app
