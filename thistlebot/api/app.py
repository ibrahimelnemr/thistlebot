from __future__ import annotations

from fastapi import FastAPI

from ..core.session import SessionStore
from ..llm.factory import build_llm_client
from .routes import build_router


def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="Thistlebot Gateway")

    client = build_llm_client(config)
    session_store = SessionStore()

    app.include_router(build_router(client, session_store, config))
    return app
