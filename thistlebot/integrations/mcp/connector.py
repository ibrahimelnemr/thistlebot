from __future__ import annotations

from typing import Protocol


class MCPConnector(Protocol):
    def connect(self, config: dict) -> None:
        raise NotImplementedError

    def list_tools(self) -> list[str]:
        raise NotImplementedError

    def invoke(self, tool_name: str, payload: dict) -> dict:
        raise NotImplementedError
