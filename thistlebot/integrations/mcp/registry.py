from __future__ import annotations

from typing import Dict

from .connector import MCPConnector


class MCPRegistry:
    def __init__(self) -> None:
        self._connectors: Dict[str, MCPConnector] = {}

    def register(self, name: str, connector: MCPConnector) -> None:
        self._connectors[name] = connector

    def list(self) -> list[str]:
        return sorted(self._connectors.keys())

    def get(self, name: str) -> MCPConnector | None:
        return self._connectors.get(name)
