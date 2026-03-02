from __future__ import annotations

from typing import Dict

from ...core.tools.base import ToolEntry
from ...core.tools.policy import ToolPolicy
from .client import HttpMCPClient, StdioMCPClient
from .connector import MCPConnector
from .tool_wrapper import wrap_mcp_tools


class MCPRegistry:
    def __init__(self) -> None:
        self._connectors: Dict[str, MCPConnector] = {}
        self._tool_entries: list[ToolEntry] = []

    def register(self, name: str, connector: MCPConnector) -> None:
        self._connectors[name] = connector

    def list(self) -> list[str]:
        return sorted(self._connectors.keys())

    def get(self, name: str) -> MCPConnector | None:
        return self._connectors.get(name)

    def tool_entries(self) -> list[ToolEntry]:
        return list(self._tool_entries)

    def statuses(self) -> list[dict]:
        rows: list[dict] = []
        for name in self.list():
            connector = self._connectors[name]
            status = connector.status()
            status.setdefault("name", name)
            rows.append(status)
        return rows


def build_mcp_registry(config: dict) -> MCPRegistry:
    registry = MCPRegistry()
    mcp_cfg = config.get("mcp", {})
    if not mcp_cfg.get("enabled"):
        return registry

    servers = mcp_cfg.get("servers", {}) or {}
    for server_name, server_cfg in servers.items():
        if not isinstance(server_cfg, dict):
            continue
        if not server_cfg.get("enabled", True):
            continue

        effective_cfg = _materialize_server_config(server_cfg, config)
        transport = str(effective_cfg.get("transport", "stdio")).lower()
        connector: MCPConnector
        if transport == "http":
            connector = HttpMCPClient(server_name)
        else:
            connector = StdioMCPClient(server_name)

        connector.connect(effective_cfg)
        registry.register(server_name, connector)

        try:
            entries = wrap_mcp_tools(server_name, connector)
            registry._tool_entries.extend(entries)
        except Exception:
            continue

    return registry


def _materialize_server_config(server_cfg: dict, config: dict) -> dict:
    effective_cfg = dict(server_cfg)
    env = dict(effective_cfg.get("env", {}) or {})
    env_from = effective_cfg.get("env_from", {}) or {}
    env.update(ToolPolicy.env_from_mapping(env_from, config))
    effective_cfg["env"] = env
    return effective_cfg
