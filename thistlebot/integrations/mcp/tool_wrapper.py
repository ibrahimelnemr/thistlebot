from __future__ import annotations

from typing import Any

from ...core.tools.base import ToolEntry, ToolResult, ToolSpec
from .connector import MCPConnector


def wrap_mcp_tools(server_name: str, connector: MCPConnector) -> list[ToolEntry]:
    entries: list[ToolEntry] = []
    for tool in connector.list_tools():
        tool_name = str(tool.get("name") or "").strip()
        if not tool_name:
            continue
        wrapped_name = f"mcp.{server_name}.{tool_name}"
        description = str(tool.get("description") or f"MCP tool {tool_name}")
        input_schema = tool.get("input_schema") or {"type": "object", "properties": {}}

        def _executor(payload: dict[str, Any], *, _tool_name: str = tool_name) -> ToolResult:
            try:
                result = connector.invoke(_tool_name, payload)
                content = result.get("content")
                if not isinstance(content, str):
                    content = str(content)
                return ToolResult(ok=True, content=content, data=result)
            except Exception as exc:
                return ToolResult(ok=False, content="", error=str(exc))

        entries.append(
            ToolEntry(
                spec=ToolSpec(
                    name=wrapped_name,
                    description=description,
                    input_schema=input_schema,
                    risk_level="medium",
                ),
                execute=_executor,
                source=f"mcp:{server_name}",
            )
        )
    return entries
