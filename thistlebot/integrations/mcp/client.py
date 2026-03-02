from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .connector import MCPConnector


class StdioMCPClient(MCPConnector):
    def __init__(self, name: str) -> None:
        self.name = name
        self._config: dict[str, Any] = {}
        self._connected = False
        self._last_error: str | None = None

    def connect(self, config: dict) -> None:
        self._config = config
        self._connected = True
        self._last_error = None

    def list_tools(self) -> list[dict]:
        if not self._connected:
            raise RuntimeError(f"MCP connector '{self.name}' is not connected")
        return _run_async(self._list_tools_async())

    def invoke(self, tool_name: str, payload: dict) -> dict:
        if not self._connected:
            raise RuntimeError(f"MCP connector '{self.name}' is not connected")
        return _run_async(self._invoke_async(tool_name, payload))

    def status(self) -> dict:
        return {
            "name": self.name,
            "connected": self._connected,
            "transport": self._config.get("transport", "stdio"),
            "last_error": self._last_error,
        }

    def close(self) -> None:
        self._connected = False

    async def _list_tools_async(self) -> list[dict]:
        try:
            session = await _open_session(self._config)
            async with session as client:
                response = await client.list_tools()
                tools = _extract_tools(response)
                return [
                    {
                        "name": _get_attr(tool, "name") or "",
                        "description": _get_attr(tool, "description") or "MCP tool",
                        "input_schema": _get_attr(tool, "inputSchema") or {"type": "object", "properties": {}},
                    }
                    for tool in tools
                    if _get_attr(tool, "name")
                ]
        except Exception as exc:
            self._last_error = str(exc)
            raise

    async def _invoke_async(self, tool_name: str, payload: dict) -> dict:
        try:
            session = await _open_session(self._config)
            async with session as client:
                response = await client.call_tool(tool_name, payload)
                return _normalize_call_result(response)
        except Exception as exc:
            self._last_error = str(exc)
            raise


class HttpMCPClient(MCPConnector):
    def __init__(self, name: str) -> None:
        self.name = name
        self._config: dict[str, Any] = {}
        self._connected = False

    def connect(self, config: dict) -> None:
        self._config = config
        self._connected = True

    def list_tools(self) -> list[dict]:
        raise RuntimeError("HTTP MCP transport is not implemented yet")

    def invoke(self, tool_name: str, payload: dict) -> dict:
        raise RuntimeError("HTTP MCP transport is not implemented yet")

    def status(self) -> dict:
        return {"name": self.name, "connected": self._connected, "transport": "http"}

    def close(self) -> None:
        self._connected = False


async def _open_session(config: dict):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        raise RuntimeError("MCP Python SDK not installed. Install package 'mcp'.") from exc

    command = config.get("command")
    if not command:
        raise RuntimeError("Missing MCP server command")

    args = list(config.get("args", []))
    env = {str(k): str(v) for k, v in (config.get("env", {}) or {}).items()}

    server_params = StdioServerParameters(command=command, args=args, env=env or None)

    stdio_ctx = stdio_client(server_params)
    read_stream, write_stream = await stdio_ctx.__aenter__()
    session_ctx = ClientSession(read_stream, write_stream)
    session = await session_ctx.__aenter__()
    initialize = getattr(session, "initialize")
    if inspect.iscoroutinefunction(initialize):
        await initialize()
    else:
        maybe = initialize()
        if inspect.isawaitable(maybe):
            await maybe

    class _SessionWrapper:
        async def __aenter__(self):
            return session

        async def __aexit__(self, exc_type, exc, tb):
            await session_ctx.__aexit__(exc_type, exc, tb)
            await stdio_ctx.__aexit__(exc_type, exc, tb)

    return _SessionWrapper()


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def _extract_tools(response: Any) -> list[Any]:
    if response is None:
        return []
    if isinstance(response, list):
        return response
    tools = _get_attr(response, "tools")
    if isinstance(tools, list):
        return tools
    return []


def _normalize_call_result(response: Any) -> dict:
    if isinstance(response, dict):
        return response
    content = _get_attr(response, "content")
    if content is None:
        return {"content": str(response)}
    return {"content": content}


def _get_attr(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
