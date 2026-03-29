from __future__ import annotations

import fnmatch
import json
from typing import Any

from ...integrations.mcp.registry import MCPRegistry
from ...integrations.registry import discover_integrations
from .base import ToolEntry, ToolResult, ToolSpec
from .native import NativeTools
from .policy import ToolPolicy


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        self._tools[entry.spec.name] = entry

    def list_specs(self) -> list[ToolSpec]:
        return [entry.spec for entry in sorted(self._tools.values(), key=lambda item: item.spec.name)]

    def to_model_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in self.list_specs():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    },
                }
            )
        return tools

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        entry = self._tools.get(tool_name)
        if entry is None:
            return ToolResult(ok=False, content="", error=f"Unknown tool: {tool_name}")
        return entry.execute(payload)

    def list_tool_names(self) -> list[str]:
        return sorted(self._tools.keys())


def build_tool_registry(
    config: dict,
    mcp_registry: MCPRegistry | None = None,
    *,
    tool_spec: dict[str, Any] | None = None,
) -> ToolRegistry:
    policy = ToolPolicy.from_config(config)
    tools_cfg = config.get("tools", {})
    native_cfg = tools_cfg.get("native", {})
    native_enabled = bool(native_cfg.get("enabled", True))

    registry = ToolRegistry()
    if native_enabled:
        native = NativeTools(policy, config=config)
        _register_native_tools(registry, native)
    for integration in discover_integrations(config):
        integration.register_tools(registry)

    if config.get("mcp", {}).get("enabled") and mcp_registry is not None:
        for mcp_entry in mcp_registry.tool_entries():
            registry.register(mcp_entry)

    if isinstance(tool_spec, dict):
        allow_raw = tool_spec.get("allow")
        deny_raw = tool_spec.get("deny")
        allow = [str(item) for item in allow_raw] if isinstance(allow_raw, list) else None
        deny = [str(item) for item in deny_raw] if isinstance(deny_raw, list) else None
        registry = filter_by_allowlist_denylist(registry, allow, deny)

    return registry


def _register_native_tools(registry: ToolRegistry, native: NativeTools) -> None:
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="list_dir",
                description="List files and folders at a path relative to the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            ),
            execute=native.list_dir,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file from the workspace.",
                input_schema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                },
            ),
            execute=native.read_file,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="write_file",
                description="Write UTF-8 text to a file in the workspace.",
                input_schema={
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean"},
                    },
                },
                risk_level="medium",
            ),
            execute=native.write_file,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="edit_file",
                description="Replace the first matching text occurrence in a file.",
                input_schema={
                    "type": "object",
                    "required": ["path", "old_text", "new_text"],
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                },
                risk_level="medium",
            ),
            execute=native.edit_file,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="exec",
                description="Run a shell command in the workspace with safety limits.",
                input_schema={
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "number"},
                    },
                },
                risk_level="high",
            ),
            execute=native.exec,
            source="native",
        )
    )


def filter_by_allowlist_denylist(
    registry: ToolRegistry,
    allow: list[str] | None,
    deny: list[str] | None,
) -> ToolRegistry:
    """Filter using flat glob-pattern allow/deny lists.

    If *allow* is non-empty, only tools matching at least one allow pattern are kept.
    If *deny* is non-empty, tools matching any deny pattern are removed.
    """
    selected = ToolRegistry()
    for name, entry in registry._tools.items():
        if deny and any(fnmatch.fnmatch(name, pattern) for pattern in deny):
            continue
        if allow and not any(fnmatch.fnmatch(name, pattern) for pattern in allow):
            continue
        selected.register(entry)
    return selected


def normalize_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            decoded = json.loads(raw_args)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
    return {}
