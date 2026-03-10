from __future__ import annotations

import fnmatch
import json
from typing import Any

from ...integrations.mcp.registry import MCPRegistry
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

    if config.get("mcp", {}).get("enabled") and mcp_registry is not None:
        for mcp_entry in mcp_registry.tool_entries():
            registry.register(mcp_entry)

    if isinstance(tool_spec, dict) and tool_spec:
        registry = _filter_registry_by_spec(registry, tool_spec)

    return registry


def _filter_registry_by_spec(registry: ToolRegistry, tool_spec: dict[str, Any]) -> ToolRegistry:
    native_patterns = tool_spec.get("native") if isinstance(tool_spec.get("native"), list) else []
    mcp_patterns = tool_spec.get("mcp") if isinstance(tool_spec.get("mcp"), list) else []
    allow_filesystem = bool(tool_spec.get("filesystem", False))
    allow_exec = bool(tool_spec.get("exec", False))

    selected = ToolRegistry()
    for name, entry in registry._tools.items():
        is_native = entry.source == "native"
        is_mcp = entry.source == "mcp"

        if is_native:
            if name == "exec" and not allow_exec:
                continue
            if name in {"list_dir", "read_file", "write_file", "edit_file"} and not allow_filesystem:
                continue
            if native_patterns and not any(fnmatch.fnmatch(name, pattern) for pattern in native_patterns):
                continue

        if is_mcp and mcp_patterns and not any(fnmatch.fnmatch(name, pattern) for pattern in mcp_patterns):
            continue

        selected.register(entry)
    return selected


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
                name="wordpress.rest.list_sites",
                description="List WordPress.com sites available to the configured REST token.",
                input_schema={"type": "object", "properties": {}},
            ),
            execute=native.wordpress_rest_list_sites,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.sites",
                description="List WordPress.com sites available to the configured token.",
                input_schema={"type": "object", "properties": {}},
            ),
            execute=native.wordpress_rest_list_sites,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.rest.list_posts",
                description="List posts for a WordPress site via REST API.",
                input_schema={
                    "type": "object",
                    "required": ["site"],
                    "properties": {
                        "site": {"type": "string"},
                        "number": {"type": "integer"},
                        "status": {"type": "string"},
                    },
                },
            ),
            execute=native.wordpress_rest_list_posts,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.list_posts",
                description="List posts for a WordPress site.",
                input_schema={
                    "type": "object",
                    "required": ["site"],
                    "properties": {
                        "site": {"type": "string"},
                        "number": {"type": "integer"},
                        "status": {"type": "string"},
                    },
                },
            ),
            execute=native.wordpress_rest_list_posts,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.rest.create_post",
                description="Create a WordPress post via REST API.",
                input_schema={
                    "type": "object",
                    "required": ["site", "title", "content"],
                    "properties": {
                        "site": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                        "tags": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "categories": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                    },
                },
                risk_level="medium",
            ),
            execute=native.wordpress_rest_create_post,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.create_post",
                description="Create a WordPress post.",
                input_schema={
                    "type": "object",
                    "required": ["site", "title", "content"],
                    "properties": {
                        "site": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                        "tags": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "categories": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                    },
                },
                risk_level="medium",
            ),
            execute=native.wordpress_rest_create_post,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.rest.update_post",
                description="Update a WordPress post via REST API.",
                input_schema={
                    "type": "object",
                    "required": ["site", "post_id"],
                    "properties": {
                        "site": {"type": "string"},
                        "post_id": {"type": "integer"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                    },
                },
                risk_level="medium",
            ),
            execute=native.wordpress_rest_update_post,
            source="native",
        )
    )
    registry.register(
        ToolEntry(
            spec=ToolSpec(
                name="wordpress.rest.get_post",
                description="Get a WordPress post by ID via REST API.",
                input_schema={
                    "type": "object",
                    "required": ["site", "post_id"],
                    "properties": {
                        "site": {"type": "string"},
                        "post_id": {"type": "integer"},
                    },
                },
            ),
            execute=native.wordpress_rest_get_post,
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
