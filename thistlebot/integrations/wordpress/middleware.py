from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ...core.tools.base import ToolResult

if TYPE_CHECKING:
    from ...core.tools.registry import ToolRegistry


@dataclass
class WordPressPublishGuard:
    """Enforce one successful publish call per step and optional draft mode."""

    def wrap_registry(
        self,
        *,
        step: dict[str, Any],
        runtime_config: dict[str, Any],
        registry: "ToolRegistry",
    ) -> "ToolRegistry":
        _ = step  # middleware contract allows step-aware behavior later
        enforce_draft_mode = _should_enforce_draft_mode(runtime_config)
        return _GuardedRegistry(base=registry, enforce_draft_mode=enforce_draft_mode)


class _GuardedRegistry:
    def __init__(self, *, base: "ToolRegistry", enforce_draft_mode: bool) -> None:
        self._base = base
        self._tools = getattr(base, "_tools", {})
        self._enforce_draft_mode = enforce_draft_mode
        self._successful_publish_count = 0

    def list_specs(self) -> list[Any]:
        return self._base.list_specs()

    def to_model_tools(self) -> list[dict[str, Any]]:
        return self._base.to_model_tools()

    def list_tool_names(self) -> list[str]:
        return self._base.list_tool_names()

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        if tool_name != "wordpress.create_post":
            return self._base.invoke(tool_name, payload)

        if self._successful_publish_count >= 1:
            return ToolResult(
                ok=False,
                content="",
                error="Publish guard: at most one successful wordpress.create_post call is allowed per step.",
            )

        call_payload = dict(payload or {})
        if self._enforce_draft_mode:
            call_payload["status"] = "draft"

        result = self._base.invoke(tool_name, call_payload)
        if result.ok:
            self._successful_publish_count += 1
        return result


def _should_enforce_draft_mode(runtime_config: dict[str, Any]) -> bool:
    enforce_flag = runtime_config.get("enforce_draft_mode")
    if isinstance(enforce_flag, bool):
        if not enforce_flag:
            return False
    elif isinstance(enforce_flag, str) and enforce_flag.strip().lower() in {"false", "0", "no", "off"}:
        return False

    publish_mode = str(runtime_config.get("publish_mode") or "draft").strip().lower()
    return publish_mode != "publish"
