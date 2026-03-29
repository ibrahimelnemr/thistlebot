from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class StepMiddleware(Protocol):
    def wrap_registry(
        self,
        *,
        step: dict[str, Any],
        runtime_config: dict[str, Any],
        registry: "ToolRegistry",
    ) -> "ToolRegistry":
        """Return a registry wrapper for this step."""


class MiddlewareChain:
    def __init__(self, middlewares: list[StepMiddleware] | None = None) -> None:
        self._middlewares = middlewares or []

    def apply(
        self,
        *,
        step: dict[str, Any],
        runtime_config: dict[str, Any],
        registry: "ToolRegistry",
    ) -> "ToolRegistry":
        wrapped = registry
        for middleware in self._middlewares:
            wrapped = middleware.wrap_registry(step=step, runtime_config=runtime_config, registry=wrapped)
        return wrapped
