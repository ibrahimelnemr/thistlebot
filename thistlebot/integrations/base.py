from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tools.registry import ToolRegistry
    from ..core.workflow.fallback import FallbackResolver
    from ..core.workflow.middleware import StepMiddleware


class Integration(Protocol):
    name: str

    def register_tools(self, registry: "ToolRegistry") -> None:
        """Register integration-backed tools in the global tool registry."""

    def workflow_middlewares(self) -> list["StepMiddleware"]:
        """Provide workflow step middleware for this integration."""

    def fallback_resolvers(self) -> list["FallbackResolver"]:
        """Provide deterministic fallback resolvers for this integration."""
