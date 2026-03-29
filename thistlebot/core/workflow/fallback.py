from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


@dataclass
class FallbackResult:
    output_text: str
    events: list[dict[str, Any]]


class FallbackResolver(Protocol):
    def resolve(
        self,
        *,
        step: dict[str, Any],
        step_id: str,
        resolved_inputs: dict[str, Any],
        registry: "ToolRegistry",
        output_text: str,
        events: list[dict[str, Any]],
    ) -> FallbackResult | None:
        """Return fallback result or None when this resolver does not apply."""


class CompositeFallbackResolver:
    def __init__(self, resolvers: list[FallbackResolver] | None = None) -> None:
        self._resolvers = resolvers or []

    def resolve(
        self,
        *,
        step: dict[str, Any],
        step_id: str,
        resolved_inputs: dict[str, Any],
        registry: "ToolRegistry",
        output_text: str,
        events: list[dict[str, Any]],
    ) -> FallbackResult:
        current_output = output_text
        current_events = list(events)

        for resolver in self._resolvers:
            resolved = resolver.resolve(
                step=step,
                step_id=step_id,
                resolved_inputs=resolved_inputs,
                registry=registry,
                output_text=current_output,
                events=current_events,
            )
            if resolved is not None:
                current_output = resolved.output_text
                current_events = resolved.events

        return FallbackResult(output_text=current_output, events=current_events)
