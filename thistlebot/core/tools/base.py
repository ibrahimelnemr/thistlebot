from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolExecutor = Callable[[dict[str, Any]], "ToolResult"]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: str = "low"


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: dict[str, Any] | None = None
    error: str | None = None
    truncated: bool = False

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "ok": self.ok,
            "content": self.content,
            "truncated": self.truncated,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class ToolEntry:
    spec: ToolSpec
    execute: ToolExecutor
    source: str = "native"
    metadata: dict[str, Any] = field(default_factory=dict)
