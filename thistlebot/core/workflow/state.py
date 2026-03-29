from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class WorkflowState:
    path: Path
    workflow_name: str
    data: dict[str, Any]

    @classmethod
    def load_or_create(cls, path: Path, workflow_name: str) -> "WorkflowState":
        if path.exists():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                parsed = {}
            if isinstance(parsed, dict) and parsed.get("status") in {"running", "failed_quality_gate"}:
                parsed.setdefault("runtime", {})
                parsed["runtime"].setdefault("revision_number", int(parsed.get("revision_count", 0)) + 1)
                return cls(path=path, workflow_name=workflow_name, data=parsed)

        data: dict[str, Any] = {
            "workflow": workflow_name,
            "status": "running",
            "started_at": ts(),
            "current_step_index": 0,
            "steps": {},
            "step_attempts": {},
            "artifacts": {},
            "revision_count": 0,
            "runtime": {"revision_number": 1},
        }
        state = cls(path=path, workflow_name=workflow_name, data=data)
        state.save()
        return state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")


def append_event(events_path: Path, payload: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()
