from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..storage.paths import agent_memory_dir


@dataclass
class MemoryEntry:
    id: str
    timestamp: str
    title: str
    type: str
    workflow: str | None = None
    step: str | None = None
    run_id: str | None = None
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    def record(self, entry: MemoryEntry) -> MemoryEntry:
        ...

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        ...

    def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        ...

    def get(self, memory_id: str) -> MemoryEntry | None:
        ...


class JsonFileMemoryStore:
    def __init__(self, agent_name: str, *, root_dir: Path | None = None) -> None:
        self.agent_name = agent_name
        self.root_dir = root_dir or agent_memory_dir(agent_name)
        self.index_path = self.root_dir / "index.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def record(self, entry: MemoryEntry) -> MemoryEntry:
        index = self._load_index()
        index["entries"].append(asdict(entry))
        self._save_index(index)
        return entry

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        q = (query or "").strip().lower()
        entries = self._entries_sorted_newest()
        if not q:
            return entries[: max(limit, 0)]

        matched: list[MemoryEntry] = []
        for entry in entries:
            haystack = " ".join(
                [
                    entry.title,
                    entry.summary,
                    entry.type,
                    entry.workflow or "",
                    entry.step or "",
                    " ".join(entry.tags),
                ]
            ).lower()
            if q in haystack:
                matched.append(entry)
            if len(matched) >= limit:
                break
        return matched

    def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        return self._entries_sorted_newest()[: max(limit, 0)]

    def get(self, memory_id: str) -> MemoryEntry | None:
        for entry in self._entries_sorted_newest():
            if entry.id == memory_id:
                return entry
        return None

    def _entries_sorted_newest(self) -> list[MemoryEntry]:
        index = self._load_index()
        raw_entries = index.get("entries", [])
        parsed: list[MemoryEntry] = []
        if not isinstance(raw_entries, list):
            return parsed

        for item in raw_entries:
            if isinstance(item, dict):
                try:
                    parsed.append(MemoryEntry(**item))
                except TypeError:
                    continue

        parsed.sort(key=lambda item: item.timestamp, reverse=True)
        return parsed

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"version": 1, "agent": self.agent_name, "entries": []}

        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "agent": self.agent_name, "entries": []}

        if not isinstance(data, dict):
            return {"version": 1, "agent": self.agent_name, "entries": []}
        if "entries" not in data or not isinstance(data.get("entries"), list):
            data["entries"] = []
        data.setdefault("version", 1)
        data.setdefault("agent", self.agent_name)
        return data

    def _save_index(self, payload: dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def new_memory_entry(
    *,
    title: str,
    type: str,
    workflow: str | None = None,
    step: str | None = None,
    run_id: str | None = None,
    tags: list[str] | None = None,
    summary: str = "",
    artifact_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        title=title,
        type=type,
        workflow=workflow,
        step=step,
        run_id=run_id,
        tags=list(tags or []),
        summary=summary,
        artifact_path=artifact_path,
        metadata=dict(metadata or {}),
    )
