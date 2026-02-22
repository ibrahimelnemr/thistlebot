from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..storage.paths import sessions_dir


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or sessions_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def append_message(self, session_id: str, message: dict[str, str]) -> None:
        path = self.root / f"{session_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=True) + "\n")

    def reset_session(self, session_id: str) -> None:
        path = self.root / f"{session_id}.jsonl"
        if path.exists():
            path.unlink()

    def read_session(self, session_id: str) -> Iterable[dict[str, str]]:
        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
