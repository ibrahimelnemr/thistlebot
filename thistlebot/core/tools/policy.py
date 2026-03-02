from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ...storage.paths import workspace_dir


@dataclass
class ToolPolicy:
    workspace_root: Path
    max_file_chars: int = 12000
    max_exec_output_chars: int = 12000
    exec_timeout_seconds: float = 45.0
    exec_approval_enabled: bool = False
    exec_denylist: tuple[str, ...] = (
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",
        "shutdown",
        "reboot",
    )
    exec_requires_approval: tuple[str, ...] = (
        "git push",
        "git reset --hard",
        "git clean -fd",
        "rm -rf",
    )

    @classmethod
    def from_config(cls, config: dict) -> "ToolPolicy":
        native_cfg = config.get("tools", {}).get("native", {})
        workspace_raw = native_cfg.get("workspace_root") or str(workspace_dir())
        workspace = Path(workspace_raw).expanduser().resolve()
        exec_cfg = native_cfg.get("exec", {})
        return cls(
            workspace_root=workspace,
            max_file_chars=int(native_cfg.get("max_file_chars", 12000)),
            max_exec_output_chars=int(exec_cfg.get("max_output_chars", 12000)),
            exec_timeout_seconds=float(exec_cfg.get("timeout_seconds", 45.0)),
            exec_approval_enabled=bool(exec_cfg.get("require_approval", False)),
            exec_requires_approval=tuple(exec_cfg.get("require_approval_for", cls.exec_requires_approval)),
        )

    def resolve_workspace_path(self, candidate: str) -> Path:
        candidate_path = (self.workspace_root / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).expanduser().resolve()
        root = self.workspace_root.resolve()
        if root == candidate_path or root in candidate_path.parents:
            return candidate_path
        raise ValueError(f"Path is outside workspace root: {candidate}")

    def normalize_output(self, content: str, max_chars: int | None = None) -> tuple[str, bool]:
        limit = max_chars if max_chars is not None else self.max_exec_output_chars
        if len(content) <= limit:
            return content, False
        clipped = content[:limit]
        return f"{clipped}\n...[truncated]", True

    def command_denied(self, command: str) -> bool:
        lowered = command.lower()
        return any(token in lowered for token in self.exec_denylist)

    def command_requires_approval(self, command: str) -> bool:
        if not self.exec_approval_enabled:
            return False
        lowered = command.lower()
        return any(token in lowered for token in self.exec_requires_approval)

    @staticmethod
    def read_lines(content: str, start_line: int | None, end_line: int | None) -> str:
        if start_line is None and end_line is None:
            return content
        lines = content.splitlines()
        start = max(1, start_line or 1)
        end = end_line or len(lines)
        if start > end:
            return ""
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def env_from_mapping(env_mapping: dict[str, str], config: dict) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for env_key, config_path in env_mapping.items():
            value = _get_by_dotted_path(config, config_path)
            if value is None:
                continue
            resolved[env_key] = str(value)
        return resolved


def _get_by_dotted_path(config: dict, dotted_path: str) -> str | None:
    cursor: object = config
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    if cursor is None:
        return None
    return str(cursor)


def stringify_command(args: Iterable[str]) -> str:
    return " ".join(args)
