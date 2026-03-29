from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .base import ToolResult
from .policy import ToolPolicy


class NativeTools:
    def __init__(self, policy: ToolPolicy, config: dict[str, Any] | None = None) -> None:
        self.policy = policy
        self.config = config or {}

    def list_dir(self, payload: dict[str, Any]) -> ToolResult:
        target = str(payload.get("path") or ".")
        try:
            resolved = self.policy.resolve_workspace_path(target)
            if not resolved.exists():
                return ToolResult(ok=False, content="", error=f"Path does not exist: {target}")
            if not resolved.is_dir():
                return ToolResult(ok=False, content="", error=f"Path is not a directory: {target}")

            children = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
            rendered = [f"{child.name}/" if child.is_dir() else child.name for child in children]
            content = "\n".join(rendered)
            clipped, truncated = self.policy.normalize_output(content, self.policy.max_file_chars)
            return ToolResult(ok=True, content=clipped, data={"path": str(resolved), "entries": rendered}, truncated=truncated)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def read_file(self, payload: dict[str, Any]) -> ToolResult:
        target = payload.get("path")
        if not target:
            return ToolResult(ok=False, content="", error="Missing required argument: path")
        try:
            resolved = self.policy.resolve_workspace_path(str(target))
            if not resolved.exists():
                return ToolResult(ok=False, content="", error=f"Path does not exist: {target}")
            if not resolved.is_file():
                return ToolResult(ok=False, content="", error=f"Path is not a file: {target}")

            raw = resolved.read_text(encoding="utf-8")
            selected = self.policy.read_lines(raw, _as_int(payload.get("start_line")), _as_int(payload.get("end_line")))
            clipped, truncated = self.policy.normalize_output(selected, self.policy.max_file_chars)
            return ToolResult(ok=True, content=clipped, data={"path": str(resolved)}, truncated=truncated)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def write_file(self, payload: dict[str, Any]) -> ToolResult:
        target = payload.get("path")
        if not target:
            return ToolResult(ok=False, content="", error="Missing required argument: path")
        content = str(payload.get("content") or "")
        append = bool(payload.get("append", False))
        try:
            resolved = self.policy.resolve_workspace_path(str(target))
            self.policy.ensure_dir(resolved)
            mode = "a" if append else "w"
            with resolved.open(mode, encoding="utf-8") as handle:
                handle.write(content)
            return ToolResult(ok=True, content=f"Wrote {len(content)} chars to {resolved}", data={"path": str(resolved)})
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def edit_file(self, payload: dict[str, Any]) -> ToolResult:
        target = payload.get("path")
        old_text = payload.get("old_text")
        new_text = payload.get("new_text")
        if not target:
            return ToolResult(ok=False, content="", error="Missing required argument: path")
        if old_text is None or new_text is None:
            return ToolResult(ok=False, content="", error="Missing required arguments: old_text/new_text")

        try:
            resolved = self.policy.resolve_workspace_path(str(target))
            if not resolved.exists() or not resolved.is_file():
                return ToolResult(ok=False, content="", error=f"File not found: {target}")

            raw = resolved.read_text(encoding="utf-8")
            old_value = str(old_text)
            new_value = str(new_text)
            if old_value not in raw:
                return ToolResult(ok=False, content="", error="old_text not found in file")
            updated = raw.replace(old_value, new_value, 1)
            resolved.write_text(updated, encoding="utf-8")
            return ToolResult(ok=True, content=f"Edited file: {resolved}", data={"path": str(resolved)})
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def exec(self, payload: dict[str, Any]) -> ToolResult:
        command = str(payload.get("command") or "").strip()
        if not command:
            return ToolResult(ok=False, content="", error="Missing required argument: command")
        if self.policy.command_denied(command):
            return ToolResult(ok=False, content="", error="Command blocked by safety policy")
        if self.policy.command_requires_approval(command):
            return ToolResult(ok=False, content="", error="Command requires explicit approval")

        cwd_raw = str(payload.get("cwd") or ".")
        timeout = float(payload.get("timeout_seconds") or self.policy.exec_timeout_seconds)

        try:
            cwd = self.policy.resolve_workspace_path(cwd_raw)
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part)
            output = output or "[no output]"
            clipped, truncated = self.policy.normalize_output(output, self.policy.max_exec_output_chars)
            ok = proc.returncode == 0
            result = ToolResult(
                ok=ok,
                content=clipped,
                data={"return_code": proc.returncode, "cwd": str(cwd), "command": command},
                truncated=truncated,
            )
            if not ok:
                result.error = f"Command failed with exit code {proc.returncode}"
            return result
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, content="", error=f"Command timed out after {timeout}s")
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def workspace_exists(path: Path) -> bool:
    return path.exists() and path.is_dir()
