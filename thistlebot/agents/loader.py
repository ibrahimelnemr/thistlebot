from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    root: Path
    manifest: dict[str, Any]

    def prompt_path(self, prompt_name: str) -> Path:
        prompts = self.manifest.get("prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError(f"Invalid prompts map for agent '{self.name}'")
        rel = prompts.get(prompt_name)
        if not isinstance(rel, str) or not rel.strip():
            raise KeyError(f"Prompt '{prompt_name}' not found for agent '{self.name}'")
        path = (self.root / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path

    def load_prompt(self, prompt_name: str) -> str:
        return self.prompt_path(prompt_name).read_text(encoding="utf-8")

    def workflow_path(self, workflow_name: str) -> Path:
        workflows = self.manifest.get("workflows", {})
        if not isinstance(workflows, dict):
            raise ValueError(f"Invalid workflows map for agent '{self.name}'")
        rel = workflows.get(workflow_name)
        if not isinstance(rel, str) or not rel.strip():
            raise KeyError(f"Workflow '{workflow_name}' not found for agent '{self.name}'")
        path = (self.root / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")
        return path

    def load_workflow(self, workflow_name: str) -> dict[str, Any]:
        path = self.workflow_path(workflow_name)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Workflow definition must be an object: {path}")
        return data


def load_agent_definition(agent_name: str, *, agents_root: Path | None = None) -> AgentDefinition:
    root = agents_root or (Path(__file__).resolve().parent / agent_name)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Agent manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest JSON: {manifest_path}")

    _validate_manifest(manifest, manifest_path)
    return AgentDefinition(name=agent_name, root=root, manifest=manifest)


def _validate_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    required = ("name", "prompts", "workflows")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Manifest missing keys {missing}: {manifest_path}")

    if not isinstance(manifest.get("prompts"), dict):
        raise ValueError(f"Manifest prompts must be an object: {manifest_path}")

    if not isinstance(manifest.get("workflows"), dict):
        raise ValueError(f"Manifest workflows must be an object: {manifest_path}")
