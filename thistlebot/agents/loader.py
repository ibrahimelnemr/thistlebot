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

    def description(self) -> str:
        value = self.manifest.get("description")
        return str(value) if isinstance(value, str) else ""

    def defaults(self) -> dict[str, Any]:
        cfg = self.manifest.get("config")
        if not isinstance(cfg, dict):
            return {}
        defaults = cfg.get("defaults")
        return defaults if isinstance(defaults, dict) else {}

    def required_config(self) -> list[str]:
        cfg = self.manifest.get("config")
        if not isinstance(cfg, dict):
            return []
        required = cfg.get("required")
        if not isinstance(required, list):
            return []
        return [str(item) for item in required if str(item).strip()]

    def schedule(self) -> dict[str, Any]:
        schedule = self.manifest.get("schedule")
        return schedule if isinstance(schedule, dict) else {}

    def workflow_overrides(self) -> dict[str, Any]:
        overrides = self.manifest.get("workflow_overrides")
        return overrides if isinstance(overrides, dict) else {}

    def tools(self) -> dict[str, Any]:
        tools = self.manifest.get("tools")
        return tools if isinstance(tools, dict) else {}

    def hooks(self) -> dict[str, Any]:
        hooks = self.manifest.get("hooks")
        return hooks if isinstance(hooks, dict) else {}

    def actions(self) -> dict[str, Any]:
        actions = self.manifest.get("actions")
        return actions if isinstance(actions, dict) else {}

    def default_workflow_name(self) -> str:
        workflows = self.manifest.get("workflows")
        if not isinstance(workflows, dict):
            raise ValueError(f"Invalid workflows map for agent '{self.name}'")
        default_name = workflows.get("default")
        if isinstance(default_name, str) and default_name.strip():
            return default_name.strip()
        for key in workflows.keys():
            if key != "default":
                return str(key)
        raise ValueError(f"No workflows configured for agent '{self.name}'")

    def load_prompt(self, prompt_name: str) -> str:
        skill = self.load_skill(prompt_name)
        return skill.instructions

    def load_skill(self, skill_name: str) -> "SkillDefinition":
        from .skill_loader import load_skill, SkillDefinition  # noqa: F401
        return load_skill(skill_name, self._skill_search_paths())

    def _skill_search_paths(self) -> list[Path]:
        paths: list[Path] = [self.root / "skills"]
        plugin_root = self.manifest.get("_plugin_root")
        if isinstance(plugin_root, str) and plugin_root.strip():
            paths.append(Path(plugin_root) / "skills")
        return paths

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
    if agents_root is not None and agents_root.name == agent_name:
        root = agents_root
    else:
        root = agents_root or (Path(__file__).resolve().parent / agent_name)

    agent_md_path = root / "AGENT.md"
    if not agent_md_path.exists():
        raise FileNotFoundError(f"Agent definition not found: {agent_md_path}")
    from .agent_md_loader import load_from_agent_md
    return load_from_agent_md(root, agent_name)
