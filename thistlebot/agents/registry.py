from __future__ import annotations

from pathlib import Path

from .loader import AgentDefinition, load_agent_definition


def discover_agents(agents_root: Path | None = None) -> list[AgentDefinition]:
    root = agents_root or Path(__file__).resolve().parent
    found: list[AgentDefinition] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if not (child / "AGENT.md").exists():
            continue
        found.append(load_agent_definition(child.name, agents_root=child))
    return found


def list_agent_names(agents_root: Path | None = None) -> list[str]:
    return [agent.name for agent in discover_agents(agents_root=agents_root)]
