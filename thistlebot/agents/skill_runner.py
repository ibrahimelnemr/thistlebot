from __future__ import annotations

import fnmatch
from typing import Any

from ..core.agent_runtime import run_tool_agent
from ..core.tools.registry import ToolRegistry
from ..llm.base import BaseLLMClient
from .skill_loader import SkillDefinition


def run_skill_standalone(
    skill: SkillDefinition,
    *,
    client: BaseLLMClient,
    registry: ToolRegistry,
    model: str,
    arguments: str = "",
) -> str:
    """Execute a skill as a standalone single-prompt task."""
    instructions = skill.instructions.replace("$ARGUMENTS", arguments)

    if skill.allowed_tools:
        registry = _filter_by_allowlist(registry, skill.allowed_tools)

    max_iterations = int(skill.metadata.get("default-max-iterations", 8))

    return run_tool_agent(
        client=client,
        registry=registry,
        model=model,
        messages=[{"role": "system", "content": instructions}],
        max_iterations=max_iterations,
    )


def _filter_by_allowlist(registry: ToolRegistry, allow_patterns: list[str]) -> ToolRegistry:
    from ..core.tools.registry import ToolRegistry as TR
    filtered = TR()
    for name, entry in registry._tools.items():
        if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
            filtered.register(entry)
    return filtered
