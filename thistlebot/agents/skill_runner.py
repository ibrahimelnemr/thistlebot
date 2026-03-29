from __future__ import annotations

from ..core.agent_runtime import run_tool_agent
from ..core.tools.registry import ToolRegistry, filter_by_allowlist_denylist
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
        registry = filter_by_allowlist_denylist(registry, skill.allowed_tools, None)

    max_iterations = int(skill.metadata.get("default-max-iterations", 8))

    return run_tool_agent(
        client=client,
        registry=registry,
        model=model,
        messages=[{"role": "system", "content": instructions}],
        max_iterations=max_iterations,
    )
