from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    allowed_tools: list[str]
    metadata: dict[str, Any]
    skill_dir: Path


def load_skill(skill_name: str, search_paths: list[Path]) -> SkillDefinition:
    """Search for skills/<skill_name>/SKILL.md in given paths."""
    for base in search_paths:
        candidate = base / skill_name / "SKILL.md"
        if candidate.exists():
            return _parse_skill_md(candidate)
    searched = ", ".join(str(p / skill_name / "SKILL.md") for p in search_paths)
    raise FileNotFoundError(f"Skill '{skill_name}' not found. Searched: {searched}")


def _parse_skill_md(path: Path) -> SkillDefinition:
    """Parse YAML frontmatter + markdown body from a SKILL.md file."""
    text = path.read_text(encoding="utf-8")
    frontmatter: dict[str, Any] = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            import yaml  # deferred import
            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip("\n")

    name = str(frontmatter.get("name") or path.parent.name)
    description = str(frontmatter.get("description") or "")

    raw_tools = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools") or []
    if isinstance(raw_tools, str):
        allowed_tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
    elif isinstance(raw_tools, list):
        allowed_tools = [str(t) for t in raw_tools]
    else:
        allowed_tools = []

    metadata: dict[str, Any] = {
        k: v for k, v in frontmatter.items()
        if k not in {"name", "description", "allowed-tools", "allowed_tools"}
    }

    return SkillDefinition(
        name=name,
        description=description,
        instructions=body,
        allowed_tools=allowed_tools,
        metadata=metadata,
        skill_dir=path.parent,
    )


def list_skills(search_paths: list[Path]) -> list[SkillDefinition]:
    """Return all skills found in the given search paths."""
    seen: set[str] = set()
    skills: list[SkillDefinition] = []
    for base in search_paths:
        if not base.exists():
            continue
        for skill_dir in sorted(base.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            skill = _parse_skill_md(skill_md)
            if skill.name not in seen:
                seen.add(skill.name)
                skills.append(skill)
    return skills
