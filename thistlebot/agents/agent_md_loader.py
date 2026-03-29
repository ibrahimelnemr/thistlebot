"""Parser for strict AGENT.md format."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_from_agent_md(root: Path, agent_name: str) -> "AgentDefinition":
    """Parse AGENT.md and return an AgentDefinition with equivalent manifest."""
    from .loader import AgentDefinition

    path = root / "AGENT.md"
    text = path.read_text(encoding="utf-8")
    frontmatter: dict[str, Any] = {}
    body = ""

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            import yaml
            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip("\n")

    manifest = _build_manifest(frontmatter, body, root, path)
    _validate_agent_md_manifest(manifest, path)

    resolved_name = manifest.get("name")
    if isinstance(resolved_name, str) and resolved_name.strip() and resolved_name != agent_name:
        raise ValueError(
            f"Agent name mismatch in {path}: '{resolved_name}' != '{agent_name}'"
        )

    return AgentDefinition(name=agent_name, root=root, manifest=manifest)


def _build_manifest(
    frontmatter: dict[str, Any],
    body: str,
    root: Path,
    path: Path,
) -> dict[str, Any]:
    ext = frontmatter.get("x-thistlebot", {}) if isinstance(frontmatter.get("x-thistlebot"), dict) else {}

    manifest: dict[str, Any] = {}

    # Standard fields
    manifest["name"] = frontmatter.get("name", root.name)
    manifest["version"] = frontmatter.get("version", "")
    manifest["description"] = _scalar_str(frontmatter.get("description", ""))

    # System prompt preamble from markdown body
    if body.strip():
        manifest["_system_prompt"] = body.strip()

    # Tools: flat list → internal format
    tools_flat = frontmatter.get("tools") or []
    disallowed = frontmatter.get("disallowedTools") or []
    if isinstance(tools_flat, list):
        allow_patterns = [str(item) for item in tools_flat if str(item).strip()]
        deny_patterns = [str(item) for item in disallowed] if isinstance(disallowed, list) else []
        manifest["tools"] = {"allow": allow_patterns, "deny": deny_patterns}
    else:
        manifest["tools"] = {}

    # Model override
    if frontmatter.get("model"):
        manifest["model"] = frontmatter["model"]

    # x-thistlebot extensions
    if "config" in ext and isinstance(ext["config"], dict):
        manifest["config"] = ext["config"]
    else:
        manifest["config"] = {"defaults": {}, "required": []}

    if "schedule" in ext:
        manifest["schedule"] = ext["schedule"]

    if "workflow_overrides" in ext and isinstance(ext["workflow_overrides"], dict):
        manifest["workflow_overrides"] = ext["workflow_overrides"]
    elif "workflow" in ext:
        wf_ext = ext["workflow"] if isinstance(ext["workflow"], dict) else {}
        explicit_overrides = wf_ext.get("overrides")
        if isinstance(explicit_overrides, dict):
            overrides = explicit_overrides
        else:
            overrides = {k: v for k, v in wf_ext.items() if k not in {"default", "aliases", "overrides"}}
        if overrides:
            manifest["workflow_overrides"] = overrides

    if "hooks" in ext:
        manifest["hooks"] = ext["hooks"]

    if "actions" in ext:
        manifest["actions"] = ext["actions"]

    # Workflow resolution: build the workflows dict from workflows/ directory
    wf_ext = ext.get("workflow", {}) if isinstance(ext.get("workflow"), dict) else {}
    default_wf_name = wf_ext.get("default") if isinstance(wf_ext.get("default"), str) else None
    aliases = wf_ext.get("aliases", {}) if isinstance(wf_ext.get("aliases"), dict) else {}

    workflows_dir = root / "workflows"
    workflows: dict[str, Any] = {}
    if workflows_dir.exists():
        for wf_file in sorted(workflows_dir.glob("*.json")):
            wf_name = wf_file.stem
            workflows[wf_name] = f"workflows/{wf_file.name}"

    if default_wf_name and default_wf_name in workflows:
        workflows["default"] = default_wf_name
    elif workflows and "default" not in workflows:
        # Auto-pick first workflow as default
        first = next(iter(workflows))
        workflows["default"] = first

    if aliases:
        manifest["workflow_aliases"] = aliases

    manifest["workflows"] = workflows

    plugin_root = _discover_plugin_root(root)
    if plugin_root is not None:
        manifest["_plugin_root"] = str(plugin_root)

    return manifest


def _scalar_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value) if value is not None else ""


def _validate_agent_md_manifest(manifest: dict[str, Any], path: Path) -> None:
    if not manifest.get("name"):
        raise ValueError(f"AGENT.md missing 'name' field: {path}")
    if not manifest.get("workflows"):
        raise ValueError(
            f"AGENT.md agent '{manifest.get('name')}' has no workflows in workflows/ directory: {path}"
        )
    skills_dir = path.parent / "skills"
    plugin_root = manifest.get("_plugin_root")
    has_plugin_skills = isinstance(plugin_root, str) and (Path(plugin_root) / "skills").exists()
    if not skills_dir.exists() and not has_plugin_skills:
        raise ValueError(
            f"AGENT.md agent '{manifest.get('name')}' has no skills directory: {skills_dir}"
        )


def _discover_plugin_root(agent_root: Path) -> Path | None:
    # Expected plugin layout: <plugin_root>/agents/<agent_name>/AGENT.md
    agents_dir = agent_root.parent
    if agents_dir.name != "agents":
        return None
    plugin_root = agents_dir.parent
    marker = plugin_root / ".thistlebot-plugin" / "plugin.json"
    if marker.exists():
        return plugin_root
    return None
