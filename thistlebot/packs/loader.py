from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PluginManifest:
    name: str
    description: str
    version: str
    integrations: list[str]
    middleware: list[dict[str, Any]]
    fallback_resolvers: list[str]
    root: Path


def load_plugin(plugin_dir: Path) -> PluginManifest:
    manifest_path = plugin_dir / ".thistlebot-plugin" / "plugin.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Plugin manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid plugin manifest: {manifest_path}")

    name = str(payload.get("name") or plugin_dir.name)
    description = str(payload.get("description") or "")
    version = str(payload.get("version") or "0.0.0")
    integrations = payload.get("integrations") if isinstance(payload.get("integrations"), list) else []
    middleware = payload.get("middleware") if isinstance(payload.get("middleware"), list) else []
    fallback_resolvers = (
        payload.get("fallback_resolvers") if isinstance(payload.get("fallback_resolvers"), list) else []
    )

    return PluginManifest(
        name=name,
        description=description,
        version=version,
        integrations=[str(item) for item in integrations],
        middleware=[item for item in middleware if isinstance(item, dict)],
        fallback_resolvers=[str(item) for item in fallback_resolvers],
        root=plugin_dir,
    )


def discover_plugins(packs_root: Path) -> list[PluginManifest]:
    if not packs_root.exists() or not packs_root.is_dir():
        return []

    plugins: list[PluginManifest] = []
    for child in sorted(packs_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        manifest_path = child / ".thistlebot-plugin" / "plugin.json"
        if not manifest_path.exists():
            continue
        plugins.append(load_plugin(child))
    return plugins
