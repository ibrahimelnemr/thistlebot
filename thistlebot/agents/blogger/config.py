from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...storage.paths import base_dir
from ...storage.state import load_config as _load_main_config

# Template defaults — no sensitive values here so this can be published.
# "site" is resolved at runtime from the main thistlebot config
# (~/.thistlebot/config.json → wordpress.blog) or from the agent's own
# runtime config (~/.thistlebot/agents/blogger/config.json → site).
DEFAULT_BLOGGER_CONFIG: dict[str, Any] = {
    "topic": (
        "Recent developments in AI, existential questions about artificial intelligence, "
        "and their implications for software developers"
    ),
    "post_status": "publish",
    "schedule": {
        "enabled": False,
        "cron": "0 9,21 * * *",
        "timezone": "UTC",
    },
    "workflow": {
        "research_max_iterations": 12,
        "draft_max_iterations": 10,
        "publish_max_iterations": 8,
    },
}


def agents_dir() -> Path:
    return base_dir() / "agents"


def blogger_dir() -> Path:
    return agents_dir() / "blogger"


def blogger_config_path() -> Path:
    return blogger_dir() / "config.json"


def blogger_runs_dir() -> Path:
    return blogger_dir() / "runs"


def _resolve_site(blogger_cfg: dict[str, Any]) -> str | None:
    """Resolve the WordPress site from blogger config or main config."""
    site = blogger_cfg.get("site")
    if isinstance(site, str) and site.strip():
        return site.strip()

    # Fall back to the main thistlebot config wordpress.blog
    main_cfg = _load_main_config()
    wp = main_cfg.get("wordpress", {})
    if isinstance(wp, dict):
        blog = wp.get("blog")
        if isinstance(blog, str) and blog.strip():
            return blog.strip()
    return None


def load_blogger_config() -> dict[str, Any]:
    """Load blogger config, creating defaults if the file does not exist.

    Sensitive values (like ``site``) are resolved from the agent's runtime
    config first, then from the main thistlebot config (``wordpress.blog``).
    This keeps the source-tree template free of personal data.
    """
    path = blogger_config_path()
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Merge defaults for any missing keys
        merged = dict(DEFAULT_BLOGGER_CONFIG)
        merged.update(raw)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_BLOGGER_CONFIG, indent=2), encoding="utf-8")
        merged = dict(DEFAULT_BLOGGER_CONFIG)

    # Resolve site from runtime config or main config
    merged["site"] = _resolve_site(merged)
    return merged


def create_run_dir() -> Path:
    """Create a timestamped run directory and return it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = blogger_runs_dir() / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, data: dict[str, Any]) -> None:
    meta_path = run_dir / "meta.json"
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def list_runs() -> list[Path]:
    """List all run directories sorted newest first."""
    runs_dir = blogger_runs_dir()
    if not runs_dir.exists():
        return []
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    dirs.sort(reverse=True)
    return dirs
