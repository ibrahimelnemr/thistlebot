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
        "Latest AI news"
    ),
    "post_status": "draft",
    "schedule": {
        "enabled": False,
        "cron": "0 9,21 * * *",
        "timezone": "UTC",
        "times_per_day": None,
        "interval_minutes": None,
        "interval_seconds": None,
    },
    "workflow": {
        "research_max_iterations": 12,
        "draft_max_iterations": 10,
        "edit_max_iterations": 8,
        "verify_max_iterations": 6,
        "publish_max_iterations": 8,
        "max_revisions": 2,
        "verify_pass_token": "VERDICT: PASS",
    },
    "ideas": {
        "auto_refresh_before_publish": True,
        "refresh_count": 6,
        "query_count": 8,
        "max_iterations": 14,
        "prefer_web": True,
        "min_refresh_interval_minutes": 180,
        "failure_selected_action": "new",
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


def blogger_ideas_dir() -> Path:
    return blogger_dir() / "ideas"


def blogger_ideas_index_path() -> Path:
    return blogger_ideas_dir() / "index.json"


def blogger_ideas_markdown_path() -> Path:
    return blogger_ideas_dir() / "ideas.md"


def blogger_ideas_batches_dir() -> Path:
    return blogger_ideas_dir() / "batches"


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
        # Merge defaults for any missing keys.
        merged = dict(DEFAULT_BLOGGER_CONFIG)
        merged.update(raw)

        # Keep nested defaults even when users have partial nested config.
        for nested_key in ("schedule", "workflow", "ideas"):
            default_nested = DEFAULT_BLOGGER_CONFIG.get(nested_key)
            merged_nested = merged.get(nested_key)
            if isinstance(default_nested, dict):
                if isinstance(merged_nested, dict):
                    nested = dict(default_nested)
                    nested.update(merged_nested)
                    merged[nested_key] = nested
                else:
                    merged[nested_key] = dict(default_nested)
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


def find_resumable_run() -> Path | None:
    """Return the newest run with an active run_state.json status."""
    for run_dir in list_runs():
        state_path = run_dir / "run_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(state, dict) and state.get("status") == "running":
            return run_dir
    return None


def get_run_dir(run_id: str | None = None) -> Path | None:
    """Return run directory by id or latest when id is omitted."""
    runs = list_runs()
    if not runs:
        return None
    if run_id is None:
        return runs[0]
    for run_dir in runs:
        if run_dir.name == run_id:
            return run_dir
    return None
