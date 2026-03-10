from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..storage.paths import agent_dir
from ..storage.state import load_config as load_main_config
from .loader import AgentDefinition


def runtime_agent_dir(name: str) -> Path:
    return agent_dir(name)


def agent_runs_dir(name: str) -> Path:
    return runtime_agent_dir(name) / "runs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _parse_env_value(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return ""
    if raw.lower() in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
        return _as_bool(raw)
    if raw.lower() in {"null", "none"}:
        return None
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
    except Exception:
        pass
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def _env_key_variants(agent_name: str, key: str) -> list[str]:
    key_token = key.replace("-", "_").replace(".", "_").upper()
    agent_token = agent_name.replace("-", "_").upper()
    return [
        f"THISTLEBOT_AGENT_{agent_token}_{key_token}",
        f"THISTLEBOT_AGENT_{key_token}",
    ]


def _load_agent_env(agent_def: AgentDefinition) -> None:
    candidates = [
        Path.cwd() / ".env",
        agent_def.root / ".env",
        runtime_agent_dir(agent_def.name) / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            load_dotenv(path, override=False)


def _apply_env_overrides(agent_name: str, config: dict[str, Any], required_keys: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    keys = set(out.keys()) | set(required_keys)
    for key in keys:
        for env_key in _env_key_variants(agent_name, key):
            raw = os.getenv(env_key)
            if raw is None:
                continue
            out[key] = _parse_env_value(raw)
            break
    return out


def _resolve_site_from_main_config(config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("site"), str) and str(config.get("site")).strip():
        return config
    main_cfg = load_main_config()
    wp = main_cfg.get("wordpress", {}) if isinstance(main_cfg.get("wordpress"), dict) else {}
    blog = wp.get("blog")
    if isinstance(blog, str) and blog.strip():
        updated = copy.deepcopy(config)
        updated["site"] = blog.strip()
        return updated
    return config


def load_agent_config(
    name: str,
    agent_def: AgentDefinition,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name != agent_def.name:
        raise ValueError(f"Agent name mismatch: expected {agent_def.name}, got {name}")

    _load_agent_env(agent_def)

    base_defaults = agent_def.defaults()
    merged = _apply_env_overrides(name, base_defaults, agent_def.required_config())
    merged = _deep_merge(merged, config_overrides or {})
    merged = _resolve_site_from_main_config(merged)

    missing = [key for key in agent_def.required_config() if merged.get(key) in {None, ""}]
    if missing:
        expected = ", ".join(missing)
        raise RuntimeError(
            f"Missing required agent config keys: {expected}. "
            "Set THISTLEBOT_AGENT_<AGENT>_<KEY> in .env or environment."
        )

    return merged


def create_run_dir(name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = agent_runs_dir(name) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def list_runs(name: str) -> list[Path]:
    runs_dir = agent_runs_dir(name)
    if not runs_dir.exists():
        return []
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    dirs.sort(reverse=True)
    return dirs


def find_resumable_run(name: str) -> Path | None:
    for run_dir in list_runs(name):
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


def get_run_dir(name: str, run_id: str | None = None) -> Path | None:
    runs = list_runs(name)
    if not runs:
        return None
    if run_id is None:
        return runs[0]
    for run_dir in runs:
        if run_dir.name == run_id:
            return run_dir
    return None


def save_run_metadata(run_dir: Path, data: dict[str, Any]) -> None:
    meta_path = run_dir / "meta.json"
    payload = dict(data)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
