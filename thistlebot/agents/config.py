from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..storage.paths import agent_dir
from .loader import AgentDefinition


def runtime_agent_dir(name: str) -> Path:
    return agent_dir(name)


def agent_runs_dir(name: str) -> Path:
    return runtime_agent_dir(name) / "runs"


def runtime_agent_config_path(name: str) -> Path:
    return runtime_agent_dir(name) / "config.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_agent_config(
    name: str,
    agent_def: AgentDefinition,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name != agent_def.name:
        raise ValueError(f"Agent name mismatch: expected {agent_def.name}, got {name}")

    base_defaults = agent_def.defaults()
    runtime_cfg_path = runtime_agent_config_path(name)
    runtime_cfg: dict[str, Any] = {}
    if runtime_cfg_path.exists():
        try:
            loaded = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                runtime_cfg = loaded
        except Exception:
            runtime_cfg = {}

    merged = _deep_merge(base_defaults, runtime_cfg)
    merged = _deep_merge(merged, config_overrides or {})

    missing = [key for key in agent_def.required_config() if merged.get(key) in {None, ""}]
    if missing:
        expected = ", ".join(missing)
        config_path = runtime_agent_config_path(name)
        raise RuntimeError(
            f"Missing required agent config keys: {expected}. "
            f"Run 'thistlebot agent {name} setup' to configure automatically, "
            f"or set values with 'thistlebot agent {name} config set'. "
            f"Runtime config path: {config_path}."
        )

    return merged


def save_agent_runtime_config(name: str, config: dict[str, Any]) -> Path:
    path = runtime_agent_config_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return path


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
