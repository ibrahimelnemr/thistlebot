from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..utils.io import read_json, write_json
from .paths import config_path, ensure_base_dirs, prompts_dir


def load_default_config() -> dict[str, Any]:
    defaults_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
    data = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    return data or {}


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return load_default_config()
    return read_json(path)


def write_config(config: dict[str, Any], force: bool = False) -> bool:
    path = config_path()
    if path.exists() and not force:
        return False
    write_json(path, config)
    return True


def ensure_prompt_files(force: bool = False) -> None:
    prompts_dir().mkdir(parents=True, exist_ok=True)
    system_path = prompts_dir() / "system.md"
    base_path = prompts_dir() / "base.md"

    if force or not system_path.exists():
        system_path.write_text("You are Thistlebot, a lightweight assistant.\n", encoding="utf-8")
    if force or not base_path.exists():
        base_path.write_text("Use concise, helpful responses.\n", encoding="utf-8")


def setup_storage(force: bool = False) -> dict[str, Any]:
    ensure_base_dirs()
    config = load_default_config()
    write_config(config, force=force)
    ensure_prompt_files(force=force)
    return config


def reset_storage() -> dict[str, Any]:
    return setup_storage(force=True)
