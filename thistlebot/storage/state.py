from __future__ import annotations

import copy
from typing import Any

from ..utils.io import read_json, write_json
from .paths import config_path, ensure_base_dirs, prompts_dir

DEFAULT_CONFIG: dict[str, Any] = {
    "gateway": {
        "host": "127.0.0.1",
        "port": 7788,
    },
    "llm": {
        "provider": "ollama",
        "model": "qwen3:0.6b",
    },
    "providers": {
        "ollama": {
            "base_url": "http://localhost:11434",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_key": None,
            "app_name": "thistlebot",
            "site_url": None,
            "default_headers": {},
        },
        "openai_compatible": {
            "base_url": "http://localhost:8000/v1",
            "api_key_env": "OPENAI_API_KEY",
            "api_key": None,
            "default_headers": {},
        },
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:0.6b",
    },
    "github": {
        "client_id": "Ov23likecnqtFvsVynK2",
        "token": None,
    },
    "wordpress": {
        "enabled": False,
        "client_id": None,
        "client_secret": None,
        "token": None,
        "expires_at": None,
        "redirect_uri": "http://127.0.0.1:8766/callback",
        "scope": "posts",
        "blog": None,
    },
    "tools": {
        "runtime": {
            "enabled": True,
            "max_iterations": 8,
        },
        "native": {
            "enabled": True,
            "workspace_root": "~/.thistlebot/workspace",
            "max_file_chars": 12000,
            "exec": {
                "require_approval": False,
                "timeout_seconds": 45,
                "max_output_chars": 12000,
                "require_approval_for": [],
            },
        },
    },
    "mcp": {
        "enabled": False,
        "servers": {
            "filesystem": {
                "enabled": False,
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "~/.thistlebot/workspace"],
                "timeout_seconds": 30,
            },
            "git": {
                "enabled": False,
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-git", "--repository", "~/.thistlebot/workspace"],
                "timeout_seconds": 30,
            },
            "github": {
                "enabled": False,
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@github/github-mcp-server"],
                "env_from": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "github.token",
                },
                "timeout_seconds": 30,
            },
            "open-web-search": {
                "enabled": False,
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "open-websearch@latest"],
                "env": {
                    "MODE": "stdio",
                    "DEFAULT_SEARCH_ENGINE": "duckduckgo",
                },
                "timeout_seconds": 30,
            },
        },
    },
}


def load_default_config() -> dict[str, Any]:
    return normalize_config(copy.deepcopy(DEFAULT_CONFIG))


def _ensure_provider_defaults(
    providers_cfg: dict[str, Any],
    provider_name: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    provider_cfg = providers_cfg.get(provider_name)
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}
    providers_cfg[provider_name] = provider_cfg
    for key, value in defaults.items():
        if key not in provider_cfg:
            provider_cfg[key] = copy.deepcopy(value)
    return provider_cfg


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config or {})
    default_llm_cfg = DEFAULT_CONFIG.get("llm", {})
    default_providers_cfg = DEFAULT_CONFIG.get("providers", {})
    default_ollama_cfg = default_providers_cfg.get("ollama", {})

    llm_cfg = cfg.get("llm")
    if not isinstance(llm_cfg, dict):
        llm_cfg = {}
    cfg["llm"] = llm_cfg

    providers_cfg = cfg.get("providers")
    if not isinstance(providers_cfg, dict):
        providers_cfg = {}
    cfg["providers"] = providers_cfg

    legacy_ollama = cfg.get("ollama")
    if not isinstance(legacy_ollama, dict):
        legacy_ollama = {}
    cfg["ollama"] = legacy_ollama

    ollama_provider_cfg = providers_cfg.get("ollama")
    if not isinstance(ollama_provider_cfg, dict):
        ollama_provider_cfg = {}
    providers_cfg["ollama"] = ollama_provider_cfg

    if "base_url" not in ollama_provider_cfg and legacy_ollama.get("base_url"):
        ollama_provider_cfg["base_url"] = legacy_ollama.get("base_url")
    if "base_url" not in ollama_provider_cfg:
        ollama_provider_cfg["base_url"] = str(default_ollama_cfg.get("base_url", "http://localhost:11434"))

    if not llm_cfg.get("provider"):
        llm_cfg["provider"] = str(default_llm_cfg.get("provider", "ollama"))

    if not llm_cfg.get("model"):
        llm_cfg["model"] = legacy_ollama.get("model") or str(default_llm_cfg.get("model", "qwen3:0.6b"))

    if "base_url" not in legacy_ollama:
        legacy_ollama["base_url"] = ollama_provider_cfg["base_url"]
    if "model" not in legacy_ollama:
        legacy_ollama["model"] = llm_cfg["model"]

    _ensure_provider_defaults(
        providers_cfg,
        "openrouter",
        dict(default_providers_cfg.get("openrouter", {})),
    )
    _ensure_provider_defaults(
        providers_cfg,
        "openai_compatible",
        dict(default_providers_cfg.get("openai_compatible", {})),
    )

    wordpress_cfg = cfg.get("wordpress")
    if not isinstance(wordpress_cfg, dict):
        wordpress_cfg = {}

    legacy_wordpress_rest_cfg = cfg.get("wordpress_rest")
    if isinstance(legacy_wordpress_rest_cfg, dict):
        for key, value in legacy_wordpress_rest_cfg.items():
            wordpress_cfg.setdefault(key, value)

    legacy_wordpress_mcp_cfg = cfg.get("wordpress_mcp")
    if isinstance(legacy_wordpress_mcp_cfg, dict) and not wordpress_cfg.get("token"):
        # Best-effort migration for users who only had legacy WordPress config.
        for key in ("client_id", "token", "token_type", "expires_in", "expires_at"):
            value = legacy_wordpress_mcp_cfg.get(key)
            if value is not None:
                wordpress_cfg.setdefault(key, value)

    cfg["wordpress"] = wordpress_cfg
    for key, value in dict(DEFAULT_CONFIG.get("wordpress", {})).items():
        if key not in wordpress_cfg:
            wordpress_cfg[key] = copy.deepcopy(value)

    # Compatibility alias for older code paths.
    cfg["wordpress_rest"] = wordpress_cfg

    mcp_cfg = cfg.get("mcp")
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
    cfg["mcp"] = mcp_cfg
    default_mcp_cfg = dict(DEFAULT_CONFIG.get("mcp", {}))
    if "enabled" not in mcp_cfg:
        mcp_cfg["enabled"] = bool(default_mcp_cfg.get("enabled", False))

    servers_cfg = mcp_cfg.get("servers")
    if not isinstance(servers_cfg, dict):
        servers_cfg = {}
    mcp_cfg["servers"] = servers_cfg
    default_servers_cfg = default_mcp_cfg.get("servers", {})
    if isinstance(default_servers_cfg, dict):
        for server_name, server_defaults in default_servers_cfg.items():
            if server_name not in servers_cfg:
                servers_cfg[server_name] = copy.deepcopy(server_defaults)

    if "wpcom-mcp" in servers_cfg:
        servers_cfg.pop("wpcom-mcp", None)

    return cfg


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return load_default_config()
    return normalize_config(read_json(path))


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
