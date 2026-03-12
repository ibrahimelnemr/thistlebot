from __future__ import annotations

import os
from typing import Any

from .base import BaseLLMClient
from .ollama_client import OllamaClient
from .openai_compatible_client import OpenAICompatibleClient


def get_llm_provider(config: dict[str, Any]) -> str:
    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    provider = llm_cfg.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    return "ollama"


def get_default_model(config: dict[str, Any], fallback: str = "llama3") -> str:
    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    llm_model = llm_cfg.get("model")
    if isinstance(llm_model, str) and llm_model.strip():
        return llm_model.strip()

    ollama_cfg = config.get("ollama", {}) if isinstance(config, dict) else {}
    ollama_model = ollama_cfg.get("model")
    if isinstance(ollama_model, str) and ollama_model.strip():
        return ollama_model.strip()

    return fallback


def get_provider_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if isinstance(providers, dict):
        provider_cfg = providers.get(provider)
        if isinstance(provider_cfg, dict):
            return provider_cfg

    if provider == "ollama":
        legacy = config.get("ollama", {}) if isinstance(config, dict) else {}
        if isinstance(legacy, dict):
            return legacy

    return {}


def resolve_api_key(
    provider_cfg: dict[str, Any],
    *,
    default_env_name: str,
) -> str | None:
    api_key = provider_cfg.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    env_name = provider_cfg.get("api_key_env")
    if isinstance(env_name, str) and env_name.strip():
        env_value = os.getenv(env_name.strip())
        if env_value:
            return env_value

    fallback_env = os.getenv(default_env_name)
    return fallback_env or None


def resolve_openrouter_api_key(provider_cfg: dict[str, Any]) -> str | None:
    api_key = provider_cfg.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return None


def build_llm_client(config: dict[str, Any]) -> BaseLLMClient:
    provider = get_llm_provider(config)

    if provider == "ollama":
        ollama_cfg = get_provider_config(config, "ollama")
        base_url = str(ollama_cfg.get("base_url", "http://127.0.0.1:11434"))
        return OllamaClient(base_url)

    if provider == "openrouter":
        provider_cfg = get_provider_config(config, "openrouter")
        base_url = str(provider_cfg.get("base_url", "https://openrouter.ai/api/v1"))
        api_key = resolve_openrouter_api_key(provider_cfg)
        headers = dict(provider_cfg.get("default_headers") or {})
        app_name = provider_cfg.get("app_name")
        site_url = provider_cfg.get("site_url")
        if isinstance(app_name, str) and app_name.strip():
            headers.setdefault("X-Title", app_name.strip())
        if isinstance(site_url, str) and site_url.strip():
            headers.setdefault("HTTP-Referer", site_url.strip())
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, default_headers=headers)

    if provider == "openai_compatible":
        provider_cfg = get_provider_config(config, "openai_compatible")
        base_url = str(provider_cfg.get("base_url", "http://localhost:8000/v1"))
        api_key = resolve_api_key(provider_cfg, default_env_name="OPENAI_API_KEY")
        headers = dict(provider_cfg.get("default_headers") or {})
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, default_headers=headers)

    raise ValueError(f"Unsupported llm.provider: {provider}")