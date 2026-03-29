from __future__ import annotations

import time
import subprocess
import webbrowser
import importlib
import shutil
import re
import sys
from pathlib import Path
from typing import Any, Optional
import json
import urllib.parse

import httpx
import questionary
import typer
from rich.console import Console
from rich.markdown import Markdown

from .core.chat_client import stream_chat
from .core.gateway import run_gateway
from .core.gateway_lifecycle import ensure_gateway
from .core.meeting_graph import MeetingConfig, run_meeting_graph
from .core.tools.registry import build_tool_registry
from .integrations.github.oauth import login_with_device_flow, poll_for_token
from .integrations.mcp.registry import build_mcp_registry
from .integrations.wordpress.rest_client import WordPressRestClient
from .integrations.wordpress.rest_oauth import (
    DEFAULT_REDIRECT_URI as WORDPRESS_DEFAULT_REDIRECT_URI,
    login_with_authorization_code_flow as wordpress_login_flow,
    token_expired as wordpress_token_expired,
)
from .llm.factory import get_default_model, get_llm_provider, get_provider_config, resolve_api_key, resolve_openrouter_api_key
from .llm.openai_compatible_client import OpenAICompatibleClient
from .storage.state import load_config, reset_storage, setup_storage, write_config

app = typer.Typer(add_completion=False)
github_app = typer.Typer(help="GitHub integrations")
ollama_app = typer.Typer(help="Ollama diagnostics")
llm_app = typer.Typer(help="LLM provider diagnostics")
mcp_app = typer.Typer(help="MCP integrations")
mcp_enable_app = typer.Typer(help="Enable MCP servers")
mcp_disable_app = typer.Typer(help="Disable MCP servers")
wordpress_app = typer.Typer(help="WordPress integrations (REST)")
agent_app = typer.Typer(help="Persistent agent management")
skill_app = typer.Typer(help="Skill management and execution")
app.add_typer(github_app, name="github")
app.add_typer(ollama_app, name="ollama")
app.add_typer(llm_app, name="llm")
app.add_typer(mcp_app, name="mcp")
mcp_app.add_typer(mcp_enable_app, name="enable")
mcp_app.add_typer(mcp_disable_app, name="disable")
app.add_typer(wordpress_app, name="wordpress")
app.add_typer(agent_app, name="agent")
app.add_typer(skill_app, name="skill")

RICH_CONSOLE = Console()
THINK_OPEN_MARKERS = (
    "<think>",
    "<thinking>",
    "<THINK>",
    "<THINKING>",
    "<|begin_of_thought|>",
)
THINK_CLOSE_MARKERS = (
    "</think>",
    "</thinking>",
    "</THINK>",
    "</THINKING>",
    "<|end_of_thought|>",
)
THISTLEBOT_ASCII = r"""
 _______ _     _ _     _   _       _           _
|__   __| |   (_) |   | | | |     | |         | |
   | |  | |__  _| |___| |_| | ___ | |__   ___ | |_
   | |  | '_ \| | / __| __| |/ _ \| '_ \ / _ \| __|
   | |  | | | | | \__ \ |_| | (_) | |_) | (_) | |_
   |_|  |_| |_|_|_|___/\__|_|\___/|_.__/ \___/ \__|
"""


def _print_banner() -> None:
    typer.secho(THISTLEBOT_ASCII, fg="green")


def _find_first_marker(text: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
    best_index = -1
    best_marker = ""
    for marker in markers:
        index = text.find(marker)
        if index == -1:
            continue
        if best_index == -1 or index < best_index:
            best_index = index
            best_marker = marker
    if best_index == -1:
        return None
    return best_index, best_marker


class StreamRenderer:
    def __init__(
        self,
        *,
        prefix: str = "",
        color: str | None = None,
        render_markdown: bool = True,
        show_loading: bool = True,
    ) -> None:
        self.prefix = prefix
        self.color = color
        self.render_markdown = render_markdown
        self.show_loading = show_loading
        self.buffer = ""
        self.in_thinking_block = False
        self.visible_parts: list[str] = []
        self._thinking_label_shown = False
        self._thinking_stream_started = False
        self._needs_prefix_after_thinking = False
        self._started = False
        self._loading_shown = False

    def start(self) -> None:
        if self.show_loading:
            self._loading_shown = True
            if self.color:
                typer.secho(f"{self.prefix}[loading...]", fg=self.color, nl=False)
            else:
                typer.echo(f"{self.prefix}[loading...]", nl=False)
        else:
            if self.color:
                typer.secho(self.prefix, fg=self.color, nl=False)
            else:
                typer.echo(self.prefix, nl=False)

    def feed(self, chunk: str) -> None:
        if not self._started:
            self._start_response()
        self.buffer += chunk
        self._process_buffer(final=False)

    def finish(self) -> str:
        if not self._started:
            self._start_response()
        self._process_buffer(final=True)
        visible_text = "".join(self.visible_parts).strip()

        if self.render_markdown:
            if visible_text:
                typer.echo("")
                RICH_CONSOLE.print(Markdown(visible_text))
            else:
                typer.echo("")
        else:
            typer.echo("")

        return visible_text

    def _emit_visible(self, text: str) -> None:
        if not text:
            return
        self.visible_parts.append(text)
        if self.render_markdown:
            return
        if self._needs_prefix_after_thinking:
            if self.color:
                typer.secho(self.prefix, fg=self.color, nl=False)
            else:
                typer.echo(self.prefix, nl=False)
            self._needs_prefix_after_thinking = False
        if self.color:
            typer.secho(text, fg=self.color, nl=False)
        else:
            typer.echo(text, nl=False)

    def _emit_thinking_marker(self) -> None:
        if self._thinking_label_shown:
            return
        self._thinking_label_shown = True
        typer.secho(" [thinking...]", fg="yellow")
        if self.color:
            typer.secho("  » ", fg=self.color, dim=True, nl=False)
        else:
            typer.secho("  » ", dim=True, nl=False)

    def _emit_thinking_text(self, text: str) -> None:
        if not text:
            return
        self._thinking_stream_started = True
        style = "italic"
        if self.color:
            style = f"italic {self.color}"
        RICH_CONSOLE.print(text, style=style, end="")

    def _finish_thinking_block(self) -> None:
        if not self._thinking_stream_started:
            return
        self._thinking_stream_started = False
        self._needs_prefix_after_thinking = not self.render_markdown
        typer.echo("")

    def _start_response(self) -> None:
        if self._started:
            return
        self._started = True
        if self._loading_shown:
            typer.echo("")
            if self.color:
                typer.secho(self.prefix, fg=self.color, nl=False)
            else:
                typer.echo(self.prefix, nl=False)
        elif self.prefix:
            if self.color:
                typer.secho(self.prefix, fg=self.color, nl=False)
            else:
                typer.echo(self.prefix, nl=False)

    def _process_buffer(self, *, final: bool) -> None:
        max_open = max(len(marker) for marker in THINK_OPEN_MARKERS)
        max_close = max(len(marker) for marker in THINK_CLOSE_MARKERS)

        while True:
            if self.in_thinking_block:
                closing = _find_first_marker(self.buffer, THINK_CLOSE_MARKERS)
                if closing is None:
                    if final:
                        self._emit_thinking_text(self.buffer)
                        self._finish_thinking_block()
                        self.buffer = ""
                    else:
                        if len(self.buffer) > max_close:
                            safe = self.buffer[:-max_close]
                            self._emit_thinking_text(safe)
                            self.buffer = self.buffer[-max_close:]
                    break
                close_index, close_marker = closing
                thinking_part = self.buffer[:close_index]
                self._emit_thinking_text(thinking_part)
                self.buffer = self.buffer[close_index + len(close_marker) :]
                self.in_thinking_block = False
                self._finish_thinking_block()
                continue

            opening = _find_first_marker(self.buffer, THINK_OPEN_MARKERS)
            if opening is None:
                if final:
                    self._emit_visible(self.buffer)
                    self.buffer = ""
                else:
                    if len(self.buffer) > max_open:
                        safe = self.buffer[:-max_open]
                        self._emit_visible(safe)
                        self.buffer = self.buffer[-max_open:]
                break

            open_index, open_marker = opening
            before = self.buffer[:open_index]
            self._emit_visible(before)
            self.buffer = self.buffer[open_index + len(open_marker) :]
            self.in_thinking_block = True
            self._emit_thinking_marker()


def _gateway_url_from_config(config: dict) -> str:
    gateway_cfg = config.get("gateway", {})
    host = gateway_cfg.get("host", "127.0.0.1")
    port = int(gateway_cfg.get("port", 7788))
    return f"http://{host}:{port}"


def _gateway_host_port_from_config(config: dict) -> tuple[str, int]:
    gateway_cfg = config.get("gateway", {})
    host = gateway_cfg.get("host", "127.0.0.1")
    port = int(gateway_cfg.get("port", 7788))
    return host, port


def _render_tool_event(event: dict) -> None:
    event_type = str(event.get("event") or "")
    if event_type == "stream_error":
        error_type = str(event.get("error_type") or "Error")
        message = str(event.get("message") or "unknown stream failure")
        typer.secho(f"[stream error] {error_type}: {message}", fg="red")
        return

    if event_type == "tool_call":
        tool_name = str(event.get("tool") or "unknown")
        args = event.get("args")
        args_text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
        if len(args_text) > 180:
            args_text = f"{args_text[:180]}..."
        typer.secho(f"\n[tool call] {tool_name}", fg="yellow")
        typer.secho(f"  args: {args_text}", fg="yellow")
        return

    if event_type == "tool_result":
        tool_name = str(event.get("tool") or "unknown")
        ok = bool(event.get("ok"))
        color = "green" if ok else "red"
        status_text = "ok" if ok else "error"
        typer.secho(f"[tool result] {tool_name} ({status_text})", fg=color)
        error_text = event.get("error")
        if error_text:
            typer.secho(f"  error: {error_text}", fg="red")
        content = str(event.get("content") or "").strip()
        if content:
            preview = content if len(content) <= 220 else f"{content[:220]}..."
            typer.secho(f"  output: {preview}", fg=color)
        return

    raw = json.dumps(event, ensure_ascii=False)
    typer.secho(f"[tool event] {raw}", fg="yellow")


def _print_stream_failure_context(*, config: dict, model: str, gateway_url: str, exc: Exception) -> None:
    provider = get_llm_provider(config)
    message = str(exc)

    typer.secho("Streaming request failed.", fg="red", err=True)
    typer.echo(f"Provider: {provider}", err=True)
    typer.echo(f"Model: {model}", err=True)
    typer.echo(f"Gateway: {gateway_url.rstrip('/')}/chat/stream", err=True)
    typer.echo(f"Failure: {message}", err=True)

    lowered = message.lower()
    if "incomplete chunked read" in lowered or "terminated early" in lowered:
        typer.echo("Likely cause: gateway exception during SSE generation.", err=True)
    typer.echo("Next step: inspect gateway stderr and ~/.thistlebot/logs/gateway.log", err=True)


def _extract_text_from_tool_response(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts)


def _decode_json_text(text: str) -> Any:
    raw = text.strip()
    if not raw:
        return None
    candidates = [raw]
    if raw.startswith("```") and raw.endswith("```"):
        stripped = "\n".join(raw.splitlines()[1:-1]).strip()
        candidates.append(stripped)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _collect_sites_like(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = str(key).lower()
                if lowered in {"sites", "blogs"} and isinstance(child, list):
                    for item in child:
                        if isinstance(item, dict):
                            found.append(item)
                walk(child)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return found


def _extract_sites_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    structured = result.get("structuredContent")
    if structured is not None:
        sites = _collect_sites_like(structured)
        if sites:
            return sites

    text = _extract_text_from_tool_response(result)
    parsed = _decode_json_text(text)
    if parsed is not None:
        sites = _collect_sites_like(parsed)
        if sites:
            return sites
    return []


def _wordpress_config(config: dict) -> dict[str, Any]:
    integrations_cfg = config.get("integrations")
    if not isinstance(integrations_cfg, dict):
        integrations_cfg = {}
        config["integrations"] = integrations_cfg
    cfg = integrations_cfg.get("wordpress")
    if not isinstance(cfg, dict):
        cfg = {}
        integrations_cfg["wordpress"] = cfg
    return cfg


def _wordpress_client(config: dict) -> WordPressRestClient:
    wp_cfg = _wordpress_config(config)
    token = wp_cfg.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("WordPress token missing. Run 'thistlebot wordpress login'.")
    if wordpress_token_expired(wp_cfg):
        raise RuntimeError("WordPress token expired. Run 'thistlebot wordpress login' to refresh it.")
    timeout = wp_cfg.get("timeout_seconds")
    timeout_value = float(timeout) if isinstance(timeout, (int, float)) else 30.0
    return WordPressRestClient(access_token=token, timeout_seconds=timeout_value)


def _wordpress_site_ref(config: dict, explicit_site: str | None) -> str:
    if explicit_site:
        return explicit_site
    wp_cfg = _wordpress_config(config)
    blog = wp_cfg.get("blog")
    if isinstance(blog, str) and blog.strip():
        return blog.strip()
    blog_url = wp_cfg.get("blog_url")
    if isinstance(blog_url, str) and blog_url.strip():
        return blog_url.strip()
    blog_id = wp_cfg.get("blog_id")
    if isinstance(blog_id, (int, float)):
        return str(int(blog_id))
    if isinstance(blog_id, str) and blog_id.strip():
        return blog_id.strip()
    raise RuntimeError("Missing site/blog reference. Provide --site or set integrations.wordpress.blog in config.")


def _ollama_base_url_from_config(config: dict) -> str:
    ollama_cfg = get_provider_config(config, "ollama")
    return str(ollama_cfg.get("base_url", "http://localhost:11434")).rstrip("/")


def _openrouter_base_url_from_config(config: dict) -> str:
    cfg = get_provider_config(config, "openrouter")
    return str(cfg.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/")


def _openai_compatible_base_url_from_config(config: dict) -> str:
    cfg = get_provider_config(config, "openai_compatible")
    return str(cfg.get("base_url", "http://localhost:8000/v1")).rstrip("/")


def _discover_ollama_models(base_url: str, timeout: float = 5.0) -> tuple[bool, list[str], Optional[str]]:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, [], str(exc)

    payload = response.json() if response.content else {}
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = [item.get("name", "") for item in models if isinstance(item, dict) and item.get("name")]
    return True, names, None


def _discover_openai_compatible_models(
    base_url: str,
    api_key: str | None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[bool, list[str], Optional[str]]:
    try:
        client = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            default_headers=headers,
        )
        models = client.list_models()
    except Exception as exc:
        return False, [], str(exc)
    return True, models, None


def _no_models_found_message() -> str:
    return (
        "No models found. Check that Ollama is running and endpoint is configured properly "
        "in ~/.thistlebot/config.json (ollama.base_url)."
    )


def _no_provider_models_found_message(provider: str, base_url: str) -> str:
    return (
        f"No models found for provider '{provider}'. Check endpoint/auth configuration "
        f"in ~/.thistlebot/config.json (providers.{provider}.base_url={base_url})."
    )


def _select_provider(current_provider: str) -> str:
    choices = ["ollama", "openrouter", "openai_compatible"]
    default_provider = current_provider if current_provider in choices else "ollama"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo("Non-interactive terminal detected; keeping configured provider.")
        return default_provider

    try:
        selected = questionary.select(
            "Select LLM provider",
            choices=choices,
            default=default_provider,
        ).ask()
    except Exception:
        typer.echo("Interactive selector unavailable; keeping configured provider.")
        return default_provider

    if selected is None:
        typer.echo("Selection cancelled; keeping existing provider.")
        return default_provider
    return str(selected)


def _select_primary_model(models: list[str], current_model: str, provider: str) -> str:
    default_model = current_model if current_model in models else models[0]

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo("Non-interactive terminal detected; selecting model automatically.")
        return default_model

    try:
        selected = questionary.select(
            f"Select primary {provider} model",
            choices=models,
            default=default_model,
        ).ask()
    except Exception:
        typer.echo("Interactive selector unavailable; selecting model automatically.")
        return default_model

    if selected is None:
        typer.echo("Selection cancelled; keeping existing model.")
        return current_model or default_model
    return str(selected)


def _ask_text(prompt: str, default: str) -> str:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default
    try:
        answer = questionary.text(prompt, default=default).ask()
    except Exception:
        return default
    return str(answer or default).strip() or default


def _ask_api_key_strategy(default_env: str) -> tuple[str, str | None]:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default_env, None

    try:
        selected = questionary.select(
            "How should API credentials be configured?",
            choices=["Use environment variable", "Store API key in config"],
            default="Use environment variable",
        ).ask()
    except Exception:
        return default_env, None

    if selected == "Store API key in config":
        try:
            api_key = questionary.password("API key").ask()
        except Exception:
            api_key = None
        return default_env, str(api_key or "").strip() or None

    env_name = _ask_text("API key environment variable", default_env)
    return env_name, None


def _ensure_builtin_tools_defaults(config: dict[str, Any]) -> None:
    tools_cfg = config.get("tools")
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}
        config["tools"] = tools_cfg

    runtime_cfg = tools_cfg.get("runtime")
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
        tools_cfg["runtime"] = runtime_cfg
    runtime_cfg["enabled"] = True

    native_cfg = tools_cfg.get("native")
    if not isinstance(native_cfg, dict):
        native_cfg = {}
        tools_cfg["native"] = native_cfg
    native_cfg["enabled"] = True


def _ensure_open_websearch_server(config: dict[str, Any], *, enabled: bool) -> tuple[bool, dict[str, Any]]:
    mcp_cfg = config.get("mcp")
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
        config["mcp"] = mcp_cfg
    mcp_cfg["enabled"] = True

    servers_cfg = mcp_cfg.get("servers")
    if not isinstance(servers_cfg, dict):
        servers_cfg = {}
        mcp_cfg["servers"] = servers_cfg

    server_cfg = servers_cfg.get("open-websearch")
    if not isinstance(server_cfg, dict):
        server_cfg = {}
        servers_cfg["open-websearch"] = server_cfg

    server_cfg["enabled"] = bool(enabled)
    server_cfg.setdefault("transport", "stdio")
    server_cfg.setdefault("command", "npx")
    server_cfg.setdefault("args", ["-y", "open-websearch@latest"])
    server_cfg.setdefault("timeout_seconds", 30)
    env_cfg = server_cfg.get("env")
    if not isinstance(env_cfg, dict):
        env_cfg = {}
        server_cfg["env"] = env_cfg
    env_cfg.setdefault("MODE", "stdio")
    env_cfg.setdefault("DEFAULT_SEARCH_ENGINE", "duckduckgo")

    npx_available = shutil.which("npx") is not None
    return npx_available, server_cfg


def _finalize_setup_defaults(config: dict[str, Any]) -> None:
    _ensure_builtin_tools_defaults(config)
    npx_available, _ = _ensure_open_websearch_server(config, enabled=True)
    write_config(config, force=True)

    typer.echo("Built-in tools enabled: tools.runtime.enabled=true, tools.native.enabled=true")
    typer.echo("MCP enabled: mcp.enabled=true")
    typer.echo("Open Web Search enabled: mcp.servers.open-websearch.enabled=true")
    if npx_available:
        typer.echo("MCP check: npx detected; open-websearch is ready to use.")
    else:
        typer.echo("MCP check: npx not found.", err=True)
        typer.echo(
            "Install Node.js (includes npx), then run 'thistlebot setup' again or 'thistlebot mcp enable open-websearch'.",
            err=True,
        )


@app.command()
def setup(force: bool = typer.Option(False, "--force", help="Overwrite existing config/prompts")) -> None:
    _print_banner()
    setup_storage(force=force)
    config = load_config()
    typer.echo("Created ~/.thistlebot structure")
    current_provider = get_llm_provider(config)
    provider = _select_provider(current_provider)
    current_model = get_default_model(config)

    config.setdefault("llm", {})
    config.setdefault("providers", {})
    config.setdefault("ollama", {})
    config["llm"]["provider"] = provider

    if provider == "ollama":
        base_url = _ask_text("Ollama base URL", _ollama_base_url_from_config(config))
        typer.echo(f"Ollama base URL: {base_url}")

        reachable, models, error = _discover_ollama_models(base_url)
        if not reachable:
            typer.echo("Unable to reach Ollama at the configured endpoint.", err=True)
            if error:
                typer.echo(f"Connection error: {error}", err=True)
            typer.echo(_no_models_found_message(), err=True)
            return

        typer.echo("Ollama is running.")
        if not models:
            typer.echo(_no_models_found_message(), err=True)
            return

        typer.echo(f"Discovered {len(models)} model(s):")
        for model_name in models:
            typer.echo(f"- {model_name}")

        selected_model = _select_primary_model(models, current_model, "ollama")
        config["llm"]["model"] = selected_model
        config["providers"].setdefault("ollama", {})
        config["providers"]["ollama"]["base_url"] = base_url
        config["ollama"]["base_url"] = base_url
        config["ollama"]["model"] = selected_model
        _finalize_setup_defaults(config)
        typer.echo(f"Primary model set to: {selected_model}")
        return

    if provider == "openrouter":
        base_url = _ask_text("OpenRouter base URL", _openrouter_base_url_from_config(config))
        provider_cfg = get_provider_config(config, "openrouter")
        existing_key = str(provider_cfg.get("api_key") or "").strip()
        direct_key = existing_key or None
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                entered_key = questionary.password(
                    "OpenRouter API key (stored in config; leave blank to keep existing)"
                ).ask()
            except Exception:
                entered_key = None
            candidate_key = str(entered_key or "").strip()
            if candidate_key:
                direct_key = candidate_key
        provider_cfg["base_url"] = base_url
        provider_cfg["api_key"] = direct_key
        provider_cfg.setdefault("app_name", "thistlebot")
        provider_cfg.setdefault("site_url", None)
        provider_cfg.setdefault("default_headers", {})
        config["providers"]["openrouter"] = provider_cfg

        headers = dict(provider_cfg.get("default_headers") or {})
        app_name = provider_cfg.get("app_name")
        site_url = provider_cfg.get("site_url")
        if isinstance(app_name, str) and app_name.strip():
            headers.setdefault("X-Title", app_name.strip())
        if isinstance(site_url, str) and site_url.strip():
            headers.setdefault("HTTP-Referer", site_url.strip())

        api_key = resolve_openrouter_api_key(provider_cfg)
        reachable, models, error = _discover_openai_compatible_models(base_url, api_key, headers=headers)
        if not reachable:
            typer.echo("Unable to reach OpenRouter at the configured endpoint.", err=True)
            if error:
                typer.echo(f"Connection error: {error}", err=True)

        selected_model = current_model
        if models:
            typer.echo(f"Discovered {len(models)} model(s):")
            for model_name in models:
                typer.echo(f"- {model_name}")
            selected_model = _select_primary_model(models, current_model, "openrouter")
        else:
            typer.echo(_no_provider_models_found_message("openrouter", base_url), err=True)
            selected_model = _ask_text("Primary OpenRouter model", current_model or "openai/gpt-4o-mini")

        config["llm"]["model"] = selected_model
        _finalize_setup_defaults(config)
        typer.echo(f"Primary model set to: {selected_model}")
        return

    base_url = _ask_text("OpenAI-compatible base URL", _openai_compatible_base_url_from_config(config))
    provider_cfg = get_provider_config(config, "openai_compatible")
    env_name, direct_key = _ask_api_key_strategy(str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")))
    provider_cfg["base_url"] = base_url
    provider_cfg["api_key_env"] = env_name
    provider_cfg["api_key"] = direct_key
    provider_cfg.setdefault("default_headers", {})
    config["providers"]["openai_compatible"] = provider_cfg

    api_key = resolve_api_key(provider_cfg, default_env_name="OPENAI_API_KEY")
    headers = dict(provider_cfg.get("default_headers") or {})
    reachable, models, error = _discover_openai_compatible_models(base_url, api_key, headers=headers)
    if not reachable:
        typer.echo("Unable to reach OpenAI-compatible endpoint.", err=True)
        if error:
            typer.echo(f"Connection error: {error}", err=True)

    selected_model = current_model
    if models:
        typer.echo(f"Discovered {len(models)} model(s):")
        for model_name in models:
            typer.echo(f"- {model_name}")
        selected_model = _select_primary_model(models, current_model, "openai_compatible")
    else:
        typer.echo(_no_provider_models_found_message("openai_compatible", base_url), err=True)
        selected_model = _ask_text(
            "Primary OpenAI-compatible model",
            current_model or "meta-llama/Meta-Llama-3.1-8B-Instruct",
        )

    config["llm"]["model"] = selected_model
    _finalize_setup_defaults(config)
    typer.echo(f"Primary model set to: {selected_model}")


@app.command()
def reset() -> None:
    _print_banner()
    config = reset_storage()
    typer.echo("Reset ~/.thistlebot config and prompts to defaults")

    ollama_url = config.get("ollama", {}).get("base_url", "")
    if ollama_url:
        typer.echo(f"Ollama base URL: {ollama_url}")


@ollama_app.command("check")
def ollama_check() -> None:
    _print_banner()
    config = load_config()
    base_url = _ollama_base_url_from_config(config)

    typer.echo(f"Ollama base URL: {base_url}")
    reachable, models, error = _discover_ollama_models(base_url)

    if not reachable:
        typer.echo("Reachable: no")
        if error:
            typer.echo(f"Error: {error}", err=True)
        typer.echo(_no_models_found_message(), err=True)
        raise typer.Exit(code=1)

    typer.echo("Reachable: yes")
    typer.echo(f"Models found: {len(models)}")

    if not models:
        typer.echo(_no_models_found_message(), err=True)
        raise typer.Exit(code=1)

    for model_name in models:
        typer.echo(f"- {model_name}")

    configured_model = get_default_model(config) if get_llm_provider(config) == "ollama" else config.get("ollama", {}).get("model", "")
    if configured_model:
        typer.echo(f"Configured primary model: {configured_model}")


@llm_app.command("check")
def llm_check() -> None:
    _print_banner()
    config = load_config()
    provider = get_llm_provider(config)
    model = get_default_model(config)
    typer.echo(f"Provider: {provider}")
    typer.echo(f"Configured primary model: {model}")

    if provider == "ollama":
        base_url = _ollama_base_url_from_config(config)
        typer.echo(f"Endpoint: {base_url}")
        reachable, models, error = _discover_ollama_models(base_url)
    elif provider == "openrouter":
        base_url = _openrouter_base_url_from_config(config)
        cfg = get_provider_config(config, "openrouter")
        api_key = resolve_openrouter_api_key(cfg)
        headers = dict(cfg.get("default_headers") or {})
        app_name = cfg.get("app_name")
        site_url = cfg.get("site_url")
        if isinstance(app_name, str) and app_name.strip():
            headers.setdefault("X-Title", app_name.strip())
        if isinstance(site_url, str) and site_url.strip():
            headers.setdefault("HTTP-Referer", site_url.strip())
        typer.echo(f"Endpoint: {base_url}")
        reachable, models, error = _discover_openai_compatible_models(base_url, api_key, headers=headers)
    elif provider == "openai_compatible":
        base_url = _openai_compatible_base_url_from_config(config)
        cfg = get_provider_config(config, "openai_compatible")
        api_key = resolve_api_key(cfg, default_env_name="OPENAI_API_KEY")
        headers = dict(cfg.get("default_headers") or {})
        typer.echo(f"Endpoint: {base_url}")
        reachable, models, error = _discover_openai_compatible_models(base_url, api_key, headers=headers)
    else:
        typer.echo(f"Unknown provider: {provider}", err=True)
        raise typer.Exit(code=1)

    if not reachable:
        typer.echo("Reachable: no")
        if error:
            typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Reachable: yes")
    typer.echo(f"Models found: {len(models)}")
    if models:
        for model_name in models[:30]:
            typer.echo(f"- {model_name}")


@app.command()
def gateway(host: Optional[str] = None, port: Optional[int] = None) -> None:
    run_gateway(host=host, port=port)


@app.command()
def chat(
    session: str = typer.Option("default", "--session"),
    model: Optional[str] = typer.Option(None, "--model"),
    auto_gateway: bool = typer.Option(
        True,
        "--auto-gateway/--no-auto-gateway",
        help="Automatically start the gateway if it is not already running",
    ),
    gateway_start_timeout: float = typer.Option(
        15.0,
        "--gateway-start-timeout",
        min=1.0,
        help="Seconds to wait for an auto-started gateway to become healthy",
    ),
    render_markdown: bool = typer.Option(
        True,
        "--render-markdown/--no-render-markdown",
        help="Render assistant replies as markdown in the terminal",
    ),
) -> None:
    _print_banner()
    config = load_config()
    gateway_url = _gateway_url_from_config(config)
    gateway_host, gateway_port = _gateway_host_port_from_config(config)
    active_model = model or get_default_model(config)

    try:
        with ensure_gateway(
            gateway_url,
            gateway_host,
            gateway_port,
            autostart=auto_gateway,
            start_timeout=gateway_start_timeout,
        ):
            typer.echo("Type a message. Use :exit to quit, :reset to clear session, :model <name> to switch.")
            messages: list[dict[str, str]] = []

            while True:
                try:
                    prompt = typer.prompt("you")
                except typer.Abort:
                    break

                if prompt.strip() == ":exit":
                    break
                if prompt.strip() == ":reset":
                    messages = []
                    typer.echo("Session reset")
                    continue
                if prompt.startswith(":model "):
                    active_model = prompt.replace(":model ", "", 1).strip() or active_model
                    typer.echo(f"Model set to {active_model}")
                    continue

                messages.append({"role": "user", "content": prompt})
                try:
                    renderer = StreamRenderer(prefix="assistant> ", render_markdown=render_markdown)
                    renderer.start()
                    for chunk in stream_chat(
                        gateway_url,
                        messages,
                        active_model,
                        session,
                        on_event=_render_tool_event,
                    ):
                        renderer.feed(chunk)
                    assistant_content = renderer.finish()
                    if assistant_content:
                        messages.append({"role": "assistant", "content": assistant_content})
                except Exception as exc:
                    _print_stream_failure_context(config=config, model=active_model, gateway_url=gateway_url, exc=exc)
                    raise typer.Exit(code=1)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command()
def meeting(
    session: str = typer.Option("meeting", "--session"),
    model_a: Optional[str] = typer.Option(None, "--model-a"),
    model_b: Optional[str] = typer.Option(None, "--model-b"),
    auto_gateway: bool = typer.Option(
        True,
        "--auto-gateway/--no-auto-gateway",
        help="Automatically start the gateway if it is not already running",
    ),
    gateway_start_timeout: float = typer.Option(
        15.0,
        "--gateway-start-timeout",
        min=1.0,
        help="Seconds to wait for an auto-started gateway to become healthy",
    ),
    system_a: Optional[str] = typer.Option(
        None,
        "--system-a",
        help="Optional system prompt override for agent_a",
    ),
    system_b: Optional[str] = typer.Option(
        None,
        "--system-b",
        help="Optional system prompt override for agent_b",
    ),
    max_turns: int = typer.Option(0, "--max-turns", min=0, help="0 means run until Ctrl+C"),
    starter: str = typer.Option(
        "Let's discuss one practical idea we can improve today.",
        "--starter",
        help="First message used to start the agent-to-agent conversation",
    ),
    render_markdown: bool = typer.Option(
        True,
        "--render-markdown/--no-render-markdown",
        help="Render agent replies as markdown in the terminal",
    ),
) -> None:
    _print_banner()
    config = load_config()
    gateway_url = _gateway_url_from_config(config)
    gateway_host, gateway_port = _gateway_host_port_from_config(config)

    active_model = get_default_model(config)
    selected_model_a = model_a or active_model
    selected_model_b = model_b or active_model

    try:
        with ensure_gateway(
            gateway_url,
            gateway_host,
            gateway_port,
            autostart=auto_gateway,
            start_timeout=gateway_start_timeout,
        ):
            typer.echo(f"Meeting started via {gateway_url}")
            typer.echo(f"agent_a model: {selected_model_a}")
            typer.echo(f"agent_b model: {selected_model_b}")
            if max_turns > 0:
                typer.echo(f"Max turns: {max_turns}")
            else:
                typer.echo("Press Ctrl+C to stop.")

            typer.secho("agent_a> ", fg="cyan", nl=False)
            if render_markdown:
                RICH_CONSOLE.print(Markdown(starter))
            else:
                typer.secho(starter)

            active_turn_renderer: StreamRenderer | None = None

            def _on_turn_start(speaker: str) -> None:
                nonlocal active_turn_renderer
                color = "cyan" if speaker == "agent_a" else "magenta"
                active_turn_renderer = StreamRenderer(
                    prefix=f"{speaker}> ",
                    color=color,
                    render_markdown=render_markdown,
                )
                active_turn_renderer.start()

            def _on_turn_chunk(chunk: str) -> None:
                if active_turn_renderer is None:
                    return
                active_turn_renderer.feed(chunk)

            def _on_turn_end() -> None:
                if active_turn_renderer is None:
                    typer.echo("")
                    return
                active_turn_renderer.finish()

            run_meeting_graph(
                config=MeetingConfig(
                    gateway_url=gateway_url,
                    session_id=session,
                    model_a=selected_model_a,
                    model_b=selected_model_b,
                    starter=starter,
                    system_a=system_a,
                    system_b=system_b,
                    max_turns=max_turns,
                ),
                on_turn_start=_on_turn_start,
                on_turn_chunk=_on_turn_chunk,
                on_turn_end=_on_turn_end,
            )
    except KeyboardInterrupt:
        typer.echo("\nMeeting stopped.")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        _print_stream_failure_context(config=config, model=f"{selected_model_a} / {selected_model_b}", gateway_url=gateway_url, exc=exc)
        raise typer.Exit(code=1)


@github_app.command("login")
def github_login(
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the verification URL in a browser"),
) -> None:
    config = load_config()
    client_id = config.get("github", {}).get("client_id") or None
    try:
        result = login_with_device_flow(client_id=client_id)
    except Exception as exc:
        typer.echo(f"GitHub login failed: {exc}", err=True)
        raise typer.Exit(code=1)

    verification_url = result["verification_uri"]
    if open_browser:
        if webbrowser.open(verification_url):
            typer.echo("Opened the browser to authorize the device.")
        else:
            typer.echo("Unable to open the browser automatically.")

    typer.echo("Open this URL and enter the code to authorize:")
    typer.echo(f"{verification_url}")
    typer.echo(f"Code: {result['user_code']}")

    device_code = result.get("device_code")
    interval = int(result.get("interval", 5))
    client_id = result.get("client_id")
    if not device_code:
        typer.echo("Device code missing from GitHub response", err=True)
        raise typer.Exit(code=1)
    if not client_id:
        typer.echo("Client id missing from GitHub response", err=True)
        raise typer.Exit(code=1)

    try:
        token = poll_for_token(client_id, device_code, interval)
    except Exception as exc:
        typer.echo(f"GitHub token polling failed: {exc}", err=True)
        raise typer.Exit(code=1)

    config.setdefault("github", {})
    config["github"]["token"] = token
    write_config(config, force=True)
    typer.echo("GitHub token stored in ~/.thistlebot/config.json")


@github_app.command("status")
def github_status() -> None:
    config = load_config()
    token = config.get("github", {}).get("token")
    if token:
        typer.echo("GitHub token present")
    else:
        typer.echo("GitHub token missing")


@github_app.command("repos")
def github_repos(
    limit: int = typer.Option(20, "--limit", min=1, help="Max repos to list"),
) -> None:
    config = load_config()
    token = config.get("github", {}).get("token")
    if not token:
        typer.echo("GitHub token missing. Run 'thistlebot github login' first.", err=True)
        raise typer.Exit(code=1)

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    params = {"per_page": min(limit, 100), "sort": "updated"}

    response = httpx.get("https://api.github.com/user/repos", headers=headers, params=params, timeout=30.0)
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        typer.echo(f"GitHub API request failed: {exc}", err=True)
        raise typer.Exit(code=1)

    repos = response.json()
    if not repos:
        typer.echo("No repos found")
        return

    for repo in repos[:limit]:
        name = repo.get("full_name") or repo.get("name")
        url = repo.get("html_url")
        if name and url:
            typer.echo(f"{name} - {url}")
        elif name:
            typer.echo(name)


@wordpress_app.command("login")
def wordpress_login(
    client_id: str = typer.Option("", "--client-id", help="WordPress.com OAuth app client_id"),
    client_secret: str = typer.Option("", "--client-secret", help="WordPress.com OAuth app client_secret"),
    scope: str = typer.Option("posts", "--scope", help="OAuth scope(s), e.g. 'posts media'"),
    blog: str = typer.Option("", "--blog", help="Optional site domain/id for single-site token"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open authorization URL in browser"),
    callback_timeout: int = typer.Option(240, "--callback-timeout", help="Seconds to wait for browser callback"),
) -> None:
    config = load_config()
    wp_cfg = config.setdefault("wordpress", {})
    if not isinstance(wp_cfg, dict):
        wp_cfg = {}
        config.setdefault("integrations", {})["wordpress"] = wp_cfg

    configured_client_id = client_id.strip() or str(wp_cfg.get("client_id") or "")
    configured_client_secret = client_secret.strip() or str(wp_cfg.get("client_secret") or "")
    if not configured_client_id or not configured_client_secret:
        typer.echo("WordPress login requires --client-id and --client-secret (or stored values).", err=True)
        raise typer.Exit(code=1)

    redirect_uri = str(wp_cfg.get("redirect_uri") or WORDPRESS_DEFAULT_REDIRECT_URI)
    selected_blog = blog.strip() or str(wp_cfg.get("blog") or "")

    typer.echo("Starting WordPress OAuth login flow...")
    try:
        token_data, authorize_url = wordpress_login_flow(
            client_id=configured_client_id,
            client_secret=configured_client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            timeout=30.0,
            callback_timeout=callback_timeout,
            open_browser=open_browser,
            blog=selected_blog or None,
        )
    except Exception as exc:
        typer.echo(f"WordPress login failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if not open_browser:
        typer.echo("Open this URL to authorize:")
        typer.echo(authorize_url)

    wp_cfg["enabled"] = True
    wp_cfg["client_id"] = configured_client_id
    wp_cfg["client_secret"] = configured_client_secret
    wp_cfg["redirect_uri"] = redirect_uri
    wp_cfg["scope"] = token_data.get("scope") or scope
    wp_cfg["blog"] = selected_blog or None
    wp_cfg["blog_id"] = token_data.get("blog_id")
    wp_cfg["blog_url"] = token_data.get("blog_url")
    wp_cfg["token"] = token_data.get("access_token")
    wp_cfg["token_type"] = token_data.get("token_type")
    wp_cfg["expires_in"] = token_data.get("expires_in")
    wp_cfg["expires_at"] = token_data.get("expires_at")

    config.setdefault("integrations", {})["wordpress"] = wp_cfg
    write_config(config, force=True)
    typer.echo("WordPress token stored in ~/.thistlebot/config.json")


@wordpress_app.command("status")
def wordpress_status() -> None:
    config = load_config()
    wp_cfg = _wordpress_config(config)

    enabled = bool(wp_cfg.get("enabled"))
    token = wp_cfg.get("token")
    client_id = wp_cfg.get("client_id")
    client_secret = wp_cfg.get("client_secret")

    typer.echo(f"WordPress enabled: {'yes' if enabled else 'no'}")
    typer.echo(f"WordPress client_id: {'present' if client_id else 'missing'}")
    typer.echo(f"WordPress client_secret: {'present' if client_secret else 'missing'}")
    typer.echo(f"WordPress token: {'present' if token else 'missing'}")

    expires_at = wp_cfg.get("expires_at")
    if token and isinstance(expires_at, (int, float)):
        state = "expired" if wordpress_token_expired(wp_cfg) else "active"
        typer.echo(f"WordPress token state: {state}")


@wordpress_app.command("sites")
def wordpress_sites() -> None:
    config = load_config()
    wp_cfg = _wordpress_config(config)
    try:
        client = _wordpress_client(config)
        try:
            result = client.list_sites()
        except Exception:
            # Some tokens are single-blog scoped and cannot call /me/sites.
            client_id = wp_cfg.get("client_id")
            if not isinstance(client_id, str) or not client_id:
                raise
            info = client.token_info(client_id)
            blog_id = info.get("blog_id")
            blog_ref = str(blog_id) if blog_id is not None else ""
            if not blog_ref:
                raise
            site = client.get_site(blog_ref)
            result = {"sites": [site], "token_scope": info.get("scope"), "single_blog_token": True}
    except Exception as exc:
        typer.echo(f"WordPress sites lookup failed: {exc}", err=True)
        raise typer.Exit(code=1)

    sites = result.get("sites") if isinstance(result, dict) else None
    if not isinstance(sites, list) or not sites:
        typer.echo("No sites returned.")
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    first_site = sites[0] if sites else None
    if isinstance(first_site, dict):
        site_id = first_site.get("ID")
        site_url = first_site.get("URL") or first_site.get("url")
        if site_id is not None:
            wp_cfg["blog_id"] = site_id
        if isinstance(site_url, str) and site_url:
            wp_cfg["blog_url"] = site_url
            if not wp_cfg.get("blog"):
                parsed = urllib.parse.urlparse(site_url)
                if parsed.netloc:
                    wp_cfg["blog"] = parsed.netloc
                else:
                    wp_cfg["blog"] = site_url
        config.setdefault("integrations", {})["wordpress"] = wp_cfg
        write_config(config, force=True)

    for item in sites:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("title") or item.get("URL") or "(unnamed)"
        url = item.get("URL") or item.get("url") or item.get("domain") or ""
        site_id = item.get("ID")
        row = str(name)
        if url:
            row += f" url={url}"
        if site_id is not None:
            row += f" id={site_id}"
        typer.echo(row)


@wordpress_app.command("create-post")
def wordpress_create_post(
    title: str = typer.Option(..., "--title", help="Post title"),
    content: str = typer.Option(..., "--content", help="Post content"),
    site: Optional[str] = typer.Option(None, "--site", help="Target site domain or site ID"),
    status: str = typer.Option("draft", "--status", help="Post status: draft|publish|pending|future|private"),
) -> None:
    config = load_config()
    try:
        site_ref = _wordpress_site_ref(config, site)
        client = _wordpress_client(config)
        result = client.create_post(site_ref, title=title, content=content, status=status)
    except Exception as exc:
        typer.echo(f"WordPress create post failed: {exc}", err=True)
        raise typer.Exit(code=1)

    post_id = result.get("ID") if isinstance(result, dict) else None
    post_url = result.get("URL") if isinstance(result, dict) else None
    typer.echo(f"WordPress post created on '{site_ref}'.")
    if post_id is not None:
        typer.echo(f"Post ID: {post_id}")
    if isinstance(post_url, str) and post_url:
        typer.echo(f"Post URL: {post_url}")


@wordpress_app.command("test")
def wordpress_test(
    site: Optional[str] = typer.Option(None, "--site", help="Target site domain or site ID"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt and create test post"),
) -> None:
    config = load_config()
    try:
        site_ref = _wordpress_site_ref(config, site)
        if not yes and not typer.confirm(f"Create a test draft post on '{site_ref}'?", default=False):
            typer.echo("Cancelled. No post created.")
            return
        client = _wordpress_client(config)
        result = client.create_post(site_ref, title="Test", content="test", status="draft")
    except Exception as exc:
        typer.echo(f"WordPress test failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("WordPress test post created successfully.")
    post_id = result.get("ID") if isinstance(result, dict) else None
    post_url = result.get("URL") if isinstance(result, dict) else None
    if post_id is not None:
        typer.echo(f"Post ID: {post_id}")
    if isinstance(post_url, str) and post_url:
        typer.echo(f"Post URL: {post_url}")


@wordpress_app.command("logout")
def wordpress_logout(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    config = load_config()
    if not yes and not typer.confirm("Clear stored WordPress credentials?", default=True):
        typer.echo("Aborted.")
        return

    wp_cfg = _wordpress_config(config)
    for key in ("token", "token_type", "expires_in", "expires_at", "blog", "blog_id", "blog_url"):
        wp_cfg[key] = None
    config.setdefault("integrations", {})["wordpress"] = wp_cfg
    write_config(config, force=True)
    typer.echo("WordPress credentials cleared.")


@mcp_app.command("status")
def mcp_status() -> None:
    config = load_config()
    registry = build_mcp_registry(config)
    statuses = registry.statuses()

    if not config.get("mcp", {}).get("enabled"):
        typer.echo("MCP is disabled (mcp.enabled=false).")
        return

    if not statuses:
        typer.echo("No MCP servers configured or enabled.")
        return

    for status in statuses:
        name = status.get("name", "unknown")
        connected = "yes" if status.get("connected") else "no"
        transport = status.get("transport", "stdio")
        last_error = status.get("last_error")
        last_tool_count = status.get("last_tool_count")
        typer.echo(f"{name}: connected={connected} transport={transport}")
        if isinstance(last_tool_count, int):
            typer.echo(f"  tool_count={last_tool_count}")
        if last_error:
            typer.echo(f"  last_error={last_error}")


@mcp_app.command("tools")
def mcp_tools() -> None:
    config = load_config()
    mcp_registry = build_mcp_registry(config)
    tool_registry = build_tool_registry(config, mcp_registry)
    names = tool_registry.list_tool_names()
    if not names:
        typer.echo("No tools available.")
        return

    for name in names:
        typer.echo(name)


@mcp_enable_app.callback(invoke_without_command=True)
def mcp_enable(server: str = typer.Argument(..., help="MCP server name, e.g. open-websearch")) -> None:
    config = load_config()
    normalized = server.strip().lower().replace("_", "-")
    if normalized == "open-web-search":
        normalized = "open-websearch"

    if normalized == "open-websearch":
        npx_available, _ = _ensure_open_websearch_server(config, enabled=True)
        write_config(config, force=True)
        typer.echo("Enabled mcp.enabled=true and mcp.servers.open-websearch.enabled=true")
        if npx_available:
            typer.echo("npx detected. Run 'thistlebot mcp status' to verify connectivity.")
        else:
            typer.echo("npx not found. Install Node.js and run 'thistlebot mcp enable open-websearch' again.", err=True)
        return

    mcp_cfg = config.get("mcp")
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
        config["mcp"] = mcp_cfg
    mcp_cfg["enabled"] = True
    servers_cfg = mcp_cfg.get("servers")
    if not isinstance(servers_cfg, dict):
        servers_cfg = {}
        mcp_cfg["servers"] = servers_cfg
    server_cfg = servers_cfg.get(normalized)
    if not isinstance(server_cfg, dict):
        server_cfg = {}
        servers_cfg[normalized] = server_cfg
    server_cfg["enabled"] = True
    write_config(config, force=True)
    typer.echo(f"Enabled mcp.servers.{normalized}.enabled=true")


@mcp_disable_app.callback(invoke_without_command=True)
def mcp_disable(server: str = typer.Argument(..., help="MCP server name, e.g. open-websearch")) -> None:
    config = load_config()
    normalized = server.strip().lower().replace("_", "-")
    if normalized == "open-web-search":
        normalized = "open-websearch"

    mcp_cfg = config.get("mcp")
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
        config["mcp"] = mcp_cfg
    servers_cfg = mcp_cfg.get("servers")
    if not isinstance(servers_cfg, dict):
        servers_cfg = {}
        mcp_cfg["servers"] = servers_cfg
    server_cfg = servers_cfg.get(normalized)
    if not isinstance(server_cfg, dict):
        server_cfg = {}
        servers_cfg[normalized] = server_cfg
    server_cfg["enabled"] = False
    write_config(config, force=True)
    typer.echo(f"Disabled mcp.servers.{normalized}.enabled=false")


@app.command()
def mcp_connect() -> None:
    typer.echo("Deprecated. Use 'thistlebot mcp status' and 'thistlebot mcp tools'.")


_AGENT_TEMPLATES: dict[str, dict[str, str]] = {
    "ai": {
        "source_dir": "ai-blogger",
        "default_topic": "Latest AI news",
    },
    "politics": {
        "source_dir": "politics-blogger",
        "default_topic": "Latest politics news",
    },
    "finance": {
        "source_dir": "finance-blogger",
        "default_topic": "Personal finance and making money",
    },
}


def _on_agent_step(step_name: str, step_status: str) -> None:
    if step_status == "started":
        RICH_CONSOLE.print(f"[cyan]>> Step: {step_name}...[/cyan]")
    elif step_status == "completed":
        RICH_CONSOLE.print(f"[green]   {step_name} completed.[/green]")


def _run_agent_workflow_command(
    *,
    name: str,
    workflow: str | None = None,
    topic: str | None = None,
    status: str | None = None,
) -> None:
    from .agents.workflow import run_agent_workflow

    config_overrides: dict[str, Any] = {}
    if topic is not None:
        config_overrides["topic"] = topic
    if status is not None:
        config_overrides["post_status"] = status
        config_overrides["publish_mode"] = "publish" if status == "publish" else "draft"

    try:
        result = run_agent_workflow(
            name,
            workflow_name=workflow,
            config_overrides=config_overrides or None,
            on_step=_on_agent_step,
        )
    except Exception as exc:
        RICH_CONSOLE.print(f"[red]Agent run failed: {exc}[/red]")
        raise typer.Exit(code=1)

    RICH_CONSOLE.print("[bold green]Workflow completed[/bold green]")
    RICH_CONSOLE.print(f"  Agent:    {result.get('agent')}")
    RICH_CONSOLE.print(f"  Workflow: {result.get('workflow')}")
    RICH_CONSOLE.print(f"  Run dir:  {result.get('run_dir')}")
    RICH_CONSOLE.print(f"  Status:   {result.get('status')}")


def _ensure_mcp_for_blogger(config: dict[str, Any]) -> None:
    mcp_cfg = config.get("mcp")
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
        config["mcp"] = mcp_cfg
    mcp_cfg["enabled"] = True

    servers_cfg = mcp_cfg.get("servers")
    if not isinstance(servers_cfg, dict):
        servers_cfg = {}
        mcp_cfg["servers"] = servers_cfg

    open_websearch_cfg = servers_cfg.get("open-websearch")
    if not isinstance(open_websearch_cfg, dict):
        open_websearch_cfg = {}
        servers_cfg["open-websearch"] = open_websearch_cfg
    open_websearch_cfg["enabled"] = True
    open_websearch_cfg.setdefault("transport", "stdio")
    open_websearch_cfg.setdefault("command", "npx")
    open_websearch_cfg.setdefault("args", ["-y", "open-websearch@latest"])
    open_websearch_cfg.setdefault("timeout_seconds", 30)
    env_cfg = open_websearch_cfg.get("env")
    if not isinstance(env_cfg, dict):
        env_cfg = {}
        open_websearch_cfg["env"] = env_cfg
    env_cfg.setdefault("MODE", "stdio")
    env_cfg.setdefault("DEFAULT_SEARCH_ENGINE", "duckduckgo")

    browser_cfg = servers_cfg.get("browser")
    if not isinstance(browser_cfg, dict):
        browser_cfg = {}
        servers_cfg["browser"] = browser_cfg
    browser_cfg["enabled"] = True
    browser_cfg.setdefault("transport", "stdio")
    browser_cfg.setdefault("command", "npx")
    browser_cfg.setdefault("args", ["-y", "@playwright/mcp@latest"])
    browser_cfg.setdefault("timeout_seconds", 30)


def _prompt_schedule_preset() -> tuple[dict[str, Any], str]:
    choices = [
        "1/day",
        "2/day",
        "3/day",
        "4/day",
        "5/day",
        "6/day",
        "every hour",
        "every 30 minutes",
        "custom cron",
    ]
    selected = questionary.select("Select posting schedule", choices=choices, default="4/day").ask()
    choice = str(selected or "4/day")

    if choice.endswith("/day"):
        times_per_day = int(choice.split("/")[0])
        preset_crons = {
            1: "0 0 * * *",
            2: "0 0,12 * * *",
            3: "0 0,8,16 * * *",
            4: "0 0,6,12,18 * * *",
            5: "0 0,5,10,15,20 * * *",
            6: "0 0,4,8,12,16,20 * * *",
        }
        cron = preset_crons.get(times_per_day, "0 0,6,12,18 * * *")
        return ({"enabled": True, "cron": cron, "timezone": "UTC"}, cron)
    if choice == "every hour":
        return ({"enabled": True, "interval_minutes": 60, "timezone": "UTC"}, "every 60 minutes")
    if choice == "every 30 minutes":
        return ({"enabled": True, "interval_minutes": 30, "timezone": "UTC"}, "every 30 minutes")

    cron = questionary.text("Enter cron expression", default="0 0,6,12,18 * * *").ask()
    cron_text = str(cron or "0 0,6,12,18 * * *").strip() or "0 0,6,12,18 * * *"
    return ({"enabled": True, "cron": cron_text, "timezone": "UTC"}, cron_text)


def _ensure_wordpress_login(config: dict[str, Any]) -> dict[str, Any]:
    wp_cfg = _wordpress_config(config)

    client_id = str(wp_cfg.get("client_id") or "").strip()
    client_secret = str(wp_cfg.get("client_secret") or "").strip()
    token = str(wp_cfg.get("token") or "").strip()

    if token and client_id and client_secret:
        return wp_cfg

    typer.echo("WordPress credentials are required.")
    typer.echo("Create an app at https://developer.wordpress.com/apps/")
    typer.echo("Use redirect URI: http://127.0.0.1:8766/callback")

    if not client_id:
        client_id = str(questionary.text("WordPress client_id").ask() or "").strip()
    if not client_secret:
        client_secret = str(questionary.password("WordPress client_secret").ask() or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("WordPress client_id and client_secret are required.")

    redirect_uri = str(wp_cfg.get("redirect_uri") or WORDPRESS_DEFAULT_REDIRECT_URI)
    scope = str(wp_cfg.get("scope") or "posts sites media")
    token_data, _ = wordpress_login_flow(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        timeout=30.0,
        callback_timeout=240,
        open_browser=True,
        blog=None,
    )
    wp_cfg["enabled"] = True
    wp_cfg["client_id"] = client_id
    wp_cfg["client_secret"] = client_secret
    wp_cfg["redirect_uri"] = redirect_uri
    wp_cfg["scope"] = token_data.get("scope") or scope
    wp_cfg["blog"] = token_data.get("blog") or wp_cfg.get("blog")
    wp_cfg["blog_id"] = token_data.get("blog_id")
    wp_cfg["blog_url"] = token_data.get("blog_url")
    wp_cfg["token"] = token_data.get("access_token")
    wp_cfg["token_type"] = token_data.get("token_type")
    wp_cfg["expires_in"] = token_data.get("expires_in")
    wp_cfg["expires_at"] = token_data.get("expires_at")
    config.setdefault("integrations", {})["wordpress"] = wp_cfg
    return wp_cfg


def _pick_wordpress_site(config: dict[str, Any]) -> str:
    wp_cfg = _wordpress_config(config)
    current_blog = str(wp_cfg.get("blog") or "").strip()
    if current_blog:
        return current_blog

    blog_url = str(wp_cfg.get("blog_url") or "").strip()
    if blog_url:
        parsed = urllib.parse.urlparse(blog_url)
        domain = parsed.netloc or blog_url
        if domain:
            wp_cfg["blog"] = domain
            config.setdefault("integrations", {})["wordpress"] = wp_cfg
            return domain

    blog_id = wp_cfg.get("blog_id")
    if isinstance(blog_id, (int, float)):
        wp_cfg["blog"] = str(int(blog_id))
        config.setdefault("integrations", {})["wordpress"] = wp_cfg
        return str(int(blog_id))
    if isinstance(blog_id, str) and blog_id.strip():
        wp_cfg["blog"] = blog_id.strip()
        config.setdefault("integrations", {})["wordpress"] = wp_cfg
        return blog_id.strip()

    client = _wordpress_client(config)
    sites_result: dict[str, Any] | None = None
    try:
        raw = client.list_sites()
        if isinstance(raw, dict):
            sites_result = raw
    except Exception:
        client_id = wp_cfg.get("client_id")
        if isinstance(client_id, str) and client_id.strip():
            try:
                info = client.token_info(client_id.strip())
                blog_id = info.get("blog_id") if isinstance(info, dict) else None
                if blog_id is not None:
                    site = client.get_site(str(blog_id))
                    sites_result = {"sites": [site]}
            except Exception:
                sites_result = None
        if sites_result is None:
            raise RuntimeError("No WordPress sites found for the authenticated token.")

    sites = sites_result.get("sites") if isinstance(sites_result, dict) else None
    if not isinstance(sites, list) or not sites:
        raise RuntimeError("No WordPress sites found for the authenticated token.")

    choices: list[str] = []
    by_domain: dict[str, dict[str, Any]] = {}
    for item in sites:
        if not isinstance(item, dict):
            continue
        site_url = item.get("URL") or item.get("url") or ""
        domain = ""
        if isinstance(site_url, str) and site_url:
            parsed = urllib.parse.urlparse(site_url)
            domain = parsed.netloc or site_url
        if not domain:
            site_id = item.get("ID")
            if site_id is not None:
                domain = str(site_id)
        if domain:
            choices.append(domain)
            by_domain[domain] = item
    if not choices:
        raise RuntimeError("No usable WordPress site domains found.")

    if sys.stdin.isatty() and sys.stdout.isatty():
        selected = questionary.select("Select WordPress site", choices=choices, default=choices[0]).ask()
        selected_domain = str(selected or choices[0])
    else:
        selected_domain = choices[0]

    selected_item = by_domain.get(selected_domain, {})
    wp_cfg["blog"] = selected_domain
    site_url = selected_item.get("URL") or selected_item.get("url") if isinstance(selected_item, dict) else None
    if isinstance(site_url, str) and site_url.strip():
        wp_cfg["blog_url"] = site_url.strip()
    site_id = selected_item.get("ID") if isinstance(selected_item, dict) else None
    if site_id is not None:
        wp_cfg["blog_id"] = site_id
    config.setdefault("integrations", {})["wordpress"] = wp_cfg
    return selected_domain


def _default_agent_name() -> str:
    from .agents.registry import list_agent_names

    existing = set(list_agent_names())
    index = 1
    while True:
        candidate = f"blogger{index}"
        if candidate not in existing:
            return candidate
        index += 1


def _template_source_path(template: str, *, pack: str = "blogging") -> Path:
    template_cfg = _AGENT_TEMPLATES.get(template)
    if not isinstance(template_cfg, dict):
        raise RuntimeError(f"Unknown template '{template}'.")
    source_dir = template_cfg.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir:
        raise RuntimeError(f"Template '{template}' has no source directory configured.")
    package_root = Path(__file__).resolve().parent
    pack_source = package_root / "packs" / pack / "agents" / source_dir
    if not pack_source.exists() or not pack_source.is_dir():
        raise RuntimeError(f"Template source not found: {pack_source}")
    return pack_source


def _create_agent_from_template(agent_name: str, template: str, *, pack: str = "blogging") -> str:
    source = _template_source_path(template, pack=pack)
    target = Path(__file__).resolve().parent / "agents" / agent_name
    if target.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)
        create_state = "resumed"
    else:
        shutil.copytree(source, target)
        create_state = "created"

    _apply_template_defaults(target, agent_name=agent_name, template=template)
    return create_state


def _apply_template_defaults(target: Path, *, agent_name: str, template: str) -> None:
    template_cfg = _AGENT_TEMPLATES.get(template, _AGENT_TEMPLATES["ai"])
    default_topic = template_cfg.get("default_topic", _AGENT_TEMPLATES["ai"]["default_topic"])

    agent_md_path = target / "AGENT.md"
    if not agent_md_path.exists():
        return

    text = agent_md_path.read_text(encoding="utf-8")
    frontmatter: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            import yaml

            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip("\n")

    frontmatter["name"] = agent_name
    ext = frontmatter.get("x-thistlebot")
    if not isinstance(ext, dict):
        ext = {}
        frontmatter["x-thistlebot"] = ext
    config_ext = ext.get("config")
    if not isinstance(config_ext, dict):
        config_ext = {}
        ext["config"] = config_ext
    defaults_ext = config_ext.get("defaults")
    if not isinstance(defaults_ext, dict):
        defaults_ext = {}
        config_ext["defaults"] = defaults_ext

    defaults_ext["topic"] = default_topic
    defaults_ext["topic_template"] = template
    defaults_ext["post_status"] = "draft"
    defaults_ext["publish_mode"] = "draft"
    defaults_ext["enforce_draft_mode"] = True

    import yaml

    rendered_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    agent_md_path.write_text(f"---\n{rendered_frontmatter}---\n\n{body.lstrip()}", encoding="utf-8")


def _workflow_alias(agent_name: str, value: str | None) -> str:
    from .agents.loader import load_agent_definition

    alias_or_name = str(value or "post").strip()
    agent_def = load_agent_definition(agent_name)
    aliases = agent_def.manifest.get("workflow_aliases")
    if isinstance(aliases, dict):
        mapped = aliases.get(alias_or_name)
        if isinstance(mapped, str) and mapped.strip():
            return mapped
    if alias_or_name == "post":
        return agent_def.default_workflow_name()
    return alias_or_name


def _resolve_agent_schedule(name: str) -> dict[str, Any]:
    from .agents.config import load_agent_config
    from .agents.loader import load_agent_definition

    agent_def = load_agent_definition(name)
    cfg = load_agent_config(name, agent_def)
    runtime_schedule = cfg.get("schedule") if isinstance(cfg.get("schedule"), dict) else {}
    merged = dict(agent_def.schedule())
    merged.update(runtime_schedule)
    return merged


def _agent_start_impl(name: str, *, foreground: bool) -> None:
    from .agents.runner import AgentDaemon, is_agent_daemon_running
    from .agents.workflow import run_agent_workflow
    from .storage.paths import agent_log_path

    schedule_cfg = _resolve_agent_schedule(name)
    if not bool(schedule_cfg.get("enabled", False)):
        typer.echo("Schedule is disabled. Run setup first to configure schedule.", err=True)
        raise typer.Exit(code=1)

    if is_agent_daemon_running(name):
        typer.echo(f"{name} daemon is already running.")
        raise typer.Exit(code=0)

    if foreground:
        daemon = AgentDaemon(agent_name=name, schedule_config=schedule_cfg, run_once=lambda: run_agent_workflow(name))
        typer.echo(f"Starting {name} daemon in foreground. Press Ctrl+C to stop.")
        daemon.run_forever()
        return

    log_path = agent_log_path(name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_fh:
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "thistlebot", "agent", name, "daemon-run"],
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    typer.echo(f"{name} daemon start requested.")
    typer.echo(f"Log file: {log_path}")


def _agent_setup_impl(
    *,
    name: str,
    site: str | None,
    topic: str | None,
    template: str | None,
    post_status: str,
    yes: bool,
) -> None:
    from .agents.config import load_agent_config, runtime_agent_config_path, save_agent_runtime_config
    from .agents.loader import load_agent_definition

    config = load_config()
    _ensure_mcp_for_blogger(config)
    _ensure_wordpress_login(config)

    agent_def = load_agent_definition(name)
    try:
        current = load_agent_config(name, agent_def)
    except RuntimeError:
        current = dict(agent_def.defaults())
        runtime_cfg_path = runtime_agent_config_path(name)
        if runtime_cfg_path.exists():
            try:
                loaded = json.loads(runtime_cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current.update(loaded)
            except Exception:
                pass

    chosen_template = (template or "").strip().lower()
    chosen_topic = (topic or "").strip()
    if not chosen_topic:
        if chosen_template not in _AGENT_TEMPLATES and not yes:
            selected = questionary.select(
                "Topic template",
                choices=["ai", "politics", "finance", "custom"],
                default="ai",
            ).ask()
            chosen_template = str(selected or "ai")
        if chosen_template == "custom":
            chosen_topic = str(questionary.text("Enter topic", default=str(current.get("topic") or "")).ask() or "").strip()
        else:
            template_cfg = _AGENT_TEMPLATES.get(chosen_template or "ai", _AGENT_TEMPLATES["ai"])
            chosen_topic = template_cfg.get("default_topic", _AGENT_TEMPLATES["ai"]["default_topic"])

    resolved_site = (site or "").strip() or _pick_wordpress_site(config)
    schedule_cfg, schedule_text = ({"enabled": True, "cron": "0 0,6,12,18 * * *", "timezone": "UTC"}, "0 0,6,12,18 * * *")
    if not yes:
        schedule_cfg, schedule_text = _prompt_schedule_preset()

    current["site"] = resolved_site
    current["topic"] = chosen_topic
    current["post_status"] = "publish" if post_status == "publish" else "draft"
    current["publish_mode"] = "publish" if post_status == "publish" else "draft"
    current["enforce_draft_mode"] = post_status != "publish"
    current["topic_template"] = chosen_template or "ai"
    current["schedule"] = schedule_cfg

    write_config(config, force=True)
    path = save_agent_runtime_config(name, current)
    typer.echo(f"Setup complete for agent '{name}'.")
    typer.echo(f"Schedule: {schedule_text} ({schedule_cfg.get('timezone', 'UTC')})")
    typer.echo(f"Config: {path}")
    typer.echo(f"Modify config: thistlebot agent {name} config set key=value")
    typer.echo(f"One run: thistlebot agent {name} workflow post")
    typer.echo(f"Scheduled daemon: thistlebot agent {name}")


@agent_app.command("list")
def agent_list() -> None:
    from .agents.registry import discover_agents

    agents = discover_agents()
    if not agents:
        typer.echo("No agents found.")
        return
    for item in agents:
        typer.echo(f"{item.name}\t{item.description()}")


@agent_app.command("create")
def agent_create(
    pack: str = typer.Option("blogging", "--pack", help="Pack name (default: blogging)"),
    template: str = typer.Option("ai", "--template", help="Template: ai|politics|finance"),
    name: Optional[str] = typer.Option(None, "--name", help="Agent name (default: bloggerN)"),
) -> None:
    chosen_template = template.strip().lower()
    if chosen_template not in _AGENT_TEMPLATES:
        typer.echo("Invalid --template. Use: ai, politics, finance.", err=True)
        raise typer.Exit(code=1)

    agent_name = (name or "").strip() or _default_agent_name()
    chosen_pack = (pack or "").strip().lower() or "blogging"
    create_state = _create_agent_from_template(agent_name, chosen_template, pack=chosen_pack)

    _agent_setup_impl(
        name=agent_name,
        site=None,
        topic=None,
        template=chosen_template,
        post_status="draft",
        yes=True,
    )
    if create_state == "resumed":
        typer.echo(f"Resumed setup for existing agent '{agent_name}' using template '{chosen_template}'.")
    else:
        typer.echo(f"Created agent '{agent_name}' from template '{chosen_template}'.")


def _build_agent_subapp(agent_name: str) -> typer.Typer:
    subapp = typer.Typer(help=f"{agent_name} agent commands")

    @subapp.callback(invoke_without_command=True)
    def agent_root(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            _agent_start_impl(agent_name, foreground=True)

    @subapp.command("workflow")
    def agent_workflow(
        name_or_alias: Optional[str] = typer.Argument(None, help="Workflow name or alias (e.g. post)"),
        topic: Optional[str] = typer.Option(None, "--topic", help="Topic override for this run"),
        status: Optional[str] = typer.Option(None, "--status", help="WordPress status override (draft/publish)"),
    ) -> None:
        resolved = _workflow_alias(agent_name, name_or_alias)
        _run_agent_workflow_command(name=agent_name, workflow=resolved, topic=topic, status=status)

    @subapp.command("setup")
    def agent_setup(
        site: Optional[str] = typer.Option(None, "--site", help="WordPress site domain"),
        topic: Optional[str] = typer.Option(None, "--topic", help="Custom topic override"),
        template: Optional[str] = typer.Option(None, "--template", help="Template: ai|politics|finance|custom"),
        post_status: str = typer.Option("draft", "--post-status", help="Default post status: draft|publish"),
        yes: bool = typer.Option(False, "--yes", help="Use defaults and skip prompts"),
    ) -> None:
        _agent_setup_impl(
            name=agent_name,
            site=site,
            topic=topic,
            template=template,
            post_status=post_status,
            yes=yes,
        )

    config_subapp = typer.Typer(help="Agent config operations")
    subapp.add_typer(config_subapp, name="config")

    @config_subapp.command("show")
    def agent_config_show() -> None:
        from .agents.config import load_agent_config
        from .agents.loader import load_agent_definition

        agent_def = load_agent_definition(agent_name)
        cfg = load_agent_config(agent_name, agent_def)
        RICH_CONSOLE.print_json(json.dumps(cfg, indent=2, default=str))

    @config_subapp.command("set")
    def agent_config_set(
        kv: list[str] = typer.Argument(..., help="One or more key=value pairs"),
    ) -> None:
        from .agents.config import load_agent_config, save_agent_runtime_config
        from .agents.loader import load_agent_definition

        agent_def = load_agent_definition(agent_name)
        cfg = load_agent_config(agent_name, agent_def)

        for item in kv:
            if "=" not in item:
                typer.echo(f"Invalid pair '{item}'. Expected key=value.", err=True)
                raise typer.Exit(code=1)
            key, raw_value = item.split("=", 1)
            key_path = [part for part in key.strip().split(".") if part]
            if not key_path:
                typer.echo(f"Invalid key in '{item}'.", err=True)
                raise typer.Exit(code=1)
            value: Any = raw_value
            lowered = raw_value.strip().lower()
            if lowered in {"true", "false"}:
                value = lowered == "true"
            elif lowered in {"null", "none"}:
                value = None
            else:
                try:
                    value = json.loads(raw_value)
                except Exception:
                    value = raw_value

            cursor: dict[str, Any] = cfg
            for part in key_path[:-1]:
                node = cursor.get(part)
                if not isinstance(node, dict):
                    node = {}
                    cursor[part] = node
                cursor = node
            cursor[key_path[-1]] = value

        path = save_agent_runtime_config(agent_name, cfg)
        typer.echo(f"Updated {path}")

    @subapp.command("status")
    def agent_status(limit: int = typer.Option(5, "--limit", "-n", help="Number of recent runs to show")) -> None:
        from .agents.config import list_runs, load_agent_config
        from .agents.loader import load_agent_definition
        from .agents.memory import JsonFileMemoryStore
        from .agents.runner import is_agent_daemon_running, read_agent_state

        agent_def = load_agent_definition(agent_name)
        cfg = load_agent_config(agent_name, agent_def)
        daemon_state = read_agent_state(agent_name)
        daemon_running = is_agent_daemon_running(agent_name)
        memory_store = JsonFileMemoryStore(agent_name)

        RICH_CONSOLE.print(f"[bold]{agent_name} agent[/bold]")
        RICH_CONSOLE.print(f"  Site:  {cfg.get('site', 'not configured')}")
        RICH_CONSOLE.print(f"  Topic: {cfg.get('topic', 'not configured')}")
        RICH_CONSOLE.print(f"  Daemon running: {daemon_running}")
        if daemon_state:
            RICH_CONSOLE.print(f"  Last run at:   {daemon_state.get('last_run_at') or 'n/a'}")
            RICH_CONSOLE.print(f"  Next run at:   {daemon_state.get('next_run_at') or 'n/a'}")
            RICH_CONSOLE.print(f"  Last status:   {daemon_state.get('last_run_status') or 'n/a'}")
            if daemon_state.get("last_error"):
                RICH_CONSOLE.print(f"  Last error:    {daemon_state.get('last_error')}")

        runs = list_runs(agent_name)
        if not runs:
            RICH_CONSOLE.print("No runs found.")
            return

        RICH_CONSOLE.print(f"[bold]Recent runs (latest {limit}):[/bold]")
        for run_dir in runs[:limit]:
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                RICH_CONSOLE.print(
                    f"  {run_dir.name} status={meta.get('status','unknown')} steps={meta.get('steps',{})} ts={meta.get('timestamp','')}"
                )

        recent_memories = memory_store.list_recent(limit=limit)
        if recent_memories:
            RICH_CONSOLE.print(f"[bold]Recent memory summaries (latest {limit}):[/bold]")
            for entry in recent_memories:
                RICH_CONSOLE.print(f"  {entry.timestamp} {entry.type} run={entry.run_id or '-'} {entry.summary[:120]}")

    @subapp.command("start")
    def agent_start() -> None:
        _agent_start_impl(agent_name, foreground=False)

    @subapp.command("stop")
    def agent_stop() -> None:
        from .agents.runner import stop_agent_daemon

        ok = stop_agent_daemon(agent_name)
        if not ok:
            typer.echo(f"No running daemon found for {agent_name}.")
            raise typer.Exit(code=1)
        typer.echo(f"Stop signal sent to {agent_name} daemon.")

    @subapp.command("daemon-run", hidden=True)
    def agent_daemon_run() -> None:
        from .agents.runner import AgentDaemon
        from .agents.workflow import run_agent_workflow

        schedule_cfg = _resolve_agent_schedule(agent_name)
        daemon = AgentDaemon(agent_name=agent_name, schedule_config=schedule_cfg, run_once=lambda: run_agent_workflow(agent_name))
        daemon.run_forever()

    @subapp.command("action")
    def agent_action(
        action: str = typer.Argument(..., help="Action name from AGENT.md actions"),
        arg: list[str] = typer.Option([], "--arg", help="Action args in key=value format. Repeat flag for multiple."),
    ) -> None:
        from .agents.loader import load_agent_definition
        from .llm.factory import build_llm_client, get_default_model

        agent_def = load_agent_definition(agent_name)
        actions = agent_def.actions()
        action_def = actions.get(action) if isinstance(actions, dict) else None
        if not isinstance(action_def, dict):
            typer.echo(f"Unknown action '{action}' for agent '{agent_name}'.", err=True)
            raise typer.Exit(code=1)

        handler_ref = action_def.get("handler")
        if not isinstance(handler_ref, str) or ":" not in handler_ref:
            typer.echo("Invalid action handler in AGENT.md", err=True)
            raise typer.Exit(code=1)
        module_name, func_name = handler_ref.split(":", 1)
        module_path = agent_def.root / f"{module_name.replace('.', '/')}.py"
        if module_path.exists():
            spec_name = f"thistlebot.agents.{agent_name}.{module_name}"
            spec = importlib.util.spec_from_file_location(spec_name, module_path)
            if spec is None or spec.loader is None:
                typer.echo("Unable to load action module", err=True)
                raise typer.Exit(code=1)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(f"thistlebot.agents.{agent_name}.{module_name}")
        handler = getattr(module, func_name, None)
        if not callable(handler):
            typer.echo("Action handler is not callable", err=True)
            raise typer.Exit(code=1)

        args_dict: dict[str, Any] = {}
        for item in arg:
            if "=" not in item:
                typer.echo(f"Invalid --arg '{item}', expected key=value", err=True)
                raise typer.Exit(code=1)
            key, value = item.split("=", 1)
            args_dict[key.strip()] = value.strip()

        cfg = load_config()
        client = build_llm_client(cfg)
        model = get_default_model(cfg)
        mcp_registry = build_mcp_registry(cfg) if cfg.get("mcp", {}).get("enabled") else None
        registry = build_tool_registry(cfg, mcp_registry, tool_spec=agent_def.tools())

        result = handler(agent_name=agent_name, client=client, registry=registry, model=model, args=args_dict)
        RICH_CONSOLE.print_json(json.dumps(result, indent=2, default=str))

    return subapp


def _register_agent_subapps() -> None:
    from .agents.registry import discover_agents

    for agent in discover_agents():
        agent_app.add_typer(_build_agent_subapp(agent.name), name=agent.name)


_register_agent_subapps()


# ---------------------------------------------------------------------------
# skill commands
# ---------------------------------------------------------------------------

@skill_app.command("list")
def skill_list(
    agent: Optional[str] = typer.Option(None, "--agent", help="List skills for a specific agent"),
) -> None:
    """List available skills."""
    from pathlib import Path as _Path
    from .agents.skill_loader import list_skills

    if agent:
        from .agents.loader import load_agent_definition
        agent_def = load_agent_definition(agent)
        search_paths = agent_def._skill_search_paths()
    else:
        # Search all agent skill directories
        agents_root = _Path(__file__).resolve().parent / "agents"
        packs_root = _Path(__file__).resolve().parent / "packs"
        search_paths = [
            child / "skills"
            for child in agents_root.iterdir()
            if child.is_dir() and not child.name.startswith("__")
        ]
        if packs_root.exists():
            for pack_dir in packs_root.iterdir():
                if not pack_dir.is_dir() or pack_dir.name.startswith("__"):
                    continue
                search_paths.append(pack_dir / "skills")

    skills = list_skills(search_paths)
    if not skills:
        typer.echo("No skills found.")
        return
    for skill in skills:
        tools_str = ", ".join(skill.allowed_tools) if skill.allowed_tools else "(inherits all)"
        typer.echo(f"{skill.name}\t{skill.description or '(no description)'}\t[tools: {tools_str}]")


@skill_app.command("run")
def skill_run(
    skill_name: str = typer.Argument(..., help="Skill name to run"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent to load skill from"),
    arguments: str = typer.Option("", "--args", help="Arguments passed to $ARGUMENTS in skill instructions"),
) -> None:
    """Run a skill standalone."""
    from pathlib import Path as _Path
    from .agents.skill_loader import load_skill
    from .agents.skill_runner import run_skill_standalone

    config = load_config()
    model = get_default_model(config)
    client = get_llm_provider(config)
    mcp_registry = build_mcp_registry(config)
    registry = build_tool_registry(config, mcp_registry)

    if agent:
        from .agents.loader import load_agent_definition
        agent_def = load_agent_definition(agent)
        search_paths = agent_def._skill_search_paths()
    else:
        agents_root = _Path(__file__).resolve().parent / "agents"
        packs_root = _Path(__file__).resolve().parent / "packs"
        search_paths = [
            child / "skills"
            for child in agents_root.iterdir()
            if child.is_dir() and not child.name.startswith("__")
        ]
        if packs_root.exists():
            for pack_dir in packs_root.iterdir():
                if not pack_dir.is_dir() or pack_dir.name.startswith("__"):
                    continue
                search_paths.append(pack_dir / "skills")

    skill = load_skill(skill_name, search_paths)
    typer.echo(f"Running skill '{skill.name}'...")
    result = run_skill_standalone(
        skill,
        client=client,
        registry=registry,
        model=model,
        arguments=arguments,
    )
    typer.echo(result)


# ---------------------------------------------------------------------------
# agent migrate command
# ---------------------------------------------------------------------------

@agent_app.command("migrate")
def agent_migrate(
    name: str = typer.Argument(..., help="Agent name to migrate"),
) -> None:
    """Migrate a legacy agent into strict AGENT.md + skills format."""
    from pathlib import Path as _Path
    import yaml as _yaml
    import shutil as _shutil

    root = _Path(__file__).resolve().parent / "agents" / name
    if not root.exists() or not root.is_dir():
        typer.echo(f"Agent directory not found: {root}", err=True)
        raise typer.Exit(code=1)

    agent_md = root / "AGENT.md"
    agent_json = root / "agent.json"

    if agent_md.exists() and not agent_json.exists():
        typer.echo(f"'{name}' already uses AGENT.md + skills format.")
        return

    if not agent_json.exists():
        typer.echo(f"No legacy agent.json found for '{name}'. Nothing to migrate.")
        return

    manifest = json.loads(agent_json.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        typer.echo(f"Invalid legacy manifest: {agent_json}", err=True)
        raise typer.Exit(code=1)

    raw_tools = manifest.get("tools") if isinstance(manifest.get("tools"), dict) else {}
    allow_tools: list[str] = []
    for key in ("native", "mcp"):
        value = raw_tools.get(key)
        if isinstance(value, list):
            allow_tools.extend(str(item) for item in value if str(item).strip())

    workflows = manifest.get("workflows") if isinstance(manifest.get("workflows"), dict) else {}
    default_workflow = workflows.get("default") if isinstance(workflows.get("default"), str) else None
    if not default_workflow:
        default_workflow = "daily_publish"

    workflow_section: dict[str, Any] = {"default": default_workflow}
    aliases = manifest.get("workflow_aliases")
    if isinstance(aliases, dict) and aliases:
        workflow_section["aliases"] = aliases
    overrides = manifest.get("workflow_overrides")
    if isinstance(overrides, dict) and overrides:
        workflow_section["overrides"] = overrides

    ext: dict[str, Any] = {
        "config": manifest.get("config", {"defaults": {}, "required": []}),
        "schedule": manifest.get("schedule", {"enabled": False}),
        "workflow": workflow_section,
    }
    if isinstance(manifest.get("hooks"), dict):
        ext["hooks"] = manifest["hooks"]
    if isinstance(manifest.get("actions"), dict):
        ext["actions"] = manifest["actions"]

    frontmatter: dict[str, Any] = {
        "name": manifest.get("name", name),
        "version": manifest.get("version", "0.3.0"),
        "description": manifest.get("description", ""),
        "tools": list(dict.fromkeys(allow_tools)),
        "disallowedTools": [],
        "model": manifest.get("model"),
        "x-thistlebot": ext,
    }

    fm = _yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    body = str(manifest.get("description") or f"You are {name}.")
    agent_md.write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")

    prompts = manifest.get("prompts") if isinstance(manifest.get("prompts"), dict) else {}
    skills_dir = root / "skills"
    skills_dir.mkdir(exist_ok=True)
    for skill_name, prompt_rel in prompts.items():
        prompt_path = root / str(prompt_rel)
        if not prompt_path.exists():
            continue
        text = prompt_path.read_text(encoding="utf-8")
        skill_path = skills_dir / str(skill_name) / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_fm = _yaml.safe_dump(
            {"name": str(skill_name), "description": f"{skill_name} skill", "allowed-tools": []},
            sort_keys=False,
            allow_unicode=True,
        )
        skill_path.write_text(f"---\n{skill_fm}---\n{text}", encoding="utf-8")

    workflows_dir = root / "workflows"
    if workflows_dir.exists():
        for wf in workflows_dir.glob("*.json"):
            workflow_data = json.loads(wf.read_text(encoding="utf-8"))
            if isinstance(workflow_data, dict) and isinstance(workflow_data.get("steps"), list):
                for step in workflow_data["steps"]:
                    if not isinstance(step, dict):
                        continue
                    prompt_name = step.pop("prompt", None)
                    if isinstance(prompt_name, str) and prompt_name.strip() and not step.get("skill"):
                        step["skill"] = prompt_name
                wf.write_text(json.dumps(workflow_data, indent=2) + "\n", encoding="utf-8")

    prompts_dir = root / "prompts"
    if prompts_dir.exists():
        _shutil.rmtree(prompts_dir)
    agent_json.unlink()
    typer.echo(f"Migration complete for '{name}'.")


if __name__ == "__main__":
    app()
