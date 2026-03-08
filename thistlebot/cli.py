from __future__ import annotations

import sys
import time
import webbrowser
from typing import Any, Optional
import json

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
from .integrations.wordpress.oauth import (
    DEFAULT_REDIRECT_URI as WORDPRESS_DEFAULT_REDIRECT_URI,
    login_with_authorization_code_flow,
    refresh_access_token as wordpress_refresh_access_token,
    register_client as wordpress_register_client,
    token_expired as wordpress_token_expired,
)
from .llm.factory import get_default_model, get_llm_provider, get_provider_config, resolve_api_key
from .llm.openai_compatible_client import OpenAICompatibleClient
from .storage.state import load_config, reset_storage, setup_storage, write_config

app = typer.Typer(add_completion=False)
github_app = typer.Typer(help="GitHub integrations")
ollama_app = typer.Typer(help="Ollama diagnostics")
llm_app = typer.Typer(help="LLM provider diagnostics")
mcp_app = typer.Typer(help="MCP integrations")
wordpress_app = typer.Typer(help="WordPress integrations")
app.add_typer(github_app, name="github")
app.add_typer(ollama_app, name="ollama")
app.add_typer(llm_app, name="llm")
app.add_typer(mcp_app, name="mcp")
app.add_typer(wordpress_app, name="wordpress")

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


def _wordpress_mcp_connector(config: dict) -> Any:
    registry = build_mcp_registry(config)
    connector = registry.get("wpcom-mcp")
    if connector is None:
        raise RuntimeError("MCP server 'wpcom-mcp' is not configured or not enabled.")
    return connector


def _maybe_refresh_wordpress_token(config: dict) -> tuple[dict, bool]:
    wp_cfg = config.get("wordpress", {})
    if not isinstance(wp_cfg, dict):
        return config, False
    token = wp_cfg.get("token")
    refresh_token = wp_cfg.get("refresh_token")
    client_id = wp_cfg.get("client_id")
    if not (isinstance(token, str) and token and isinstance(refresh_token, str) and isinstance(client_id, str)):
        return config, False
    if not wordpress_token_expired(wp_cfg):
        return config, False

    refreshed = wordpress_refresh_access_token(
        client_id=client_id,
        refresh_token=refresh_token,
        timeout=30.0,
    )
    wp_cfg["token"] = refreshed.get("access_token")
    wp_cfg["refresh_token"] = refreshed.get("refresh_token") or wp_cfg.get("refresh_token")
    wp_cfg["token_type"] = refreshed.get("token_type")
    wp_cfg["scope"] = refreshed.get("scope") or wp_cfg.get("scope")
    wp_cfg["expires_in"] = refreshed.get("expires_in")
    expires_in = refreshed.get("expires_in")
    if isinstance(expires_in, (int, float)):
        wp_cfg["expires_at"] = int(time.time() + int(expires_in))
    config["wordpress"] = wp_cfg

    mcp_cfg = config.get("mcp", {})
    if isinstance(mcp_cfg, dict):
        servers = mcp_cfg.get("servers", {})
        if isinstance(servers, dict):
            wp_server = servers.get("wpcom-mcp", {})
            if isinstance(wp_server, dict):
                auth_cfg = wp_server.get("auth", {})
                if not isinstance(auth_cfg, dict):
                    auth_cfg = {}
                auth_cfg["token"] = wp_cfg.get("token")
                wp_server["auth"] = auth_cfg
                servers["wpcom-mcp"] = wp_server
                mcp_cfg["servers"] = servers
                config["mcp"] = mcp_cfg
    return config, True


def _site_identifier(site: dict[str, Any]) -> str:
    for key in ("domain", "site_url", "URL", "url", "site_URL", "slug", "blogname", "name"):
        value = site.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    site_id = site.get("ID")
    if site_id is not None:
        return str(site_id)
    return ""


def _build_content_authoring_execute_payload(site_ref: str, confirmation_text: str) -> dict[str, Any]:
    return {
        "wpcom_site": site_ref,
        "action": "execute",
        "operation": "posts.create",
        "params": {
            "title": "Test",
            "content": "test",
            "status": "draft",
            "user_confirmed": confirmation_text,
        },
    }


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
        write_config(config, force=True)
        typer.echo(f"Primary model set to: {selected_model}")
        return

    if provider == "openrouter":
        base_url = _ask_text("OpenRouter base URL", _openrouter_base_url_from_config(config))
        provider_cfg = get_provider_config(config, "openrouter")
        env_name, direct_key = _ask_api_key_strategy(str(provider_cfg.get("api_key_env", "OPENROUTER_API_KEY")))
        provider_cfg["base_url"] = base_url
        provider_cfg["api_key_env"] = env_name
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

        api_key = resolve_api_key(provider_cfg, default_env_name="OPENROUTER_API_KEY")
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
        write_config(config, force=True)
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
    write_config(config, force=True)
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
        api_key = resolve_api_key(cfg, default_env_name="OPENROUTER_API_KEY")
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
                    typer.echo(f"Error: {exc}", err=True)
                    sys.exit(1)
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
        typer.echo(f"Error: {exc}", err=True)
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
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open authorization URL in browser"),
    force_register: bool = typer.Option(False, "--force-register", help="Register a fresh OAuth client id"),
    callback_timeout: int = typer.Option(240, "--callback-timeout", help="Seconds to wait for browser callback"),
) -> None:
    config = load_config()
    wp_cfg = config.setdefault("wordpress", {})
    if not isinstance(wp_cfg, dict):
        wp_cfg = {}
        config["wordpress"] = wp_cfg

    redirect_uri = str(wp_cfg.get("redirect_uri") or WORDPRESS_DEFAULT_REDIRECT_URI)
    scope = str(wp_cfg.get("scope") or "auth")
    client_name = str(wp_cfg.get("client_name") or "Thistlebot WordPress MCP")
    timeout = 30.0

    client_id = wp_cfg.get("client_id")
    if force_register or not client_id:
        try:
            registration = wordpress_register_client(client_name=client_name, redirect_uri=redirect_uri, timeout=timeout)
        except Exception as exc:
            typer.echo(f"WordPress client registration failed: {exc}", err=True)
            raise typer.Exit(code=1)
        client_id = str(registration.get("client_id") or "")
        if not client_id:
            typer.echo("WordPress client registration did not return client_id", err=True)
            raise typer.Exit(code=1)
        wp_cfg["client_id"] = client_id
        wp_cfg["registration"] = registration

    typer.echo("Starting WordPress OAuth login flow...")
    try:
        token_data, authorize_url = login_with_authorization_code_flow(
            client_id=str(client_id),
            redirect_uri=redirect_uri,
            scope=scope,
            timeout=timeout,
            callback_timeout=callback_timeout,
            open_browser=open_browser,
        )
    except Exception as exc:
        typer.echo(f"WordPress login failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if not open_browser:
        typer.echo("Open this URL to authorize:")
        typer.echo(authorize_url)

    wp_cfg["token"] = token_data.get("access_token")
    wp_cfg["refresh_token"] = token_data.get("refresh_token")
    wp_cfg["token_type"] = token_data.get("token_type")
    wp_cfg["scope"] = token_data.get("scope") or scope
    wp_cfg["expires_in"] = token_data.get("expires_in")
    wp_cfg["expires_at"] = token_data.get("expires_at")
    wp_cfg["redirect_uri"] = redirect_uri
    wp_cfg["client_name"] = client_name

    mcp_cfg = config.setdefault("mcp", {})
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
        config["mcp"] = mcp_cfg
    mcp_cfg["enabled"] = True
    servers = mcp_cfg.setdefault("servers", {})
    if not isinstance(servers, dict):
        servers = {}
        mcp_cfg["servers"] = servers

    server_cfg = servers.setdefault("wpcom-mcp", {})
    if not isinstance(server_cfg, dict):
        server_cfg = {}
        servers["wpcom-mcp"] = server_cfg
    server_cfg["enabled"] = True
    server_cfg["transport"] = "http"
    server_cfg["url"] = "https://public-api.wordpress.com/wpcom/v2/mcp/v1"
    server_cfg["timeout_seconds"] = 30
    auth_cfg = server_cfg.setdefault("auth", {})
    if not isinstance(auth_cfg, dict):
        auth_cfg = {}
        server_cfg["auth"] = auth_cfg
    auth_cfg["type"] = "bearer"
    auth_cfg["token"] = wp_cfg.get("token")
    auth_cfg["token_env"] = auth_cfg.get("token_env") or "WORDPRESS_ACCESS_TOKEN"

    write_config(config, force=True)
    typer.echo("WordPress token stored in ~/.thistlebot/config.json")
    typer.echo("MCP server 'wpcom-mcp' enabled.")


@wordpress_app.command("status")
def wordpress_status() -> None:
    config = load_config()
    wp_cfg = config.get("wordpress", {})
    if not isinstance(wp_cfg, dict):
        wp_cfg = {}

    token = wp_cfg.get("token")
    refresh_token = wp_cfg.get("refresh_token")
    client_id = wp_cfg.get("client_id")

    typer.echo(f"WordPress client_id: {'present' if client_id else 'missing'}")
    typer.echo(f"WordPress token: {'present' if token else 'missing'}")
    typer.echo(f"WordPress refresh_token: {'present' if refresh_token else 'missing'}")

    expires_at = wp_cfg.get("expires_at")
    if token and isinstance(expires_at, (int, float)):
        if wordpress_token_expired({"expires_at": expires_at}):
            typer.echo("WordPress token state: expired")
        else:
            typer.echo("WordPress token state: active")

    mcp_cfg = config.get("mcp", {})
    servers = mcp_cfg.get("servers", {}) if isinstance(mcp_cfg, dict) else {}
    wp_server = servers.get("wpcom-mcp", {}) if isinstance(servers, dict) else {}
    enabled = bool(isinstance(wp_server, dict) and wp_server.get("enabled"))
    typer.echo(f"MCP wpcom-mcp: {'enabled' if enabled else 'disabled'}")

    if token and isinstance(wp_cfg.get("refresh_token"), str) and isinstance(client_id, str):
        if wordpress_token_expired(wp_cfg):
            try:
                config, refreshed = _maybe_refresh_wordpress_token(config)
                if not refreshed:
                    return
                write_config(config, force=True)
                typer.echo("WordPress token auto-refreshed.")
            except Exception as exc:
                typer.echo(f"WordPress token refresh failed: {exc}")


@wordpress_app.command("logout")
def wordpress_logout(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    config = load_config()
    if not yes and not typer.confirm("Clear stored WordPress token and disable wpcom-mcp?", default=True):
        typer.echo("Aborted.")
        return

    wp_cfg = config.get("wordpress", {})
    if not isinstance(wp_cfg, dict):
        wp_cfg = {}
    wp_cfg["token"] = None
    wp_cfg["refresh_token"] = None
    wp_cfg["expires_in"] = None
    wp_cfg["expires_at"] = None
    wp_cfg["token_type"] = None
    config["wordpress"] = wp_cfg

    mcp_cfg = config.get("mcp", {})
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
    servers = mcp_cfg.get("servers", {})
    if not isinstance(servers, dict):
        servers = {}
    wp_server = servers.get("wpcom-mcp", {})
    if not isinstance(wp_server, dict):
        wp_server = {}
    wp_server["enabled"] = False
    auth_cfg = wp_server.get("auth", {})
    if not isinstance(auth_cfg, dict):
        auth_cfg = {}
    auth_cfg["token"] = None
    wp_server["auth"] = auth_cfg
    servers["wpcom-mcp"] = wp_server
    mcp_cfg["servers"] = servers

    if not any(bool(isinstance(v, dict) and v.get("enabled")) for v in servers.values()):
        mcp_cfg["enabled"] = False

    config["mcp"] = mcp_cfg
    write_config(config, force=True)
    typer.echo("WordPress credentials cleared and wpcom-mcp disabled.")


@wordpress_app.command("sites")
def wordpress_sites() -> None:
    config = load_config()
    try:
        config, refreshed = _maybe_refresh_wordpress_token(config)
        if refreshed:
            write_config(config, force=True)
    except Exception as exc:
        typer.echo(f"WordPress token refresh failed: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        connector = _wordpress_mcp_connector(config)
        result = connector.invoke("wpcom-mcp-user-sites", {})
    except Exception as exc:
        typer.echo(f"WordPress sites lookup failed: {exc}", err=True)
        raise typer.Exit(code=1)

    sites = _extract_sites_from_result(result)
    if not sites:
        text = _extract_text_from_tool_response(result).strip()
        typer.echo("No site/blog data parsed from tool output.")
        if text:
            typer.echo(text)
        return

    for site in sites:
        label = site.get("blogname") or site.get("name") or site.get("title") or "(unnamed)"
        domain = site.get("domain")
        url = site.get("site_url") or site.get("URL") or site.get("url") or site.get("site_URL")
        row = str(label)
        if isinstance(domain, str) and domain:
            row += f" domain={domain}"
        if isinstance(url, str) and url:
            row += f" url={url}"
        typer.echo(row)


@wordpress_app.command("test")
def wordpress_test(
    site: Optional[str] = typer.Option(None, "--site", help="Target site/domain to publish test post"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt and publish test post"),
) -> None:
    config = load_config()
    try:
        config, refreshed = _maybe_refresh_wordpress_token(config)
        if refreshed:
            write_config(config, force=True)
    except Exception as exc:
        typer.echo(f"WordPress token refresh failed: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        connector = _wordpress_mcp_connector(config)
        sites_result = connector.invoke("wpcom-mcp-user-sites", {})
    except Exception as exc:
        typer.echo(f"WordPress site discovery failed: {exc}", err=True)
        raise typer.Exit(code=1)

    sites = _extract_sites_from_result(sites_result)
    if not sites:
        typer.echo("Unable to parse available sites from WordPress MCP response.", err=True)
        raise typer.Exit(code=1)

    typer.echo("Available blogs/sites:")
    for idx, item in enumerate(sites, start=1):
        label = item.get("blogname") or item.get("name") or item.get("title") or "(unnamed)"
        ref = _site_identifier(item)
        typer.echo(f"{idx}. {label} ({ref})")

    site_ref = ""
    if site:
        site_ref = site
    else:
        if sys.stdin.isatty() and sys.stdout.isatty():
            choices: list[questionary.Choice] = []
            for item in sites:
                label = item.get("blogname") or item.get("name") or item.get("title") or "(unnamed)"
                ref = _site_identifier(item)
                if not ref:
                    continue
                choices.append(questionary.Choice(title=f"{label} ({ref})", value=ref))
            if choices:
                selected = questionary.select("Select site for test post", choices=choices).ask()
                if isinstance(selected, str) and selected:
                    site_ref = selected
        if not site_ref:
            site_ref = _site_identifier(sites[0])

    if not site_ref:
        typer.echo("Could not determine a valid site identifier.", err=True)
        raise typer.Exit(code=1)

    if not yes and not typer.confirm(f"Publish a test draft post to '{site_ref}'?", default=False):
        typer.echo("Cancelled. No post created.")
        return

    try:
        tools = connector.list_tools()
    except Exception as exc:
        typer.echo(f"Unable to list WordPress tools: {exc}", err=True)
        raise typer.Exit(code=1)

    content_tool = next((tool for tool in tools if str(tool.get("name")) == "wpcom-mcp-content-authoring"), None)
    if not content_tool:
        typer.echo("Tool 'wpcom-mcp-content-authoring' not available for this account/role.", err=True)
        raise typer.Exit(code=1)

    confirmation_text = "yes" if yes else "user confirmed in cli"
    payload = _build_content_authoring_execute_payload(site_ref, confirmation_text)

    try:
        result = connector.invoke("wpcom-mcp-content-authoring", payload)
    except Exception as exc:
        typer.echo(f"Test post creation failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Test publish call completed.")
    text = _extract_text_from_tool_response(result).strip()
    if text:
        typer.echo(text)
    else:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


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


@app.command()
def mcp_connect() -> None:
    typer.echo("Deprecated. Use 'thistlebot mcp status' and 'thistlebot mcp tools'.")


if __name__ == "__main__":
    app()
