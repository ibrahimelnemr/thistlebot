from __future__ import annotations

import sys
import time
import webbrowser
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
    DEFAULT_REDIRECT_URI as WORDPRESS_REST_DEFAULT_REDIRECT_URI,
    login_with_authorization_code_flow as wordpress_rest_login_flow,
    token_expired as wordpress_rest_token_expired,
)
from .llm.factory import get_default_model, get_llm_provider, get_provider_config, resolve_api_key
from .llm.openai_compatible_client import OpenAICompatibleClient
from .storage.state import load_config, reset_storage, setup_storage, write_config

app = typer.Typer(add_completion=False)
github_app = typer.Typer(help="GitHub integrations")
ollama_app = typer.Typer(help="Ollama diagnostics")
llm_app = typer.Typer(help="LLM provider diagnostics")
mcp_app = typer.Typer(help="MCP integrations")
wordpress_app = typer.Typer(help="WordPress integrations (REST)")
agent_app = typer.Typer(help="Persistent agent management")
blogger_app = typer.Typer(help="Autonomous blogging agent")
app.add_typer(github_app, name="github")
app.add_typer(ollama_app, name="ollama")
app.add_typer(llm_app, name="llm")
app.add_typer(mcp_app, name="mcp")
app.add_typer(wordpress_app, name="wordpress")
app.add_typer(agent_app, name="agent")
agent_app.add_typer(blogger_app, name="blogger")

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


def _wordpress_rest_config(config: dict) -> dict[str, Any]:
    cfg = config.get("wordpress", {})
    if not isinstance(cfg, dict) or not cfg:
        cfg = config.get("wordpress_rest", {})
    if isinstance(cfg, dict):
        return cfg
    return {}


def _wordpress_rest_client(config: dict) -> WordPressRestClient:
    rest_cfg = _wordpress_rest_config(config)
    token = rest_cfg.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("WordPress token missing. Run 'thistlebot wordpress login'.")
    if wordpress_rest_token_expired(rest_cfg):
        raise RuntimeError("WordPress token expired. Run 'thistlebot wordpress login' to refresh it.")
    timeout = rest_cfg.get("timeout_seconds")
    timeout_value = float(timeout) if isinstance(timeout, (int, float)) else 30.0
    return WordPressRestClient(access_token=token, timeout_seconds=timeout_value)


def _wordpress_rest_site_ref(config: dict, explicit_site: str | None) -> str:
    if explicit_site:
        return explicit_site
    rest_cfg = _wordpress_rest_config(config)
    blog = rest_cfg.get("blog")
    if isinstance(blog, str) and blog.strip():
        return blog.strip()
    blog_url = rest_cfg.get("blog_url")
    if isinstance(blog_url, str) and blog_url.strip():
        return blog_url.strip()
    blog_id = rest_cfg.get("blog_id")
    if isinstance(blog_id, (int, float)):
        return str(int(blog_id))
    if isinstance(blog_id, str) and blog_id.strip():
        return blog_id.strip()
    raise RuntimeError("Missing site/blog reference. Provide --site or set wordpress.blog in config.")


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
def wordpress_rest_login(
    client_id: str = typer.Option("", "--client-id", help="WordPress.com OAuth app client_id"),
    client_secret: str = typer.Option("", "--client-secret", help="WordPress.com OAuth app client_secret"),
    scope: str = typer.Option("posts", "--scope", help="OAuth scope(s), e.g. 'posts media'"),
    blog: str = typer.Option("", "--blog", help="Optional site domain/id for single-site token"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open authorization URL in browser"),
    callback_timeout: int = typer.Option(240, "--callback-timeout", help="Seconds to wait for browser callback"),
) -> None:
    config = load_config()
    rest_cfg = config.setdefault("wordpress", {})
    if not isinstance(rest_cfg, dict):
        rest_cfg = {}
        config["wordpress"] = rest_cfg

    configured_client_id = client_id.strip() or str(rest_cfg.get("client_id") or "")
    configured_client_secret = client_secret.strip() or str(rest_cfg.get("client_secret") or "")
    if not configured_client_id or not configured_client_secret:
        typer.echo("WordPress REST login requires --client-id and --client-secret (or stored values).", err=True)
        raise typer.Exit(code=1)

    redirect_uri = str(rest_cfg.get("redirect_uri") or WORDPRESS_REST_DEFAULT_REDIRECT_URI)
    selected_blog = blog.strip() or str(rest_cfg.get("blog") or "")

    typer.echo("Starting WordPress REST OAuth login flow...")
    try:
        token_data, authorize_url = wordpress_rest_login_flow(
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
        typer.echo(f"WordPress REST login failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if not open_browser:
        typer.echo("Open this URL to authorize:")
        typer.echo(authorize_url)

    rest_cfg["enabled"] = True
    rest_cfg["client_id"] = configured_client_id
    rest_cfg["client_secret"] = configured_client_secret
    rest_cfg["redirect_uri"] = redirect_uri
    rest_cfg["scope"] = token_data.get("scope") or scope
    rest_cfg["blog"] = selected_blog or None
    rest_cfg["blog_id"] = token_data.get("blog_id")
    rest_cfg["blog_url"] = token_data.get("blog_url")
    rest_cfg["token"] = token_data.get("access_token")
    rest_cfg["token_type"] = token_data.get("token_type")
    rest_cfg["expires_in"] = token_data.get("expires_in")
    rest_cfg["expires_at"] = token_data.get("expires_at")

    config["wordpress"] = rest_cfg
    config["wordpress_rest"] = rest_cfg
    write_config(config, force=True)
    typer.echo("WordPress REST token stored in ~/.thistlebot/config.json")


@wordpress_app.command("status")
def wordpress_rest_status() -> None:
    config = load_config()
    rest_cfg = _wordpress_rest_config(config)

    enabled = bool(rest_cfg.get("enabled"))
    token = rest_cfg.get("token")
    client_id = rest_cfg.get("client_id")
    client_secret = rest_cfg.get("client_secret")

    typer.echo(f"WordPress REST enabled: {'yes' if enabled else 'no'}")
    typer.echo(f"WordPress REST client_id: {'present' if client_id else 'missing'}")
    typer.echo(f"WordPress REST client_secret: {'present' if client_secret else 'missing'}")
    typer.echo(f"WordPress REST token: {'present' if token else 'missing'}")

    expires_at = rest_cfg.get("expires_at")
    if token and isinstance(expires_at, (int, float)):
        state = "expired" if wordpress_rest_token_expired(rest_cfg) else "active"
        typer.echo(f"WordPress REST token state: {state}")


@wordpress_app.command("sites")
def wordpress_rest_sites() -> None:
    config = load_config()
    rest_cfg = _wordpress_rest_config(config)
    try:
        client = _wordpress_rest_client(config)
        try:
            result = client.list_sites()
        except Exception:
            # Some tokens are single-blog scoped and cannot call /me/sites.
            client_id = rest_cfg.get("client_id")
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
        typer.echo(f"WordPress REST sites lookup failed: {exc}", err=True)
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
            rest_cfg["blog_id"] = site_id
        if isinstance(site_url, str) and site_url:
            rest_cfg["blog_url"] = site_url
            if not rest_cfg.get("blog"):
                parsed = urllib.parse.urlparse(site_url)
                if parsed.netloc:
                    rest_cfg["blog"] = parsed.netloc
                else:
                    rest_cfg["blog"] = site_url
        config["wordpress"] = rest_cfg
        config["wordpress_rest"] = rest_cfg
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
def wordpress_rest_create_post(
    title: str = typer.Option(..., "--title", help="Post title"),
    content: str = typer.Option(..., "--content", help="Post content"),
    site: Optional[str] = typer.Option(None, "--site", help="Target site domain or site ID"),
    status: str = typer.Option("draft", "--status", help="Post status: draft|publish|pending|future|private"),
) -> None:
    config = load_config()
    try:
        site_ref = _wordpress_rest_site_ref(config, site)
        client = _wordpress_rest_client(config)
        result = client.create_post(site_ref, title=title, content=content, status=status)
    except Exception as exc:
        typer.echo(f"WordPress REST create post failed: {exc}", err=True)
        raise typer.Exit(code=1)

    post_id = result.get("ID") if isinstance(result, dict) else None
    post_url = result.get("URL") if isinstance(result, dict) else None
    typer.echo(f"WordPress REST post created on '{site_ref}'.")
    if post_id is not None:
        typer.echo(f"Post ID: {post_id}")
    if isinstance(post_url, str) and post_url:
        typer.echo(f"Post URL: {post_url}")


@wordpress_app.command("test")
def wordpress_rest_test(
    site: Optional[str] = typer.Option(None, "--site", help="Target site domain or site ID"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt and create test post"),
) -> None:
    config = load_config()
    try:
        site_ref = _wordpress_rest_site_ref(config, site)
        if not yes and not typer.confirm(f"Create a REST test draft post on '{site_ref}'?", default=False):
            typer.echo("Cancelled. No post created.")
            return
        client = _wordpress_rest_client(config)
        result = client.create_post(site_ref, title="Test", content="test", status="draft")
    except Exception as exc:
        typer.echo(f"WordPress REST test failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("WordPress REST test post created successfully.")
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
    if not yes and not typer.confirm("Clear stored WordPress REST credentials?", default=True):
        typer.echo("Aborted.")
        return

    rest_cfg = _wordpress_rest_config(config)
    for key in ("token", "token_type", "expires_in", "expires_at", "blog", "blog_id", "blog_url"):
        rest_cfg[key] = None
    config["wordpress"] = rest_cfg
    config["wordpress_rest"] = rest_cfg
    write_config(config, force=True)
    typer.echo("WordPress REST credentials cleared.")


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


# ---------------------------------------------------------------------------
# Agent: Blogger
# ---------------------------------------------------------------------------


@blogger_app.command("run")
def blogger_run(
    topic: Optional[str] = typer.Option(None, "--topic", help="Override the configured topic"),
    status: Optional[str] = typer.Option(None, "--status", help="WordPress post status (draft/publish)"),
) -> None:
    """Execute one blogger workflow: research, draft, and publish."""
    from .agents.blogger.workflow import run_publish_workflow

    console = RICH_CONSOLE

    def _on_step(step_name: str, step_status: str) -> None:
        if step_status == "started":
            console.print(f"[cyan]>> Step: {step_name}...[/cyan]")
        elif step_status == "completed":
            console.print(f"[green]   {step_name} completed.[/green]")

    console.print("[bold]Blogger agent — running publish workflow[/bold]")

    try:
        result = run_publish_workflow(
            topic_override=topic,
            status_override=status,
            on_step=_on_step,
        )
    except Exception as exc:
        console.print(f"[red]Workflow failed: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print()
    console.print(f"[bold green]Workflow completed![/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    console.print(f"  Topic:   {result['topic']}")
    console.print(f"  Status:  {result['post_status']}")
    console.print(f"  Model:   {result['model']}")

    # Show final summary from the publish step
    from pathlib import Path

    final_path = Path(result["run_dir"]) / "final.md"
    if final_path.exists():
        console.print()
        console.print("[bold]Publish summary:[/bold]")
        console.print(final_path.read_text(encoding="utf-8")[:2000])


@blogger_app.command("status")
def blogger_status(
    limit: int = typer.Option(5, "--limit", "-n", help="Number of recent runs to show"),
) -> None:
    """Show recent blogger workflow runs."""
    from .agents.blogger.config import list_runs, load_blogger_config

    console = RICH_CONSOLE
    blogger_cfg = load_blogger_config()
    console.print(f"[bold]Blogger agent[/bold]")
    console.print(f"  Site:  {blogger_cfg.get('site', 'not configured')}")
    console.print(f"  Topic: {blogger_cfg.get('topic', 'not configured')}")
    console.print()

    runs = list_runs()
    if not runs:
        console.print("No runs found.")
        return

    console.print(f"[bold]Recent runs (latest {limit}):[/bold]")
    for run_dir in runs[:limit]:
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            run_status = meta.get("status", "unknown")
            steps = meta.get("steps", {})
            ts = meta.get("timestamp", "")
            console.print(f"  {run_dir.name}  status={run_status}  steps={steps}  ts={ts}")
        else:
            # Infer status from which files exist
            files = [f.name for f in run_dir.iterdir() if f.is_file()]
            console.print(f"  {run_dir.name}  files={files}")


@blogger_app.command("config")
def blogger_config_show() -> None:
    """Show current blogger configuration."""
    from .agents.blogger.config import load_blogger_config

    import json

    cfg = load_blogger_config()
    RICH_CONSOLE.print_json(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    app()
