from __future__ import annotations

import sys
import webbrowser
from typing import Optional
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
from .llm.factory import get_default_model, get_llm_provider, get_provider_config, resolve_api_key
from .llm.openai_compatible_client import OpenAICompatibleClient
from .storage.state import load_config, reset_storage, setup_storage, write_config

app = typer.Typer(add_completion=False)
github_app = typer.Typer(help="GitHub integrations")
ollama_app = typer.Typer(help="Ollama diagnostics")
llm_app = typer.Typer(help="LLM provider diagnostics")
mcp_app = typer.Typer(help="MCP integrations")
app.add_typer(github_app, name="github")
app.add_typer(ollama_app, name="ollama")
app.add_typer(llm_app, name="llm")
app.add_typer(mcp_app, name="mcp")

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
        typer.echo(f"{name}: connected={connected} transport={transport}")
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
