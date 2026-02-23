from __future__ import annotations

import sys
import webbrowser
from typing import Optional

import httpx
import questionary
import typer
from rich.console import Console
from rich.markdown import Markdown

from .core.chat_client import stream_chat
from .core.gateway import run_gateway
from .core.meeting_graph import MeetingConfig, run_meeting_graph
from .integrations.github.oauth import login_with_device_flow, poll_for_token
from .storage.state import load_config, reset_storage, setup_storage, write_config

app = typer.Typer(add_completion=False)
github_app = typer.Typer(help="GitHub integrations")
ollama_app = typer.Typer(help="Ollama diagnostics")
app.add_typer(github_app, name="github")
app.add_typer(ollama_app, name="ollama")

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


def _ensure_gateway_running(gateway_url: str) -> None:
    health_url = f"{gateway_url.rstrip('/')}/health"
    try:
        response = httpx.get(health_url, timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError:
        typer.echo(
            f"Gateway not reachable at {gateway_url}. Start it with 'thistlebot gateway'.",
            err=True,
        )
        raise typer.Exit(code=1)


def _ollama_base_url_from_config(config: dict) -> str:
    ollama_cfg = config.get("ollama", {})
    return ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")


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


def _no_models_found_message() -> str:
    return (
        "No models found. Check that Ollama is running and endpoint is configured properly "
        "in ~/.thistlebot/config.json (ollama.base_url)."
    )


def _select_primary_model(models: list[str], current_model: str) -> str:
    default_model = current_model if current_model in models else models[0]

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo("Non-interactive terminal detected; selecting model automatically.")
        return default_model

    try:
        selected = questionary.select(
            "Select primary Ollama model",
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


@app.command()
def setup(force: bool = typer.Option(False, "--force", help="Overwrite existing config/prompts")) -> None:
    _print_banner()
    setup_storage(force=force)
    config = load_config()
    typer.echo("Created ~/.thistlebot structure")

    base_url = _ollama_base_url_from_config(config)
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

    current_model = config.get("ollama", {}).get("model", "")
    selected_model = _select_primary_model(models, current_model)
    config.setdefault("ollama", {})
    config["ollama"]["model"] = selected_model
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

    configured_model = config.get("ollama", {}).get("model", "")
    if configured_model:
        typer.echo(f"Configured primary model: {configured_model}")


@app.command()
def gateway(host: Optional[str] = None, port: Optional[int] = None) -> None:
    run_gateway(host=host, port=port)


@app.command()
def chat(
    session: str = typer.Option("default", "--session"),
    model: Optional[str] = typer.Option(None, "--model"),
    render_markdown: bool = typer.Option(
        True,
        "--render-markdown/--no-render-markdown",
        help="Render assistant replies as markdown in the terminal",
    ),
) -> None:
    _print_banner()
    config = load_config()
    gateway_url = _gateway_url_from_config(config)
    _ensure_gateway_running(gateway_url)
    active_model = model or config.get("ollama", {}).get("model", "llama3")

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
            for chunk in stream_chat(gateway_url, messages, active_model, session):
                renderer.feed(chunk)
            assistant_content = renderer.finish()
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            sys.exit(1)


@app.command()
def meeting(
    session: str = typer.Option("meeting", "--session"),
    model_a: Optional[str] = typer.Option(None, "--model-a"),
    model_b: Optional[str] = typer.Option(None, "--model-b"),
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
    _ensure_gateway_running(gateway_url)

    active_model = config.get("ollama", {}).get("model", "llama3")
    selected_model_a = model_a or active_model
    selected_model_b = model_b or active_model

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

    try:
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


@app.command()
def mcp_connect() -> None:
    typer.echo("MCP connect scaffold ready. Configure connectors in ~/.thistlebot/config.json")


if __name__ == "__main__":
    app()
