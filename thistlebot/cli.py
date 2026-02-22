from __future__ import annotations

import sys
import webbrowser
from typing import Optional

import httpx
import typer

from .core.chat_client import stream_chat
from .core.gateway import run_gateway
from .integrations.github.oauth import login_with_device_flow, poll_for_token
from .storage.state import load_config, reset_storage, setup_storage, write_config

app = typer.Typer(add_completion=False)
github_app = typer.Typer(help="GitHub integrations")
app.add_typer(github_app, name="github")


@app.command()
def setup(force: bool = typer.Option(False, "--force", help="Overwrite existing config/prompts")) -> None:
    config = setup_storage(force=force)
    typer.echo("Created ~/.thistlebot structure")

    ollama_url = config.get("ollama", {}).get("base_url", "")
    if ollama_url:
        typer.echo(f"Ollama base URL: {ollama_url}")


@app.command()
def reset() -> None:
    config = reset_storage()
    typer.echo("Reset ~/.thistlebot config and prompts to defaults")

    ollama_url = config.get("ollama", {}).get("base_url", "")
    if ollama_url:
        typer.echo(f"Ollama base URL: {ollama_url}")


@app.command()
def gateway(host: Optional[str] = None, port: Optional[int] = None) -> None:
    run_gateway(host=host, port=port)


@app.command()
def chat(
    session: str = typer.Option("default", "--session"),
    model: Optional[str] = typer.Option(None, "--model"),
    gateway_url: str = typer.Option("http://127.0.0.1:7788", "--gateway"),
) -> None:
    config = load_config()
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
            assistant_content = ""
            for chunk in stream_chat(gateway_url, messages, active_model, session):
                assistant_content += chunk
                typer.echo(chunk, nl=False)
            typer.echo("")
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            sys.exit(1)


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
