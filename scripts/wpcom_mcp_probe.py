#!/usr/bin/env python3
"""Standalone WordPress.com MCP probe.

This script performs OAuth 2.1 (authorization code + PKCE), stores tokens locally,
and prints available WordPress.com blogs/sites for the authenticated user.

It can also optionally try `tools/list` against the WordPress.com MCP endpoint.
"""

from __future__ import annotations

import argparse
import anyio
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
except Exception:  # pragma: no cover - resolved at runtime
    ClientSession = None  # type: ignore[assignment]
    create_mcp_http_client = None  # type: ignore[assignment]
    streamable_http_client = None  # type: ignore[assignment]

PUBLIC_API_BASE = "https://public-api.wordpress.com"
REGISTER_URL = f"{PUBLIC_API_BASE}/oauth2-1/register"
AUTHORIZE_URL = f"{PUBLIC_API_BASE}/oauth2-1/authorize"
TOKEN_URL = f"{PUBLIC_API_BASE}/oauth2-1/token"
MCP_URL = f"{PUBLIC_API_BASE}/wpcom/v2/mcp/v1"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/callback"
DEFAULT_SCOPE = "auth"
DEFAULT_CLIENT_NAME = "WordPress MCP Probe"


@dataclass
class OAuthCallbackResult:
    code: str | None = None
    state: str | None = None
    error: str | None = None


class OAuthCallbackServer:
    def __init__(self, redirect_uri: str) -> None:
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.scheme != "http":
            raise ValueError("redirect_uri must use http:// for local callback testing")
        if not parsed.hostname:
            raise ValueError("redirect_uri must include a host")
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("redirect_uri host must be localhost or 127.0.0.1")
        if not parsed.port:
            raise ValueError("redirect_uri must include an explicit port")

        self._host = parsed.hostname
        self._port = parsed.port
        self._path = parsed.path or "/"
        self.result = OAuthCallbackResult()
        self._event = threading.Event()
        self._server: http.server.ThreadingHTTPServer | None = None

    def start(self) -> None:
        callback_path = self._path
        result = self.result
        done_event = self._event

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not Found")
                    return

                params = urllib.parse.parse_qs(parsed.query)
                result.code = _first(params.get("code"))
                result.state = _first(params.get("state"))
                result.error = _first(params.get("error"))
                done_event.set()

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h3>Authorization complete.</h3>"
                    b"<p>You can close this tab and return to the terminal.</p></body></html>"
                )

            def log_message(self, fmt: str, *args: object) -> None:
                return

        self._server = http.server.ThreadingHTTPServer((self._host, self._port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def wait(self, timeout_seconds: int) -> OAuthCallbackResult:
        if not self._event.wait(timeout_seconds):
            raise TimeoutError("Timed out waiting for OAuth callback")
        return self.result

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _first(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[0]


def _pkce_verifier() -> str:
    token = secrets.token_urlsafe(64)
    return token[:96]


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _save_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def register_client(client_name: str, redirect_uri: str, timeout: float) -> dict[str, Any]:
    payload = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
    }
    response = httpx.post(REGISTER_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not body.get("client_id"):
        raise RuntimeError("Registration succeeded but no client_id returned")
    return body


def build_authorize_url(
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    code_challenge: str,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    timeout: float,
) -> dict[str, Any]:
    form = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    response = httpx.post(
        TOKEN_URL,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not body.get("access_token"):
        raise RuntimeError("Token exchange succeeded but no access_token returned")
    return body


def refresh_access_token(
    client_id: str,
    refresh_token: str,
    timeout: float,
) -> dict[str, Any]:
    form = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    response = httpx.post(
        TOKEN_URL,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not body.get("access_token"):
        raise RuntimeError("Refresh succeeded but no access_token returned")
    return body


def token_expired(token_data: dict[str, Any], skew_seconds: int = 30) -> bool:
    expires_at = token_data.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return False
    return time.time() >= float(expires_at) - float(skew_seconds)


def token_from_cache_or_refresh(state: dict[str, Any], timeout: float) -> dict[str, Any] | None:
    token_data = state.get("token")
    client_id = state.get("client_id")
    if not isinstance(token_data, dict) or not isinstance(client_id, str):
        return None

    if not token_expired(token_data):
        return token_data

    refresh_token_value = token_data.get("refresh_token")
    if not isinstance(refresh_token_value, str) or not refresh_token_value:
        return None

    refreshed = refresh_access_token(client_id, refresh_token_value, timeout)
    normalized = normalize_token_data(refreshed)
    state["token"] = normalized
    return normalized


def normalize_token_data(raw: dict[str, Any]) -> dict[str, Any]:
    token_data = dict(raw)
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        token_data["expires_at"] = int(time.time() + int(expires_in))
    return token_data


def _normalize_tool(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "input_schema": tool.get("inputSchema") or tool.get("input_schema") or {},
        }
    return {
        "name": str(getattr(tool, "name", "") or ""),
        "description": str(getattr(tool, "description", "") or ""),
        "input_schema": getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {},
    }


def _extract_text_content(result: Any) -> str:
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks)


def _parse_json_from_text_blob(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _collect_sites_like(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                key_lower = str(key).lower()
                if key_lower in {"sites", "blogs"} and isinstance(child, list):
                    for item in child:
                        if isinstance(item, dict):
                            found.append(item)
                walk(child)
            return

        if isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return found


def _tool_needs_no_required_args(input_schema: Any) -> bool:
    if not isinstance(input_schema, dict):
        return True
    required = input_schema.get("required")
    if not isinstance(required, list):
        return True
    return len(required) == 0


async def probe_wordpress_mcp(access_token: str, timeout: float) -> dict[str, Any]:
    if ClientSession is None or create_mcp_http_client is None or streamable_http_client is None:
        raise RuntimeError("MCP SDK transport not available. Ensure package 'mcp' is installed.")

    headers = {"Authorization": f"Bearer {access_token}"}
    http_client = create_mcp_http_client(headers=headers, timeout=httpx.Timeout(timeout))

    async with streamable_http_client(MCP_URL, http_client=http_client) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            init_result = await session.initialize()
            tools_response = await session.list_tools()
            raw_tools = getattr(tools_response, "tools", [])
            tools = [_normalize_tool(t) for t in raw_tools]

            site_candidates = [
                tool
                for tool in tools
                if "site" in tool["name"].lower() or "blog" in tool["name"].lower()
            ]

            sites: list[dict[str, Any]] = []
            attempted_calls: list[str] = []
            call_errors: list[str] = []
            for tool in site_candidates:
                tool_name = tool["name"]
                if not _tool_needs_no_required_args(tool.get("input_schema")):
                    continue
                attempted_calls.append(tool_name)
                try:
                    result = await session.call_tool(tool_name, {})
                    text_blob = _extract_text_content(result)
                    parsed = _parse_json_from_text_blob(text_blob)
                    if parsed is not None:
                        sites.extend(_collect_sites_like(parsed))
                except Exception as exc:  # noqa: BLE001
                    call_errors.append(f"{tool_name}: {exc}")

            return {
                "session_id": get_session_id(),
                "protocol_version": str(getattr(init_result, "protocolVersion", "unknown")),
                "tools": tools,
                "site_candidates": [tool["name"] for tool in site_candidates],
                "attempted_calls": attempted_calls,
                "call_errors": call_errors,
                "sites": sites,
            }


def print_sites(sites: list[dict[str, Any]]) -> None:
    if not sites:
        print("No blogs/sites returned for this account.")
        return

    print("\nAvailable blogs/sites:")
    for site in sites:
        if not isinstance(site, dict):
            continue
        site_id = site.get("ID")
        name = site.get("name") or site.get("title") or "(unnamed)"
        url = site.get("URL") or site.get("url") or site.get("site_URL") or ""
        domain = site.get("domain") or ""

        line = f"- {name}"
        if isinstance(site_id, int):
            line += f" [id={site_id}]"
        if isinstance(domain, str) and domain:
            line += f" domain={domain}"
        if isinstance(url, str) and url:
            line += f" url={url}"
        print(line)


def ensure_client_registration(
    state: dict[str, Any],
    client_name: str,
    redirect_uri: str,
    timeout: float,
    force_register: bool,
) -> str:
    if not force_register:
        existing_client_id = state.get("client_id")
        existing_redirect = state.get("redirect_uri")
        if isinstance(existing_client_id, str) and existing_client_id and existing_redirect == redirect_uri:
            return existing_client_id

    print("Registering OAuth client with WordPress.com...")
    registration = register_client(client_name, redirect_uri, timeout)
    client_id = str(registration["client_id"])
    state["client_id"] = client_id
    state["redirect_uri"] = redirect_uri
    state["client_registration"] = registration
    return client_id


def run_oauth_flow(
    state: dict[str, Any],
    client_id: str,
    redirect_uri: str,
    scope: str,
    timeout: float,
    callback_timeout: int,
) -> dict[str, Any]:
    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    expected_state = secrets.token_urlsafe(24)

    auth_url = build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=expected_state,
        code_challenge=challenge,
    )

    callback_server = OAuthCallbackServer(redirect_uri)
    callback_server.start()
    print("\nOpen this URL to authorize access:")
    print(auth_url)

    opened = webbrowser.open(auth_url)
    if opened:
        print("\nA browser window/tab should have opened.")
    else:
        print("\nCould not open browser automatically; open the URL manually.")

    try:
        callback_result = callback_server.wait(callback_timeout)
    finally:
        callback_server.stop()

    if callback_result.error:
        raise RuntimeError(f"Authorization failed with error: {callback_result.error}")
    if not callback_result.code:
        raise RuntimeError("Authorization callback did not include code")
    if callback_result.state != expected_state:
        raise RuntimeError("State mismatch in OAuth callback")

    token_raw = exchange_code_for_tokens(
        client_id=client_id,
        code=callback_result.code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
        timeout=timeout,
    )
    return normalize_token_data(token_raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WordPress.com MCP OAuth probe")
    parser.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI, help="OAuth redirect URI")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, help="OAuth scope")
    parser.add_argument("--client-name", default=DEFAULT_CLIENT_NAME, help="OAuth dynamic registration name")
    parser.add_argument(
        "--state-file",
        default=str(Path.home() / ".wpcom_mcp_probe.json"),
        help="Path to store client registration and tokens",
    )
    parser.add_argument("--force-register", action="store_true", help="Force dynamic client re-registration")
    parser.add_argument("--force-login", action="store_true", help="Ignore cached token and run OAuth login")
    parser.add_argument(
        "--show-all-tools",
        action="store_true",
        help="Print all discovered MCP tool names",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout seconds")
    parser.add_argument(
        "--callback-timeout",
        type=int,
        default=180,
        help="Seconds to wait for OAuth browser callback",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_file = Path(os.path.expanduser(args.state_file)).resolve()

    try:
        state = _load_state(state_file)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load state file {state_file}: {exc}", file=sys.stderr)
        return 1

    try:
        client_id = ensure_client_registration(
            state=state,
            client_name=args.client_name,
            redirect_uri=args.redirect_uri,
            timeout=args.timeout,
            force_register=args.force_register,
        )

        token_data: dict[str, Any] | None = None
        if not args.force_login:
            token_data = token_from_cache_or_refresh(state, args.timeout)

        if token_data is None:
            print("Starting OAuth login flow...")
            token_data = run_oauth_flow(
                state=state,
                client_id=client_id,
                redirect_uri=args.redirect_uri,
                scope=args.scope,
                timeout=args.timeout,
                callback_timeout=args.callback_timeout,
            )
            state["token"] = token_data

        _save_state(state_file, state)

        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("Missing access_token after auth")

        print("\nConnecting to WordPress MCP...")
        probe = anyio.run(probe_wordpress_mcp, access_token, args.timeout)
        print(f"MCP session id: {probe.get('session_id')}")
        print(f"MCP protocol version: {probe.get('protocol_version')}")
        tools = probe.get("tools", [])
        print(f"Discovered MCP tools: {len(tools)}")
        if args.show_all_tools:
            for tool in tools:
                print(f"- {tool.get('name')}")

        sites = probe.get("sites", [])
        print_sites(sites if isinstance(sites, list) else [])

        site_candidates = probe.get("site_candidates", [])
        attempted_calls = probe.get("attempted_calls", [])
        if not sites:
            print("\nSite/blog auto-discovery notes:")
            print(f"- site/blog candidate tools found: {len(site_candidates)}")
            if site_candidates:
                print(f"- candidate tools: {', '.join(site_candidates)}")
            print(f"- candidate tools called with empty args: {len(attempted_calls)}")
            call_errors = probe.get("call_errors", [])
            if call_errors:
                print("- call errors:")
                for item in call_errors[:10]:
                    print(f"  {item}")
            print("- If no sites were found automatically, rerun with --show-all-tools and call a specific site tool manually.")

        print(f"\nState saved to: {state_file}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
