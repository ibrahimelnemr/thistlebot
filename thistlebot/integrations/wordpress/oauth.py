from __future__ import annotations

import base64
import hashlib
import http.server
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from typing import Any

import httpx

PUBLIC_API_BASE = "https://public-api.wordpress.com"
REGISTER_URL = f"{PUBLIC_API_BASE}/oauth2-1/register"
AUTHORIZE_URL = f"{PUBLIC_API_BASE}/oauth2-1/authorize"
TOKEN_URL = f"{PUBLIC_API_BASE}/oauth2-1/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/callback"


@dataclass
class OAuthCallbackResult:
    code: str | None = None
    state: str | None = None
    error: str | None = None


class OAuthCallbackServer:
    def __init__(self, redirect_uri: str) -> None:
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.scheme != "http":
            raise ValueError("redirect_uri must use http:// for local callback")
        if not parsed.hostname:
            raise ValueError("redirect_uri must include a host")
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("redirect_uri host must be localhost or 127.0.0.1")
        if not parsed.port:
            raise ValueError("redirect_uri must include a port")

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
                    b"<html><body><h3>WordPress authorization complete.</h3>"
                    b"<p>You can close this tab and return to the terminal.</p></body></html>"
                )

            def log_message(self, fmt: str, *args: object) -> None:
                return

        self._server = http.server.ThreadingHTTPServer((self._host, self._port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def wait(self, timeout_seconds: int) -> OAuthCallbackResult:
        if not self._event.wait(timeout_seconds):
            raise TimeoutError("Timed out waiting for WordPress OAuth callback")
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


def normalize_token_data(raw: dict[str, Any]) -> dict[str, Any]:
    token_data = dict(raw)
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        token_data["expires_at"] = int(time.time() + int(expires_in))
    return token_data


def token_expired(token_data: dict[str, Any], skew_seconds: int = 30) -> bool:
    expires_at = token_data.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return False
    return time.time() >= float(expires_at) - float(skew_seconds)


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
        raise RuntimeError("WordPress registration succeeded but no client_id was returned")
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
        raise RuntimeError("WordPress token exchange succeeded but no access_token was returned")
    return body


def refresh_access_token(client_id: str, refresh_token: str, timeout: float) -> dict[str, Any]:
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
        raise RuntimeError("WordPress token refresh succeeded but no access_token was returned")
    return body


def login_with_authorization_code_flow(
    client_id: str,
    redirect_uri: str,
    scope: str,
    timeout: float,
    callback_timeout: int,
    open_browser: bool = True,
) -> tuple[dict[str, Any], str]:
    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    expected_state = secrets.token_urlsafe(24)

    authorize_url = build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=expected_state,
        code_challenge=challenge,
    )

    callback_server = OAuthCallbackServer(redirect_uri)
    callback_server.start()
    if open_browser:
        webbrowser.open(authorize_url)

    try:
        callback_result = callback_server.wait(callback_timeout)
    finally:
        callback_server.stop()

    if callback_result.error:
        raise RuntimeError(f"WordPress authorization failed with error: {callback_result.error}")
    if not callback_result.code:
        raise RuntimeError("WordPress callback did not include an authorization code")
    if callback_result.state != expected_state:
        raise RuntimeError("WordPress callback state mismatch")

    token_raw = exchange_code_for_tokens(
        client_id=client_id,
        code=callback_result.code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
        timeout=timeout,
    )
    return normalize_token_data(token_raw), authorize_url