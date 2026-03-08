from __future__ import annotations

import secrets
import time
import urllib.parse
import webbrowser
from typing import Any

import httpx

from .oauth import OAuthCallbackServer

PUBLIC_API_BASE = "https://public-api.wordpress.com"
AUTHORIZE_URL = f"{PUBLIC_API_BASE}/oauth2/authorize"
TOKEN_URL = f"{PUBLIC_API_BASE}/oauth2/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8766/callback"


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


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    blog: str | None = None,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
    }
    if blog:
        params["blog"] = blog
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_tokens(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    timeout: float,
) -> dict[str, Any]:
    form = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    response = httpx.post(
        TOKEN_URL,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        detail = response.text
        try:
            payload = response.json()
            if isinstance(payload, dict):
                msg = payload.get("error_description") or payload.get("error") or payload.get("message")
                if msg:
                    detail = str(msg)
        except Exception:
            pass
        raise RuntimeError(f"WordPress REST token exchange failed ({response.status_code}): {detail}")
    body = response.json()
    if not isinstance(body, dict) or not body.get("access_token"):
        raise RuntimeError("WordPress REST token exchange succeeded but no access_token was returned")
    return body


def login_with_authorization_code_flow(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    scope: str,
    timeout: float,
    callback_timeout: int,
    open_browser: bool = True,
    blog: str | None = None,
) -> tuple[dict[str, Any], str]:
    expected_state = secrets.token_urlsafe(24)
    authorize_url = build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=expected_state,
        blog=blog,
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
        raise RuntimeError(f"WordPress REST authorization failed with error: {callback_result.error}")
    if not callback_result.code:
        raise RuntimeError("WordPress REST callback did not include an authorization code")
    if callback_result.state != expected_state:
        raise RuntimeError("WordPress REST callback state mismatch")

    token_raw = exchange_code_for_tokens(
        client_id=client_id,
        client_secret=client_secret,
        code=callback_result.code,
        redirect_uri=redirect_uri,
        timeout=timeout,
    )
    return normalize_token_data(token_raw), authorize_url
