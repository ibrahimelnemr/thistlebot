from __future__ import annotations

import os
import time
from typing import Optional

import httpx

DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"


def start_device_flow(client_id: str, scope: str = "repo") -> dict:
    headers = {"Accept": "application/json"}
    response = httpx.post(
        DEVICE_CODE_URL,
        data={"client_id": client_id, "scope": scope},
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    return payload


def poll_for_token(client_id: str, device_code: str, interval: int) -> str:
    headers = {"Accept": "application/json"}
    while True:
        response = httpx.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("access_token"):
            return payload["access_token"]
        if payload.get("error") in {"authorization_pending", "slow_down"}:
            time.sleep(interval)
            continue
        raise RuntimeError(f"GitHub device flow failed: {payload.get('error')}")


def login_with_device_flow(client_id: Optional[str] = None) -> dict:
    client_id = client_id or os.getenv("THISTLEBOT_GITHUB_CLIENT_ID")
    if not client_id:
        raise RuntimeError("Missing GitHub client id. Set github.client_id or THISTLEBOT_GITHUB_CLIENT_ID.")
    device_payload = start_device_flow(client_id)
    device_payload["client_id"] = client_id
    return device_payload
