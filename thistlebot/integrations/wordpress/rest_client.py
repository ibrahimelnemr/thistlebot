from __future__ import annotations

from typing import Any

import httpx

REST_BASE_URL = "https://public-api.wordpress.com/rest/v1.1"


class WordPressRestClient:
    def __init__(self, access_token: str, timeout_seconds: float = 30.0) -> None:
        if not access_token:
            raise ValueError("WordPress REST access token is required")
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds

    def list_sites(self) -> dict[str, Any]:
        return self._request("GET", "/me/sites")

    def get_site(self, site: str) -> dict[str, Any]:
        return self._request("GET", f"/sites/{site}")

    def token_info(self, client_id: str) -> dict[str, Any]:
        url = "https://public-api.wordpress.com/oauth2/token-info"
        params = {"client_id": client_id, "token": self._access_token}
        try:
            response = httpx.get(url, params=params, timeout=self._timeout_seconds)
        except Exception as exc:
            raise RuntimeError(f"WordPress token-info request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    msg = payload.get("error") or payload.get("message")
                    if msg:
                        detail = str(msg)
            except Exception:
                pass
            raise RuntimeError(f"WordPress token-info failed ({response.status_code}): {detail}")
        body = response.json() if response.content else {}
        return body if isinstance(body, dict) else {"data": body}

    def list_posts(self, site: str, *, number: int = 20, status: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"number": max(1, min(100, int(number)))}
        if status:
            params["status"] = status
        return self._request("GET", f"/sites/{site}/posts", params=params)

    def get_post(self, site: str, post_id: int) -> dict[str, Any]:
        return self._request("GET", f"/sites/{site}/posts/{int(post_id)}")

    def create_post(
        self,
        site: str,
        *,
        title: str,
        content: str,
        status: str = "draft",
        tags: list[str] | str | None = None,
        categories: list[str] | str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": title,
            "content": content,
            "status": status,
        }
        if tags:
            payload["tags"] = tags
        if categories:
            payload["categories"] = categories
        return self._request("POST", f"/sites/{site}/posts/new", data=payload)

    def update_post(
        self,
        site: str,
        post_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if content is not None:
            payload["content"] = content
        if status is not None:
            payload["status"] = status
        if not payload:
            raise ValueError("At least one of title/content/status must be provided")
        return self._request("POST", f"/sites/{site}/posts/{int(post_id)}", data=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{REST_BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

        try:
            response = httpx.request(
                method,
                url,
                headers=headers,
                params=params,
                data=data,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"WordPress REST request failed: {exc}") from exc

        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"error": response.text}
            message = payload.get("message") if isinstance(payload, dict) else str(payload)
            code = payload.get("error") if isinstance(payload, dict) else None
            detail = f"{response.status_code}"
            if code:
                detail += f" {code}"
            if message:
                detail += f": {message}"
            raise RuntimeError(f"WordPress REST API error {detail}")

        body = response.json() if response.content else {}
        if isinstance(body, dict):
            return body
        return {"data": body}
