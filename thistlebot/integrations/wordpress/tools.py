from __future__ import annotations

from typing import Any

from .rest_client import WordPressRestClient
from ...core.tools.base import ToolResult


class WordPressTools:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def list_sites(self, payload: dict[str, Any]) -> ToolResult:
        try:
            client = self._wordpress_client()
            result = client.list_sites()
            return ToolResult(ok=True, content=_json_dump(result), data=result)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def list_posts(self, payload: dict[str, Any]) -> ToolResult:
        site = str(payload.get("site") or "").strip()
        if not site:
            return ToolResult(ok=False, content="", error="Missing required argument: site")
        number = _as_int(payload.get("number")) or 20
        status = payload.get("status")
        try:
            client = self._wordpress_client()
            result = client.list_posts(site, number=number, status=str(status) if isinstance(status, str) else None)
            return ToolResult(ok=True, content=_json_dump(result), data=result)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def create_post(self, payload: dict[str, Any]) -> ToolResult:
        site = str(payload.get("site") or "").strip()
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "").strip()
        if not site:
            return ToolResult(ok=False, content="", error="Missing required argument: site")
        if not title:
            return ToolResult(ok=False, content="", error="Missing required argument: title")
        if not content:
            return ToolResult(ok=False, content="", error="Missing required argument: content")

        status = str(payload.get("status") or "draft")
        tags = payload.get("tags")
        categories = payload.get("categories")
        try:
            client = self._wordpress_client()
            result = client.create_post(
                site,
                title=title,
                content=content,
                status=status,
                tags=tags,
                categories=categories,
            )
            return ToolResult(ok=True, content=_json_dump(result), data=result)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def update_post(self, payload: dict[str, Any]) -> ToolResult:
        site = str(payload.get("site") or "").strip()
        post_id = _as_int(payload.get("post_id"))
        if not site:
            return ToolResult(ok=False, content="", error="Missing required argument: site")
        if post_id is None:
            return ToolResult(ok=False, content="", error="Missing required argument: post_id")

        title = payload.get("title")
        content = payload.get("content")
        status = payload.get("status")
        try:
            client = self._wordpress_client()
            result = client.update_post(
                site,
                post_id,
                title=str(title) if isinstance(title, str) else None,
                content=str(content) if isinstance(content, str) else None,
                status=str(status) if isinstance(status, str) else None,
            )
            return ToolResult(ok=True, content=_json_dump(result), data=result)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def get_post(self, payload: dict[str, Any]) -> ToolResult:
        site = str(payload.get("site") or "").strip()
        post_id = _as_int(payload.get("post_id"))
        if not site:
            return ToolResult(ok=False, content="", error="Missing required argument: site")
        if post_id is None:
            return ToolResult(ok=False, content="", error="Missing required argument: post_id")
        try:
            client = self._wordpress_client()
            result = client.get_post(site, post_id)
            return ToolResult(ok=True, content=_json_dump(result), data=result)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def _wordpress_client(self) -> WordPressRestClient:
        wp_cfg = _wordpress_config(self.config)
        token = wp_cfg.get("token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("WordPress token missing. Run 'thistlebot wordpress login'.")
        timeout = wp_cfg.get("timeout_seconds")
        timeout_value = float(timeout) if isinstance(timeout, (int, float)) else 30.0
        return WordPressRestClient(access_token=token, timeout_seconds=timeout_value)


def _wordpress_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}

    integrations_cfg = config.get("integrations")
    if isinstance(integrations_cfg, dict):
        wp_cfg = integrations_cfg.get("wordpress")
        if isinstance(wp_cfg, dict):
            return wp_cfg

    return {}


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
