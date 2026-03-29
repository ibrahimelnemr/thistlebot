from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from ...core.workflow.fallback import FallbackResult

if TYPE_CHECKING:
    from ...core.tools.registry import ToolRegistry


class BlogDraftFallbackResolver:
    """Deterministic fallback for publish steps that miss required WordPress tool calls."""

    def resolve(
        self,
        *,
        step: dict[str, Any],
        step_id: str,
        resolved_inputs: dict[str, Any],
        registry: "ToolRegistry",
        output_text: str,
        events: list[dict[str, Any]],
    ) -> FallbackResult | None:
        required_tools = _required_step_tools(step)
        if not required_tools:
            return None

        if _has_successful_required_call(step, events):
            return None

        fallback_spec = step.get("fallback")
        if not isinstance(fallback_spec, dict):
            return None

        fallback_tool = str(fallback_spec.get("tool") or "").strip()
        if not fallback_tool:
            return None

        payload = _build_fallback_payload(step_id=step_id, resolved_inputs=resolved_inputs, fallback_spec=fallback_spec)
        if not payload:
            return None

        fallback_events: list[dict[str, Any]] = [
            {
                "event": "required_tool_fallback_attempt",
                "step": step_id,
                "reason": "no_successful_required_tool_call_from_model",
            }
        ]

        fallback_events.append({"event": "tool_call", "tool": fallback_tool, "args": payload, "fallback": True})
        result = registry.invoke(fallback_tool, payload)
        fallback_events.append(
            {
                "event": "tool_result",
                "tool": fallback_tool,
                "ok": result.ok,
                "error": result.error,
                "content": result.content,
                "truncated": result.truncated,
                "fallback": True,
            }
        )
        if result.ok:
            summary = _fallback_success_summary(step_id=step_id, result=result)
            return FallbackResult(output_text=summary, events=events + fallback_events)

        return FallbackResult(output_text=output_text, events=events + fallback_events)


def _required_step_tools(step: dict[str, Any]) -> list[str]:
    required_tools_raw = step.get("required_success_tools")
    if isinstance(required_tools_raw, list) and required_tools_raw:
        return [str(item) for item in required_tools_raw if str(item).strip()]
    return []


def _has_successful_required_call(step: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    required_tools = set(_required_step_tools(step))
    for event in events:
        if event.get("event") != "tool_result":
            continue
        if str(event.get("tool") or "") not in required_tools:
            continue
        if bool(event.get("ok")):
            return True
    return False


def _build_fallback_payload(*, step_id: str, resolved_inputs: dict[str, Any], fallback_spec: dict[str, Any]) -> dict[str, Any]:
    _ = step_id
    args_map = fallback_spec.get("args_from_inputs")
    if not isinstance(args_map, dict):
        return {}

    draft = str(resolved_inputs.get("draft") or "")
    parsed_title, parsed_body = _parse_draft_title_and_body(draft)
    payload: dict[str, Any] = {}
    for key, mapping in args_map.items():
        if isinstance(mapping, str) and mapping == "__derive_title__":
            if parsed_title:
                payload[key] = parsed_title
            continue
        if isinstance(mapping, str) and mapping == "__derive_body__":
            if parsed_body:
                payload[key] = parsed_body
            continue
        if isinstance(mapping, str) and mapping == "__derive_tags__":
            source_title = parsed_title or str(resolved_inputs.get("topic") or "")
            payload[key] = _tags_from_title(source_title or "AI News Update")
            continue
        if isinstance(mapping, str):
            value = resolved_inputs.get(mapping)
            if value is not None:
                payload[key] = value
            continue
        payload[key] = mapping

    required_fields = fallback_spec.get("required_fields")
    if isinstance(required_fields, list):
        for field_name in required_fields:
            name = str(field_name)
            if name and payload.get(name) in {None, ""}:
                return {}
    return payload


def _parse_draft_title_and_body(draft: str) -> tuple[str, str]:
    lines = draft.splitlines()
    if not lines:
        return "", ""
    title = lines[0].strip()
    if len(lines) >= 3:
        body = "\n".join(lines[2:]).strip()
    elif len(lines) >= 2:
        body = "\n".join(lines[1:]).strip()
    else:
        body = ""

    if not body:
        # Last resort: treat full draft as body and synthesize title.
        body = draft.strip()
        if not title:
            title = "AI News Update"

    return title, body


def _tags_from_title(title: str) -> list[str]:
    tokens = [tok for tok in re.findall(r"[a-zA-Z0-9]+", title.lower()) if len(tok) > 2]
    tags = ["ai", "news"]
    for token in tokens:
        if token in {"ai", "news", "latest", "update"}:
            continue
        if token not in tags:
            tags.append(token)
        if len(tags) >= 6:
            break
    return tags


def _fallback_success_summary(*, step_id: str, result: Any) -> str:
    payload = result.data if isinstance(result.data, dict) else {}
    inner = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    post_id = inner.get("ID") if isinstance(inner, dict) else None
    post_url = None
    if isinstance(inner, dict):
        post_url = inner.get("URL") or inner.get("url")
    return (
        f"STEP_STATUS: SUCCESS\n"
        f"STEP_ID: {step_id}\n"
        f"POST_ID: {post_id if post_id is not None else 'unknown'}\n"
        f"POST_URL: {post_url or 'unknown'}\n"
        "Summary: Completed using deterministic fallback after missing model tool call."
    )
