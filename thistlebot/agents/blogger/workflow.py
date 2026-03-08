"""Blogger agent workflow — sequential research -> draft -> publish."""

from __future__ import annotations

from typing import Any

from ...core.agent_runtime import run_tool_agent
from ...core.tools.registry import build_tool_registry
from ...integrations.mcp.registry import build_mcp_registry
from ...llm.factory import build_llm_client, get_default_model
from ...storage.state import load_config
from .config import create_run_dir, load_blogger_config, save_run_metadata
from .prompts import draft_messages, publish_messages, research_messages


def run_publish_workflow(
    *,
    topic_override: str | None = None,
    status_override: str | None = None,
    on_step: None | Any = None,
) -> dict[str, Any]:
    """Execute the full research -> draft -> publish workflow once.

    Args:
        topic_override: Override the configured topic for this run.
        status_override: Override the WordPress post status (draft/publish).
        on_step: Optional callback ``(step_name, status)`` called at step boundaries.

    Returns:
        A dict with run metadata including step results and events.
    """
    config = load_config()
    blogger_cfg = load_blogger_config()

    topic = topic_override or blogger_cfg["topic"]
    post_status = status_override or blogger_cfg.get("post_status", "publish")
    site = blogger_cfg.get("site")
    if not site:
        raise RuntimeError(
            "No WordPress site configured. Set 'site' in "
            "~/.thistlebot/agents/blogger/config.json or 'blog' in "
            "~/.thistlebot/config.json under the wordpress section."
        )
    wf_cfg = blogger_cfg.get("workflow", {})

    # Build LLM client and tool registry
    client = build_llm_client(config)
    model = get_default_model(config)
    mcp_registry = build_mcp_registry(config) if config.get("mcp", {}).get("enabled") else None
    registry = build_tool_registry(config, mcp_registry)

    run_dir = create_run_dir()

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "topic": topic,
        "site": site,
        "post_status": post_status,
        "model": model,
        "status": "running",
        "steps": {},
    }

    def _notify(step: str, status: str) -> None:
        if on_step is not None:
            on_step(step, status)

    # ---- Step 1: Research ----
    _notify("research", "started")
    research_output = run_tool_agent(
        client=client,
        registry=registry,
        model=model,
        messages=research_messages(topic),
        max_iterations=wf_cfg.get("research_max_iterations", 12),
    )
    (run_dir / "research.md").write_text(str(research_output), encoding="utf-8")
    result["steps"]["research"] = "completed"
    _notify("research", "completed")

    # ---- Step 2: Draft ----
    _notify("draft", "started")
    draft_output = run_tool_agent(
        client=client,
        registry=registry,
        model=model,
        messages=draft_messages(topic, str(research_output)),
        max_iterations=wf_cfg.get("draft_max_iterations", 10),
    )
    (run_dir / "draft.md").write_text(str(draft_output), encoding="utf-8")
    result["steps"]["draft"] = "completed"
    _notify("draft", "completed")

    # ---- Step 3: Edit + Publish ----
    _notify("publish", "started")
    publish_result = run_tool_agent(
        client=client,
        registry=registry,
        model=model,
        messages=publish_messages(str(draft_output), site, post_status),
        max_iterations=wf_cfg.get("publish_max_iterations", 8),
        return_events=True,
    )
    if isinstance(publish_result, tuple):
        publish_text, events = publish_result
    else:
        publish_text, events = str(publish_result), []

    (run_dir / "final.md").write_text(publish_text, encoding="utf-8")
    result["steps"]["publish"] = "completed"
    result["status"] = "completed"
    result["publish_events"] = events
    _notify("publish", "completed")

    save_run_metadata(run_dir, result)
    return result
