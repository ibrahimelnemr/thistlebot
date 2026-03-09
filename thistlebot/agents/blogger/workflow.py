"""Blogger agent workflow runner backed by declarative workflow definition."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...agents.loader import load_agent_definition
from ...agents.memory import JsonFileMemoryStore, new_memory_entry
from ...agents.runner import execute_workflow
from ...core.tools.registry import build_tool_registry
from ...integrations.mcp.registry import build_mcp_registry
from ...llm.factory import build_llm_client, get_default_model
from ...storage.state import load_config
from .config import create_run_dir, find_resumable_run, get_run_dir, load_blogger_config, save_run_metadata
from .ideas import (
    refresh_idea_backlog,
    resolve_topic_from_backlog,
    update_selected_idea_outcome,
    write_selected_idea_artifact,
)


def run_publish_workflow(
    *,
    topic_override: str | None = None,
    status_override: str | None = None,
    on_step: None | Any = None,
) -> dict[str, Any]:
    """Execute the full declarative daily_publish workflow once.

    Args:
        topic_override: Override the configured topic for this run.
        status_override: Override the WordPress post status (draft/publish).
        on_step: Optional callback ``(step_name, status)`` called at step boundaries.

    Returns:
        A dict with run metadata including step results and events.
    """
    config = load_config()
    blogger_cfg = load_blogger_config()

    default_topic = str(blogger_cfg.get("topic") or "")
    post_status = status_override or blogger_cfg.get("post_status", "publish")
    site = blogger_cfg.get("site")
    if not site:
        raise RuntimeError(
            "No WordPress site configured. Set 'site' in "
            "~/.thistlebot/agents/blogger/config.json or 'blog' in "
            "~/.thistlebot/config.json under the wordpress section."
        )
    wf_cfg = blogger_cfg.get("workflow", {}) if isinstance(blogger_cfg.get("workflow"), dict) else {}

    # Build LLM client and tool registry.
    client = build_llm_client(config)
    model = get_default_model(config)
    mcp_registry = build_mcp_registry(config) if config.get("mcp", {}).get("enabled") else None
    registry = build_tool_registry(config, mcp_registry)

    resumable_run = find_resumable_run()
    run_dir = resumable_run or create_run_dir()
    resuming = resumable_run is not None
    memory_store = JsonFileMemoryStore("blogger")

    ideas_cfg = blogger_cfg.get("ideas", {}) if isinstance(blogger_cfg.get("ideas"), dict) else {}
    selected_idea: dict[str, Any] | None = None
    selected_idea_id: str | None = None
    ideas_refresh_result: dict[str, Any] | None = None

    if not resuming and bool(ideas_cfg.get("auto_refresh_before_publish", True)):
        try:
            ideas_refresh_result = refresh_idea_backlog(
                client=client,
                registry=registry,
                model=model,
                topic=topic_override or default_topic,
                count=int(ideas_cfg.get("refresh_count", 6)),
                query_count=int(ideas_cfg.get("query_count", 8)),
                max_iterations=int(ideas_cfg.get("max_iterations", 14)),
                prefer_web=bool(ideas_cfg.get("prefer_web", True)),
                min_refresh_interval_minutes=int(ideas_cfg.get("min_refresh_interval_minutes", 180)),
            )
        except Exception as exc:
            ideas_refresh_result = {
                "skipped": True,
                "reason": f"idea_refresh_error:{exc}",
                "created_count": 0,
            }

    if resuming:
        selected_path = Path(run_dir) / "selected_idea.json"
        if selected_path.exists() and not topic_override:
            try:
                selected_idea = json.loads(selected_path.read_text(encoding="utf-8"))
            except Exception:
                selected_idea = None
        if selected_idea is not None and str(selected_idea.get("title") or "").strip() and not topic_override:
            topic = str(selected_idea.get("title") or default_topic)
        else:
            topic, selected_idea = resolve_topic_from_backlog(
                explicit_topic=topic_override,
                default_topic=default_topic,
            )
    else:
        topic, selected_idea = resolve_topic_from_backlog(
            explicit_topic=topic_override,
            default_topic=default_topic,
        )
    if selected_idea is not None:
        selected_idea_id = str(selected_idea.get("id") or "") or None
        write_selected_idea_artifact(run_dir=Path(run_dir), idea=selected_idea)

    agent_definition = load_agent_definition("blogger")
    workflow_def = copy.deepcopy(agent_definition.load_workflow("daily_publish"))
    _apply_workflow_overrides(workflow_def, wf_cfg)

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "topic": topic,
        "site": site,
        "post_status": post_status,
        "model": model,
        "status": "running",
        "steps": {},
        "selected_idea_id": selected_idea_id,
        "ideas_refresh": ideas_refresh_result,
    }

    def _notify(step: str, status: str) -> None:
        if on_step is not None:
            on_step(step, status)

    try:
        exec_result = execute_workflow(
            agent_definition=agent_definition,
            workflow_name=str(workflow_def.get("id") or "daily_publish"),
            workflow_definition=workflow_def,
            run_dir=Path(run_dir),
            runtime_config={
                "topic": topic,
                "site": site,
                "post_status": post_status,
            },
            client=client,
            registry=registry,
            model=model,
            on_step=_notify,
            memory_store=memory_store,
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        _mark_run_state_failed(Path(run_dir), str(exc))
        save_run_metadata(Path(run_dir), result)
        update_selected_idea_outcome(
            idea_id=selected_idea_id,
            success=False,
            on_failure=str(ideas_cfg.get("failure_selected_action", "new")),
        )
        memory_store.record(
            new_memory_entry(
                title="daily_publish:failed",
                type="workflow_failure",
                workflow="daily_publish",
                run_id=Path(run_dir).name,
                tags=["workflow", "daily_publish", "failed"],
                summary=str(exc),
                artifact_path=str((Path(run_dir) / "run_state.json").resolve()),
                metadata={"status": "failed"},
            )
        )
        raise

    result["steps"] = exec_result.get("steps", {})
    result["status"] = str(exec_result.get("status", "unknown"))
    result["artifacts"] = exec_result.get("artifacts", {})
    result["revision_count"] = exec_result.get("revision_count", 0)
    result["run_state"] = exec_result.get("run_state")
    result["events_path"] = exec_result.get("events_path")

    update_selected_idea_outcome(
        idea_id=selected_idea_id,
        success=result["status"] == "completed",
        on_failure=str(ideas_cfg.get("failure_selected_action", "new")),
    )

    save_run_metadata(Path(run_dir), result)
    return result


def _apply_workflow_overrides(workflow_def: dict[str, Any], wf_cfg: dict[str, Any]) -> None:
    """Apply per-step iteration and loop overrides from blogger config."""
    steps = workflow_def.get("steps")
    if not isinstance(steps, list):
        return

    step_override_keys = {
        "research": "research_max_iterations",
        "draft": "draft_max_iterations",
        "edit": "edit_max_iterations",
        "verify": "verify_max_iterations",
        "publish": "publish_max_iterations",
    }

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "")
        cfg_key = step_override_keys.get(step_id)
        if cfg_key is None:
            continue
        if cfg_key in wf_cfg:
            try:
                step["max_iterations"] = int(wf_cfg[cfg_key])
            except (TypeError, ValueError):
                pass

    revision_loop = workflow_def.get("revision_loop")
    if not isinstance(revision_loop, dict):
        return

    if "max_revisions" in wf_cfg:
        try:
            revision_loop["max_revisions"] = int(wf_cfg["max_revisions"])
        except (TypeError, ValueError):
            pass

    if isinstance(wf_cfg.get("verify_pass_token"), str) and wf_cfg["verify_pass_token"].strip():
        revision_loop["pass_token"] = wf_cfg["verify_pass_token"].strip()


def retry_publish_from_run(
    *,
    run_id: str | None = None,
    status_override: str | None = None,
    on_step: None | Any = None,
) -> dict[str, Any]:
    """Re-attempt only the publish step using existing artifacts from a prior run."""
    config = load_config()
    blogger_cfg = load_blogger_config()

    run_dir = get_run_dir(run_id)
    if run_dir is None:
        raise RuntimeError("No blogger runs found for retry.")

    site = blogger_cfg.get("site")
    if not site:
        raise RuntimeError(
            "No WordPress site configured. Set 'site' in "
            "~/.thistlebot/agents/blogger/config.json or 'blog' in "
            "~/.thistlebot/config.json under the wordpress section."
        )

    wf_cfg = blogger_cfg.get("workflow", {}) if isinstance(blogger_cfg.get("workflow"), dict) else {}
    topic = str(blogger_cfg.get("topic") or "")
    post_status = status_override or blogger_cfg.get("post_status", "publish")

    client = build_llm_client(config)
    model = get_default_model(config)
    mcp_registry = build_mcp_registry(config) if config.get("mcp", {}).get("enabled") else None
    registry = build_tool_registry(config, mcp_registry)

    agent_definition = load_agent_definition("blogger")
    workflow_def = copy.deepcopy(agent_definition.load_workflow("daily_publish"))
    _apply_workflow_overrides(workflow_def, wf_cfg)

    publish_idx = _step_index(workflow_def, "publish")
    if publish_idx is None:
        raise RuntimeError("Publish step not found in workflow definition.")

    state_path = Path(run_dir) / "run_state.json"
    memory_store = JsonFileMemoryStore("blogger")
    state = _load_or_create_retry_state(state_path, workflow_name=str(workflow_def.get("id") or "daily_publish"))
    state["status"] = "running"
    state["current_step_index"] = publish_idx
    state.setdefault("runtime", {})
    state["runtime"].setdefault("revision_number", int(state.get("revision_count", 0)) + 1)
    state.setdefault("artifacts", {})

    draft_name = _resolve_existing_artifact(Path(run_dir), state.get("artifacts", {}), preferred_key="draft_current")
    if draft_name is None:
        raise RuntimeError("Could not find an existing draft artifact to publish.")
    state["artifacts"]["draft_current"] = draft_name

    qa_name = _resolve_existing_artifact(
        Path(run_dir), state.get("artifacts", {}), preferred_key="qa_report", allow_missing=True
    )
    if qa_name:
        state["artifacts"]["qa_report"] = qa_name

    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "topic": topic,
        "site": site,
        "post_status": post_status,
        "model": model,
        "status": "running",
        "steps": state.get("steps", {}),
    }

    def _notify(step: str, status: str) -> None:
        if on_step is not None:
            on_step(step, status)

    try:
        exec_result = execute_workflow(
            agent_definition=agent_definition,
            workflow_name=str(workflow_def.get("id") or "daily_publish"),
            workflow_definition=workflow_def,
            run_dir=Path(run_dir),
            runtime_config={
                "topic": topic,
                "site": site,
                "post_status": post_status,
            },
            client=client,
            registry=registry,
            model=model,
            on_step=_notify,
            memory_store=memory_store,
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        _mark_run_state_failed(Path(run_dir), str(exc))
        save_run_metadata(Path(run_dir), result)
        memory_store.record(
            new_memory_entry(
                title="daily_publish:retry_publish_failed",
                type="workflow_failure",
                workflow="daily_publish",
                run_id=Path(run_dir).name,
                tags=["workflow", "daily_publish", "retry", "failed"],
                summary=str(exc),
                artifact_path=str((Path(run_dir) / "run_state.json").resolve()),
                metadata={"status": "failed", "retry": True},
            )
        )
        raise

    result["steps"] = exec_result.get("steps", {})
    result["status"] = str(exec_result.get("status", "unknown"))
    result["artifacts"] = exec_result.get("artifacts", {})
    result["revision_count"] = exec_result.get("revision_count", 0)
    result["run_state"] = exec_result.get("run_state")
    result["events_path"] = exec_result.get("events_path")
    save_run_metadata(Path(run_dir), result)
    return result


def _step_index(workflow_def: dict[str, Any], step_id: str) -> int | None:
    steps = workflow_def.get("steps")
    if not isinstance(steps, list):
        return None
    for idx, step in enumerate(steps):
        if isinstance(step, dict) and str(step.get("id") or "") == step_id:
            return idx
    return None


def _load_or_create_retry_state(state_path: Path, workflow_name: str) -> dict[str, Any]:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
        if isinstance(state, dict):
            return state

    return {
        "workflow": workflow_name,
        "status": "running",
        "started_at": "",
        "current_step_index": 0,
        "steps": {},
        "step_attempts": {},
        "artifacts": {},
        "revision_count": 0,
        "runtime": {"revision_number": 1},
    }


def _resolve_existing_artifact(
    run_dir: Path,
    artifacts: dict[str, Any],
    *,
    preferred_key: str,
    allow_missing: bool = False,
) -> str | None:
    existing = artifacts.get(preferred_key)
    if isinstance(existing, str) and existing and (run_dir / existing).exists():
        return existing

    if preferred_key == "draft_current":
        candidates = sorted(run_dir.glob("draft_edit_v*.md"), reverse=True)
        candidates += sorted(run_dir.glob("draft_edit*.md"), reverse=True)
        candidates += sorted(run_dir.glob("draft_v*.md"), reverse=True)
        candidates += sorted(run_dir.glob("draft*.md"), reverse=True)
    elif preferred_key == "qa_report":
        candidates = sorted(run_dir.glob("qa_report_v*.md"), reverse=True)
        candidates += sorted(run_dir.glob("qa_report*.md"), reverse=True)
    else:
        candidates = []

    for candidate in candidates:
        if candidate.is_file():
            return candidate.name

    if allow_missing:
        return None
    return None


def _mark_run_state_failed(run_dir: Path, error: str) -> None:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(state, dict):
        return
    state["status"] = "failed"
    state["error"] = error
    state["failed_at"] = state.get("failed_at") or datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
