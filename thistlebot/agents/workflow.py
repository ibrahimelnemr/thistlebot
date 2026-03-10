from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from .config import (
    create_run_dir,
    find_resumable_run,
    get_run_dir,
    load_agent_config,
    save_run_metadata,
)
from .hooks.base import run_hooks
from .loader import load_agent_definition
from .memory import JsonFileMemoryStore
from .runner import execute_workflow
from ..core.tools.registry import build_tool_registry
from ..integrations.mcp.registry import build_mcp_registry
from ..llm.factory import build_llm_client, get_default_model
from ..storage.state import load_config


def _apply_workflow_overrides(workflow_def: dict[str, Any], overrides: dict[str, Any]) -> None:
    steps = workflow_def.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id") or "")
            if not step_id:
                continue
            max_key = f"{step_id}_max_iterations"
            if max_key in overrides:
                try:
                    step["max_iterations"] = int(overrides[max_key])
                except (TypeError, ValueError):
                    pass

    revision_loop = workflow_def.get("revision_loop")
    if not isinstance(revision_loop, dict):
        return

    if "max_revisions" in overrides:
        try:
            revision_loop["max_revisions"] = int(overrides["max_revisions"])
        except (TypeError, ValueError):
            pass

    if isinstance(overrides.get("verify_pass_token"), str) and overrides["verify_pass_token"].strip():
        revision_loop["pass_token"] = overrides["verify_pass_token"].strip()


def run_agent_workflow(
    agent_name: str,
    *,
    workflow_name: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    on_step: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    config = load_config()
    agent_definition = load_agent_definition(agent_name)
    runtime_config = load_agent_config(agent_name, agent_definition, config_overrides=config_overrides)

    client = build_llm_client(config)
    model = get_default_model(config)
    mcp_registry = build_mcp_registry(config) if config.get("mcp", {}).get("enabled") else None
    registry = build_tool_registry(config, mcp_registry, tool_spec=agent_definition.tools())

    resolved_workflow_name = workflow_name or agent_definition.default_workflow_name()
    workflow_def = copy.deepcopy(agent_definition.load_workflow(resolved_workflow_name))
    _apply_workflow_overrides(workflow_def, agent_definition.workflow_overrides())

    resumable_run = find_resumable_run(agent_name)
    run_dir = resumable_run or create_run_dir(agent_name)
    memory_store = JsonFileMemoryStore(agent_name)

    run_hooks(
        phase="pre_run",
        agent_name=agent_name,
        agent_definition=agent_definition,
        agent_config=runtime_config,
        client=client,
        registry=registry,
        model=model,
        run_dir=Path(run_dir),
        result=None,
    )

    topic_hook_results = run_hooks(
        phase="pre_topic_resolve",
        agent_name=agent_name,
        agent_definition=agent_definition,
        agent_config=runtime_config,
        client=client,
        registry=registry,
        model=model,
        run_dir=Path(run_dir),
        result=None,
    )
    for hook_result in topic_hook_results:
        if isinstance(hook_result.data, dict) and isinstance(hook_result.data.get("topic"), str):
            runtime_config["topic"] = hook_result.data["topic"].strip()

    result: dict[str, Any] = {
        "agent": agent_name,
        "workflow": resolved_workflow_name,
        "run_dir": str(run_dir),
        "status": "running",
        "model": model,
        "config": {k: v for k, v in runtime_config.items() if k not in {"token", "client_secret"}},
    }

    try:
        exec_result = execute_workflow(
            agent_definition=agent_definition,
            workflow_name=str(workflow_def.get("id") or resolved_workflow_name),
            workflow_definition=workflow_def,
            run_dir=Path(run_dir),
            runtime_config=runtime_config,
            client=client,
            registry=registry,
            model=model,
            on_step=on_step,
            memory_store=memory_store,
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        _mark_run_state_failed(Path(run_dir), str(exc))
        run_hooks(
            phase="post_run",
            agent_name=agent_name,
            agent_definition=agent_definition,
            agent_config=runtime_config,
            client=client,
            registry=registry,
            model=model,
            run_dir=Path(run_dir),
            result=result,
        )
        save_run_metadata(Path(run_dir), result)
        raise

    result["status"] = str(exec_result.get("status", "unknown"))
    result["steps"] = exec_result.get("steps", {})
    result["artifacts"] = exec_result.get("artifacts", {})
    result["revision_count"] = exec_result.get("revision_count", 0)
    result["run_state"] = exec_result.get("run_state")
    result["events_path"] = exec_result.get("events_path")

    run_hooks(
        phase="post_run",
        agent_name=agent_name,
        agent_definition=agent_definition,
        agent_config=runtime_config,
        client=client,
        registry=registry,
        model=model,
        run_dir=Path(run_dir),
        result=result,
    )
    save_run_metadata(Path(run_dir), result)
    return result


def retry_step_from_run(
    agent_name: str,
    step_id: str,
    *,
    run_id: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    on_step: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    run_dir = get_run_dir(agent_name, run_id=run_id)
    if run_dir is None:
        raise RuntimeError(f"No runs found for agent '{agent_name}'.")

    config = load_config()
    agent_definition = load_agent_definition(agent_name)
    runtime_config = load_agent_config(agent_name, agent_definition, config_overrides=config_overrides)

    client = build_llm_client(config)
    model = get_default_model(config)
    mcp_registry = build_mcp_registry(config) if config.get("mcp", {}).get("enabled") else None
    registry = build_tool_registry(config, mcp_registry, tool_spec=agent_definition.tools())

    workflow_name = agent_definition.default_workflow_name()
    workflow_def = copy.deepcopy(agent_definition.load_workflow(workflow_name))
    _apply_workflow_overrides(workflow_def, agent_definition.workflow_overrides())

    step_index = _step_index(workflow_def, step_id)
    if step_index is None:
        raise RuntimeError(f"Step '{step_id}' not found in workflow '{workflow_name}'.")

    state_path = Path(run_dir) / "run_state.json"
    state = _load_or_create_retry_state(state_path, workflow_name=str(workflow_def.get("id") or workflow_name))
    state["status"] = "running"
    state["current_step_index"] = step_index
    state.setdefault("runtime", {})
    state["runtime"].setdefault("revision_number", int(state.get("revision_count", 0)) + 1)
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    memory_store = JsonFileMemoryStore(agent_name)
    exec_result = execute_workflow(
        agent_definition=agent_definition,
        workflow_name=str(workflow_def.get("id") or workflow_name),
        workflow_definition=workflow_def,
        run_dir=Path(run_dir),
        runtime_config=runtime_config,
        client=client,
        registry=registry,
        model=model,
        on_step=on_step,
        memory_store=memory_store,
    )

    result = {
        "agent": agent_name,
        "workflow": workflow_name,
        "run_dir": str(run_dir),
        "status": str(exec_result.get("status", "unknown")),
        "steps": exec_result.get("steps", {}),
        "artifacts": exec_result.get("artifacts", {}),
        "revision_count": exec_result.get("revision_count", 0),
        "run_state": exec_result.get("run_state"),
        "events_path": exec_result.get("events_path"),
    }
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
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
