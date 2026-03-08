from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.agent_runtime import run_tool_agent
from ..core.tools.registry import ToolRegistry
from ..llm.base import BaseLLMClient
from .loader import AgentDefinition

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}")


class WorkflowFactory:
    """Builds and executes declarative workflows from agent definitions."""

    def __init__(self, agent_definition: AgentDefinition) -> None:
        self.agent_definition = agent_definition

    def build(self, workflow_name: str) -> dict[str, Any]:
        workflow = self.agent_definition.load_workflow(workflow_name)
        if not isinstance(workflow.get("steps"), list):
            raise ValueError(f"Workflow '{workflow_name}' has no valid 'steps' list")
        return workflow


def execute_workflow(
    *,
    agent_definition: AgentDefinition,
    workflow_name: str,
    workflow_definition: dict[str, Any] | None,
    run_dir: Path,
    runtime_config: dict[str, Any],
    client: BaseLLMClient,
    registry: ToolRegistry,
    model: str,
    on_step: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    factory = WorkflowFactory(agent_definition)
    workflow = workflow_definition if isinstance(workflow_definition, dict) else factory.build(workflow_name)
    steps = workflow["steps"]
    revision_loop = workflow.get("revision_loop", {}) if isinstance(workflow.get("revision_loop"), dict) else {}

    step_index_by_id = {str(step.get("id")): i for i, step in enumerate(steps)}

    state_path = run_dir / "run_state.json"
    events_path = run_dir / "events.jsonl"
    state = _load_or_create_state(state_path, workflow_name)

    # Resume from last unfinished index when the run is still active.
    idx = int(state.get("current_step_index", 0))

    while idx < len(steps):
        step = steps[idx]
        step_id = str(step.get("id") or "")
        if not step_id:
            raise ValueError("Each workflow step must define a non-empty 'id'")

        _notify(on_step, step_id, "started")
        _append_event(events_path, {"event": "step_started", "step": step_id, "timestamp": _ts()})

        output_text, step_events, artifact_name = _execute_step(
            step=step,
            run_dir=run_dir,
            state=state,
            runtime_config=runtime_config,
            agent_definition=agent_definition,
            client=client,
            registry=registry,
            model=model,
        )

        _validate_publish_step(step=step, step_events=step_events)

        state.setdefault("step_attempts", {})
        state["step_attempts"][step_id] = int(state["step_attempts"].get(step_id, 0)) + 1

        artifact_key = str(step.get("artifact_key") or step_id)
        state.setdefault("artifacts", {})
        state["artifacts"][artifact_key] = artifact_name
        state.setdefault("steps", {})
        state["steps"][step_id] = "completed"
        state["last_step"] = step_id

        _append_event(
            events_path,
            {
                "event": "step_completed",
                "step": step_id,
                "artifact": artifact_name,
                "timestamp": _ts(),
            },
        )

        for event in step_events:
            event_payload = dict(event)
            event_payload.setdefault("step", step_id)
            event_payload.setdefault("timestamp", _ts())
            _append_event(events_path, event_payload)

        if step_id == str(revision_loop.get("verify_step", "")):
            pass_token = str(revision_loop.get("pass_token", "VERDICT: PASS"))
            max_revisions = int(revision_loop.get("max_revisions", 0))
            passed = pass_token in output_text
            state["quality_gate_passed"] = passed

            if not passed:
                revision_count = int(state.get("revision_count", 0))
                if revision_count >= max_revisions:
                    state["status"] = "failed_quality_gate"
                    state["error"] = (
                        f"Quality gate failed after {revision_count} revision attempts. "
                        f"Expected token '{pass_token}' in verify output."
                    )
                    _save_state(state_path, state)
                    raise RuntimeError(str(state["error"]))

                state["revision_count"] = revision_count + 1
                state["runtime"]["revision_number"] = state["revision_count"] + 1
                loop_step_id = str(revision_loop.get("edit_step", ""))
                if loop_step_id not in step_index_by_id:
                    raise RuntimeError("Revision loop configured with unknown edit_step")
                idx = step_index_by_id[loop_step_id]
                state["current_step_index"] = idx
                _save_state(state_path, state)
                _notify(on_step, step_id, "completed")
                continue

        idx += 1
        state["current_step_index"] = idx
        _save_state(state_path, state)
        _notify(on_step, step_id, "completed")

    state["status"] = "completed"
    state["completed_at"] = _ts()
    _save_state(state_path, state)

    return {
        "status": state["status"],
        "steps": state.get("steps", {}),
        "artifacts": state.get("artifacts", {}),
        "revision_count": state.get("revision_count", 0),
        "run_state": str(state_path),
        "events_path": str(events_path),
    }


def _execute_step(
    *,
    step: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
    runtime_config: dict[str, Any],
    agent_definition: AgentDefinition,
    client: BaseLLMClient,
    registry: ToolRegistry,
    model: str,
) -> tuple[str, list[dict[str, Any]], str]:
    step_type = str(step.get("type", "llm"))
    if step_type != "llm":
        raise ValueError(f"Unsupported step type: {step_type}")

    prompt_name = str(step.get("prompt") or "")
    if not prompt_name:
        raise ValueError("LLM step requires a prompt")

    prompt_text = agent_definition.load_prompt(prompt_name)
    resolved_inputs = _resolve_inputs(
        step.get("input", {}),
        runtime_config=runtime_config,
        run_dir=run_dir,
        state=state,
    )

    messages = [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": _build_user_payload(step_id=str(step.get("id") or ""), inputs=resolved_inputs),
        },
    ]

    max_iterations = int(step.get("max_iterations", 8))
    response = run_tool_agent(
        client=client,
        registry=registry,
        model=model,
        messages=messages,
        max_iterations=max_iterations,
        return_events=True,
    )

    if isinstance(response, tuple):
        output_text, events = response
    else:
        output_text, events = str(response), []

    artifact_name = _artifact_name_for_attempt(step, state)
    artifact_path = run_dir / artifact_name
    artifact_path.write_text(output_text, encoding="utf-8")

    return output_text, events, artifact_name


def _build_user_payload(*, step_id: str, inputs: dict[str, Any]) -> str:
    lines = [f"Workflow step: {step_id}", "", "Inputs:"]
    for key, value in inputs.items():
        lines.append(f"[{key}]")
        lines.append(str(value))
        lines.append("")
    return "\n".join(lines).strip()


def _resolve_inputs(
    raw_inputs: Any,
    *,
    runtime_config: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_inputs, dict):
        return {}

    resolved: dict[str, Any] = {}
    for key, value in raw_inputs.items():
        if isinstance(value, str):
            resolved[key] = _resolve_template(value, runtime_config=runtime_config, run_dir=run_dir, state=state)
        else:
            resolved[key] = value
    return resolved


def _resolve_template(template: str, *, runtime_config: dict[str, Any], run_dir: Path, state: dict[str, Any]) -> str:
    def _replacement(match: re.Match[str]) -> str:
        token = match.group(1)
        return _resolve_token(token, runtime_config=runtime_config, run_dir=run_dir, state=state)

    return _TEMPLATE_RE.sub(_replacement, template)


def _resolve_token(token: str, *, runtime_config: dict[str, Any], run_dir: Path, state: dict[str, Any]) -> str:
    parts = token.split(".")
    if len(parts) < 2:
        return ""

    scope = parts[0]
    key = ".".join(parts[1:])

    if scope == "config":
        return str(runtime_config.get(key, ""))

    if scope == "runtime":
        runtime_state = state.get("runtime", {}) if isinstance(state.get("runtime"), dict) else {}
        return str(runtime_state.get(key, ""))

    if scope == "artifacts":
        artifacts = state.get("artifacts", {}) if isinstance(state.get("artifacts"), dict) else {}
        artifact_name = artifacts.get(key)
        if not isinstance(artifact_name, str) or not artifact_name:
            return ""
        artifact_path = run_dir / artifact_name
        if not artifact_path.exists():
            return ""
        return artifact_path.read_text(encoding="utf-8")

    return ""


def _artifact_name_for_attempt(step: dict[str, Any], state: dict[str, Any]) -> str:
    base = str(step.get("output_artifact") or f"{step.get('id', 'step')}.md")
    step_id = str(step.get("id") or "")

    attempts = state.get("step_attempts", {}) if isinstance(state.get("step_attempts"), dict) else {}
    attempt_no = int(attempts.get(step_id, 0)) + 1

    if attempt_no <= 1:
        return base

    path = Path(base)
    suffix = path.suffix
    stem = path.stem
    if suffix:
        return f"{stem}_v{attempt_no}{suffix}"
    return f"{base}_v{attempt_no}"


def _load_or_create_state(state_path: Path, workflow_name: str) -> dict[str, Any]:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(state, dict) and state.get("status") in {"running", "failed_quality_gate"}:
            state.setdefault("runtime", {})
            state["runtime"].setdefault("revision_number", int(state.get("revision_count", 0)) + 1)
            return state

    state = {
        "workflow": workflow_name,
        "status": "running",
        "started_at": _ts(),
        "current_step_index": 0,
        "steps": {},
        "step_attempts": {},
        "artifacts": {},
        "revision_count": 0,
        "runtime": {"revision_number": 1},
    }
    _save_state(state_path, state)
    return state


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _append_event(events_path: Path, payload: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _notify(on_step: Callable[[str, str], None] | None, step_id: str, status: str) -> None:
    if on_step is not None:
        on_step(step_id, status)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_publish_step(*, step: dict[str, Any], step_events: list[dict[str, Any]]) -> None:
    """Require publish steps to actually call and succeed on WordPress create post tools."""
    step_id = str(step.get("id") or "")
    if step_id != "publish":
        return

    required_tools_raw = step.get("required_success_tools")
    if isinstance(required_tools_raw, list) and required_tools_raw:
        required_tools = [str(item) for item in required_tools_raw if str(item).strip()]
    else:
        required_tools = ["wordpress.create_post", "wordpress.rest.create_post"]

    seen_required_call = False
    for event in step_events:
        if event.get("event") != "tool_result":
            continue
        tool = str(event.get("tool") or "")
        if tool not in required_tools:
            continue
        seen_required_call = True
        if bool(event.get("ok")):
            return

    if not seen_required_call:
        raise RuntimeError(
            "Publish step did not call a required WordPress create-post tool. "
            "No post was published."
        )

    raise RuntimeError(
        "Publish step called WordPress create-post tool, but no successful result was recorded."
    )
