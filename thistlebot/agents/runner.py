from __future__ import annotations

import json
import os
import re
import signal
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.agent_runtime import run_tool_agent
from ..core.tools.base import ToolResult
from ..core.tools.registry import ToolRegistry
from ..llm.base import BaseLLMClient
from ..storage.paths import agent_log_path, agent_state_path
from .loader import AgentDefinition
from .memory import MemoryStore, new_memory_entry

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
    memory_store: MemoryStore | None = None,
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

        _validate_step_required_tools(step=step, step_events=step_events)

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

        if memory_store is not None:
            run_id = run_dir.name
            artifact_path = str((run_dir / artifact_name).resolve())
            memory_store.record(
                new_memory_entry(
                    title=f"{workflow_name}:{step_id}",
                    type="workflow_step",
                    workflow=workflow_name,
                    step=step_id,
                    run_id=run_id,
                    tags=["workflow", workflow_name, step_id],
                    summary=_summarize_text(output_text),
                    artifact_path=artifact_path,
                    metadata={
                        "artifact": artifact_name,
                        "attempt": int(state["step_attempts"].get(step_id, 1)),
                    },
                )
            )

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

    if memory_store is not None:
        memory_store.record(
            new_memory_entry(
                title=f"{workflow_name}:completed",
                type="workflow_completion",
                workflow=workflow_name,
                run_id=run_dir.name,
                tags=["workflow", workflow_name, "completed"],
                summary=(
                    f"Run completed with {len(state.get('steps', {}))} steps and "
                    f"{int(state.get('revision_count', 0))} revisions."
                ),
                artifact_path=str(state_path.resolve()),
                metadata={"status": "completed"},
            )
        )

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

    prompt_name = str(step.get("skill") or step.get("prompt") or "")
    if not prompt_name:
        raise ValueError("LLM step requires a 'skill' or 'prompt' field")

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
    output_text = ""
    events: list[dict[str, Any]] = []
    step_id = str(step.get("id") or "")
    step_registry = _StepToolRegistryProxy(
        base=registry,
        enforce_draft_mode=_should_enforce_draft_mode(runtime_config),
    )
    try:
        response = run_tool_agent(
            client=client,
            registry=step_registry,
            model=model,
            messages=messages,
            max_iterations=max_iterations,
            return_events=True,
        )

        if isinstance(response, tuple):
            output_text, events = response
        else:
            output_text, events = str(response), []
    except Exception as exc:
        if not _required_step_tools(step):
            raise
        events = [
            {
                "event": "required_tool_model_error",
                "step": step_id,
                "error": str(exc),
            }
        ]
        output_text = "Model step failed; attempting deterministic required-tool fallback."

    # Reliability fallback for steps that require successful tool calls.
    output_text, events = _maybe_required_tool_fallback(
        step=step,
        step_id=step_id,
        resolved_inputs=resolved_inputs,
        registry=step_registry,
        output_text=output_text,
        events=events,
    )

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


def _validate_step_required_tools(*, step: dict[str, Any], step_events: list[dict[str, Any]]) -> None:
    required_tools = _required_step_tools(step)
    if not required_tools:
        return

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
            f"Step '{step.get('id')}' did not call any required_success_tools: {required_tools}."
        )

    raise RuntimeError(
        f"Step '{step.get('id')}' called required tools, but none returned success."
    )


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


def _maybe_required_tool_fallback(
    *,
    step: dict[str, Any],
    step_id: str,
    resolved_inputs: dict[str, Any],
    registry: Any,
    output_text: str,
    events: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    required_tools = _required_step_tools(step)
    if not required_tools:
        return output_text, events

    if _has_successful_required_call(step, events):
        return output_text, events

    fallback_spec = step.get("fallback")
    if not isinstance(fallback_spec, dict):
        return output_text, events

    fallback_tool = str(fallback_spec.get("tool") or "").strip()
    if not fallback_tool:
        return output_text, events

    payload = _build_fallback_payload(step_id=step_id, resolved_inputs=resolved_inputs, fallback_spec=fallback_spec)
    if not payload:
        return output_text, events

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
        return summary, events + fallback_events

    return output_text, events + fallback_events


def _should_enforce_draft_mode(runtime_config: dict[str, Any]) -> bool:
    enforce_flag = runtime_config.get("enforce_draft_mode")
    if isinstance(enforce_flag, bool):
        if not enforce_flag:
            return False
    elif isinstance(enforce_flag, str) and enforce_flag.strip().lower() in {"false", "0", "no", "off"}:
        return False

    publish_mode = str(runtime_config.get("publish_mode") or "draft").strip().lower()
    return publish_mode != "publish"


class _StepToolRegistryProxy:
    def __init__(self, *, base: ToolRegistry, enforce_draft_mode: bool) -> None:
        self._base = base
        self._enforce_draft_mode = enforce_draft_mode
        self._successful_publish_count = 0

    def list_specs(self) -> list[Any]:
        return self._base.list_specs()

    def to_model_tools(self) -> list[dict[str, Any]]:
        return self._base.to_model_tools()

    def list_tool_names(self) -> list[str]:
        return self._base.list_tool_names()

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        if tool_name != "wordpress.create_post":
            return self._base.invoke(tool_name, payload)

        if self._successful_publish_count >= 1:
            return ToolResult(
                ok=False,
                content="",
                error="Publish guard: at most one successful wordpress.create_post call is allowed per step.",
            )

        call_payload = dict(payload or {})
        if self._enforce_draft_mode:
            call_payload["status"] = "draft"

        result = self._base.invoke(tool_name, call_payload)
        if result.ok:
            self._successful_publish_count += 1
        return result


def _build_fallback_payload(*, step_id: str, resolved_inputs: dict[str, Any], fallback_spec: dict[str, Any]) -> dict[str, Any]:
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


def _summarize_text(text: str, max_chars: int = 280) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."


class AgentDaemon:
    """Foreground daemon for scheduled agent workflow execution."""

    def __init__(
        self,
        *,
        agent_name: str,
        schedule_config: dict[str, Any],
        run_once: Callable[[], dict[str, Any]],
    ) -> None:
        self.agent_name = agent_name
        self.schedule_config = schedule_config
        self.run_once = run_once
        self.state_path = agent_state_path(agent_name)
        self.log_path = agent_log_path(agent_name)
        self.scheduler: Any = None

    def run_forever(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler

        trigger = _build_schedule_trigger(self.schedule_config)
        self.scheduler = BlockingScheduler(timezone=trigger.timezone)
        self.scheduler.add_job(self._run_job, trigger=trigger, id=f"{self.agent_name}-workflow", replace_existing=True)

        self._write_state(
            {
                "agent": self.agent_name,
                "pid": _pid(),
                "started_at": _ts(),
                "status": "running",
                "last_run_at": None,
                "next_run_at": None,
                "last_error": None,
                "log_path": str(self.log_path),
            }
        )
        self._update_next_run_at()

        def _shutdown_handler(signum: int, _frame: Any) -> None:
            state = self._load_state()
            state["status"] = "stopped"
            state["stopped_at"] = _ts()
            state["stop_signal"] = signum
            self._write_state(state)
            if self.scheduler is not None:
                with suppress(Exception):
                    self.scheduler.shutdown(wait=False)

        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)

        try:
            self.scheduler.start()
        finally:
            state = self._load_state()
            if state.get("status") == "running":
                state["status"] = "stopped"
                state["stopped_at"] = _ts()
                self._write_state(state)

    def _run_job(self) -> None:
        state = self._load_state()
        state["last_run_at"] = _ts()
        state["last_error"] = None
        state["status"] = "running"
        self._write_state(state)

        try:
            result = self.run_once()
            state["last_run_status"] = str(result.get("status", "unknown"))
            state["last_run_dir"] = result.get("run_dir")
        except Exception as exc:
            state["last_run_status"] = "failed"
            state["last_error"] = str(exc)
            self._write_state(state)
            self._update_next_run_at()
            return

        self._write_state(state)
        self._update_next_run_at()

    def _update_next_run_at(self) -> None:
        if self.scheduler is None:
            return
        jobs = self.scheduler.get_jobs()
        next_run_dt = None
        if jobs:
            next_run_dt = getattr(jobs[0], "next_run_time", None)
            if next_run_dt is None:
                next_run_dt = _estimate_next_fire_time(jobs[0])
        next_run = next_run_dt.isoformat() if next_run_dt else None
        state = self._load_state()
        state["next_run_at"] = next_run
        self._write_state(state)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def read_agent_state(agent_name: str) -> dict[str, Any]:
    state_path = agent_state_path(agent_name)
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def stop_agent_daemon(agent_name: str) -> bool:
    state = read_agent_state(agent_name)
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        agent_log_path(agent_name).parent.mkdir(parents=True, exist_ok=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not _pid_is_alive(pid):
                break
            time.sleep(0.1)
        if _pid_is_alive(pid):
            os.kill(pid, signal.SIGKILL)
        state["status"] = "stop_requested"
        state["stop_requested_at"] = _ts()
        state["next_run_at"] = None
        state_path = agent_state_path(agent_name)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except ProcessLookupError:
        return False
    except Exception:
        return False
    return True


def is_agent_daemon_running(agent_name: str) -> bool:
    state = read_agent_state(agent_name)
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True


def _build_schedule_trigger(schedule_config: dict[str, Any]) -> Any:
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    timezone_name = str(schedule_config.get("timezone") or "UTC")

    interval_seconds = schedule_config.get("interval_seconds")
    if isinstance(interval_seconds, int) and interval_seconds > 0:
        return IntervalTrigger(seconds=interval_seconds, timezone=timezone_name)

    interval_minutes = schedule_config.get("interval_minutes")
    if isinstance(interval_minutes, int) and interval_minutes > 0:
        return IntervalTrigger(minutes=interval_minutes, timezone=timezone_name)

    times_per_day = schedule_config.get("times_per_day")
    if isinstance(times_per_day, int) and times_per_day > 0:
        minutes = max(int(1440 / times_per_day), 1)
        return IntervalTrigger(minutes=minutes, timezone=timezone_name)

    cron_expr = str(schedule_config.get("cron") or "0 9 * * *")
    return CronTrigger.from_crontab(cron_expr, timezone=timezone_name)


def _pid() -> int:
    return int(os.getpid())


def _estimate_next_fire_time(job: Any) -> Any:
    trigger = getattr(job, "trigger", None)
    if trigger is None:
        return None

    now = datetime.now(timezone.utc)
    with suppress(Exception):
        return trigger.get_next_fire_time(None, now)
    return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True
