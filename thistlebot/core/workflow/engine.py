from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from ...agents.loader import AgentDefinition
from ...agents.memory import MemoryStore, new_memory_entry
from ..agent_runtime import run_tool_agent
from ..tools.registry import ToolRegistry, filter_by_allowlist_denylist
from ..workflow.fallback import CompositeFallbackResolver
from ..workflow.middleware import MiddlewareChain, StepMiddleware
from ..workflow.state import WorkflowState, append_event, ts
from ...llm.base import BaseLLMClient

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


class WorkflowEngine:
    def __init__(
        self,
        *,
        agent_definition: AgentDefinition,
        client: BaseLLMClient,
        registry: ToolRegistry,
        model: str,
        middlewares: list[StepMiddleware] | None = None,
        fallback_resolver: CompositeFallbackResolver | None = None,
    ) -> None:
        self.agent_definition = agent_definition
        self.client = client
        self.registry = registry
        self.model = model
        self.middleware_chain = MiddlewareChain(middlewares)
        self.fallback_resolver = fallback_resolver or CompositeFallbackResolver([])

    def execute(
        self,
        *,
        workflow_name: str,
        workflow_definition: dict[str, Any] | None,
        run_dir: Path,
        runtime_config: dict[str, Any],
        on_step: Callable[[str, str], None] | None = None,
        memory_store: MemoryStore | None = None,
    ) -> dict[str, Any]:
        factory = WorkflowFactory(self.agent_definition)
        workflow = workflow_definition if isinstance(workflow_definition, dict) else factory.build(workflow_name)
        steps = workflow["steps"]
        revision_loop = workflow.get("revision_loop", {}) if isinstance(workflow.get("revision_loop"), dict) else {}

        step_index_by_id = {str(step.get("id")): i for i, step in enumerate(steps)}

        state_path = run_dir / "run_state.json"
        events_path = run_dir / "events.jsonl"
        state = WorkflowState.load_or_create(state_path, workflow_name)

        # Resume from last unfinished index when the run is still active.
        idx = int(state.data.get("current_step_index", 0))

        while idx < len(steps):
            step = steps[idx]
            step_id = str(step.get("id") or "")
            if not step_id:
                raise ValueError("Each workflow step must define a non-empty 'id'")

            _notify(on_step, step_id, "started")
            append_event(events_path, {"event": "step_started", "step": step_id, "timestamp": ts()})

            output_text, step_events, artifact_name = self._execute_step(
                step=step,
                run_dir=run_dir,
                state=state.data,
                runtime_config=runtime_config,
            )

            _validate_step_required_tools(step=step, step_events=step_events)

            state.data.setdefault("step_attempts", {})
            state.data["step_attempts"][step_id] = int(state.data["step_attempts"].get(step_id, 0)) + 1

            artifact_key = str(step.get("artifact_key") or step_id)
            state.data.setdefault("artifacts", {})
            state.data["artifacts"][artifact_key] = artifact_name
            state.data.setdefault("steps", {})
            state.data["steps"][step_id] = "completed"
            state.data["last_step"] = step_id

            append_event(
                events_path,
                {
                    "event": "step_completed",
                    "step": step_id,
                    "artifact": artifact_name,
                    "timestamp": ts(),
                },
            )

            for event in step_events:
                event_payload = dict(event)
                event_payload.setdefault("step", step_id)
                event_payload.setdefault("timestamp", ts())
                append_event(events_path, event_payload)

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
                            "attempt": int(state.data["step_attempts"].get(step_id, 1)),
                        },
                    )
                )

            if step_id == str(revision_loop.get("verify_step", "")):
                pass_token = str(revision_loop.get("pass_token", "VERDICT: PASS"))
                max_revisions = int(revision_loop.get("max_revisions", 0))
                passed = pass_token in output_text
                state.data["quality_gate_passed"] = passed

                if not passed:
                    revision_count = int(state.data.get("revision_count", 0))
                    if revision_count >= max_revisions:
                        state.data["status"] = "failed_quality_gate"
                        state.data["error"] = (
                            f"Quality gate failed after {revision_count} revision attempts. "
                            f"Expected token '{pass_token}' in verify output."
                        )
                        state.save()
                        raise RuntimeError(str(state.data["error"]))

                    state.data["revision_count"] = revision_count + 1
                    state.data["runtime"]["revision_number"] = state.data["revision_count"] + 1
                    loop_step_id = str(revision_loop.get("edit_step", ""))
                    if loop_step_id not in step_index_by_id:
                        raise RuntimeError("Revision loop configured with unknown edit_step")
                    idx = step_index_by_id[loop_step_id]
                    state.data["current_step_index"] = idx
                    state.save()
                    _notify(on_step, step_id, "completed")
                    continue

            idx += 1
            state.data["current_step_index"] = idx
            state.save()
            _notify(on_step, step_id, "completed")

        state.data["status"] = "completed"
        state.data["completed_at"] = ts()
        state.save()

        if memory_store is not None:
            memory_store.record(
                new_memory_entry(
                    title=f"{workflow_name}:completed",
                    type="workflow_completion",
                    workflow=workflow_name,
                    run_id=run_dir.name,
                    tags=["workflow", workflow_name, "completed"],
                    summary=(
                        f"Run completed with {len(state.data.get('steps', {}))} steps and "
                        f"{int(state.data.get('revision_count', 0))} revisions."
                    ),
                    artifact_path=str(state.path.resolve()),
                    metadata={"status": "completed"},
                )
            )

        return {
            "status": state.data["status"],
            "steps": state.data.get("steps", {}),
            "artifacts": state.data.get("artifacts", {}),
            "revision_count": state.data.get("revision_count", 0),
            "run_state": str(state.path),
            "events_path": str(events_path),
        }

    def _execute_step(
        self,
        *,
        step: dict[str, Any],
        run_dir: Path,
        state: dict[str, Any],
        runtime_config: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], str]:
        step_type = str(step.get("type", "llm"))
        if step_type != "llm":
            raise ValueError(f"Unsupported step type: {step_type}")

        skill_name = str(step.get("skill") or "")
        if not skill_name:
            raise ValueError("LLM step requires a non-empty 'skill' field")

        skill_def = self.agent_definition.load_skill(skill_name)
        prompt_text = skill_def.instructions
        step_allowed_tools: list[str] = list(skill_def.allowed_tools)

        system_preamble = self.agent_definition.manifest.get("_system_prompt")
        system_sections: list[str] = []
        if isinstance(system_preamble, str) and system_preamble.strip():
            system_sections.append(system_preamble.strip())
        system_sections.append(prompt_text)

        resolved_inputs = _resolve_inputs(
            step.get("input", {}),
            runtime_config=runtime_config,
            run_dir=run_dir,
            state=state,
        )

        messages = [
            {"role": "system", "content": "\n\n".join(system_sections).strip()},
            {
                "role": "user",
                "content": _build_user_payload(step_id=str(step.get("id") or ""), inputs=resolved_inputs),
            },
        ]

        max_iterations = int(step.get("max_iterations", 8))
        output_text = ""
        events: list[dict[str, Any]] = []
        step_id = str(step.get("id") or "")

        step_registry = self.middleware_chain.apply(step=step, runtime_config=runtime_config, registry=self.registry)
        if step_allowed_tools:
            step_registry = filter_by_allowlist_denylist(step_registry, step_allowed_tools, None)
        else:
            # Skills that do not declare allowed tools should execute without tool access.
            step_registry = ToolRegistry()

        try:
            response = run_tool_agent(
                client=self.client,
                registry=step_registry,
                model=self.model,
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

        if _required_step_tools(step) and not _has_successful_required_call(step, events):
            resolved_fallback = self.fallback_resolver.resolve(
                step=step,
                step_id=step_id,
                resolved_inputs=resolved_inputs,
                registry=step_registry,
                output_text=output_text,
                events=events,
            )
            output_text, events = resolved_fallback.output_text, resolved_fallback.events

        artifact_name = _artifact_name_for_attempt(step, state)
        artifact_path = run_dir / artifact_name
        artifact_path.write_text(output_text, encoding="utf-8")

        return output_text, events, artifact_name


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
    middlewares: list[StepMiddleware] | None = None,
    fallback_resolver: CompositeFallbackResolver | None = None,
) -> dict[str, Any]:
    engine = WorkflowEngine(
        agent_definition=agent_definition,
        client=client,
        registry=registry,
        model=model,
        middlewares=middlewares,
        fallback_resolver=fallback_resolver,
    )
    return engine.execute(
        workflow_name=workflow_name,
        workflow_definition=workflow_definition,
        run_dir=run_dir,
        runtime_config=runtime_config,
        on_step=on_step,
        memory_store=memory_store,
    )


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


def _notify(on_step: Callable[[str, str], None] | None, step_id: str, status: str) -> None:
    if on_step is not None:
        on_step(step_id, status)


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


def _summarize_text(text: str, max_chars: int = 280) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}..."
