from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class HookContext:
    agent_name: str
    agent_definition: Any
    agent_config: dict[str, Any]
    hook_config: dict[str, Any]
    run_dir: Path | None
    client: Any
    registry: Any
    model: str
    result: dict[str, Any] | None = None


@dataclass
class HookResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


@runtime_checkable
class AgentHook(Protocol):
    hook_type: str

    def execute(self, context: HookContext) -> HookResult:
        ...


def _load_hooks_from_directory(directory: Path, module_ns: str) -> list[AgentHook]:
    hooks: list[AgentHook] = []
    if not directory.exists() or not directory.is_dir():
        return hooks

    for file_path in sorted(directory.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        mod_name = f"{module_ns}.{file_path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type):
                hook_type = getattr(attr, "hook_type", None)
                execute_fn = getattr(attr, "execute", None)
                if isinstance(hook_type, str) and callable(execute_fn):
                    hooks.append(attr())
    return hooks


def resolve_hooks(
    *,
    agent_name: str,
    hook_declarations: list[dict[str, Any]],
    agents_root: Path,
) -> list[tuple[dict[str, Any], AgentHook]]:
    local_dir = agents_root / agent_name / "hooks"
    shared_dir = agents_root / "hooks"

    local_hooks = {hook.hook_type: hook for hook in _load_hooks_from_directory(local_dir, f"thistlebot.agents.{agent_name}.hooks")}
    shared_hooks = {hook.hook_type: hook for hook in _load_hooks_from_directory(shared_dir, "thistlebot.agents.hooks")}

    resolved: list[tuple[dict[str, Any], AgentHook]] = []
    for decl in hook_declarations:
        hook_type = str(decl.get("type") or "").strip()
        if not hook_type:
            raise RuntimeError("Hook declaration missing type")
        hook = local_hooks.get(hook_type) or shared_hooks.get(hook_type)
        if hook is None:
            raise RuntimeError(f"Hook type '{hook_type}' not found for agent '{agent_name}'")
        resolved.append((decl, hook))
    return resolved


def run_hooks(
    *,
    phase: str,
    agent_name: str,
    agent_definition: Any,
    agent_config: dict[str, Any],
    client: Any,
    registry: Any,
    model: str,
    run_dir: Path | None,
    result: dict[str, Any] | None,
) -> list[HookResult]:
    hooks_raw = agent_definition.hooks().get(phase, [])
    if not isinstance(hooks_raw, list) or not hooks_raw:
        return []

    agents_root = agent_definition.root.parent
    declarations = [item for item in hooks_raw if isinstance(item, dict) and item.get("enabled", True)]
    resolved = resolve_hooks(
        agent_name=agent_name,
        hook_declarations=declarations,
        agents_root=agents_root,
    )

    outputs: list[HookResult] = []
    for declaration, hook in resolved:
        hook_cfg = declaration.get("config") if isinstance(declaration.get("config"), dict) else {}
        ctx = HookContext(
            agent_name=agent_name,
            agent_definition=agent_definition,
            agent_config=agent_config,
            hook_config=hook_cfg,
            run_dir=run_dir,
            client=client,
            registry=registry,
            model=model,
            result=result,
        )
        out = hook.execute(ctx)
        outputs.append(out)
        if not out.ok:
            raise RuntimeError(out.error or f"Hook failed: {hook.hook_type}")
    return outputs
