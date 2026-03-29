from .fallback import CompositeFallbackResolver, FallbackResolver, FallbackResult
from .middleware import MiddlewareChain, StepMiddleware
from .state import WorkflowState

__all__ = [
    "WorkflowEngine",
    "execute_workflow",
    "CompositeFallbackResolver",
    "FallbackResolver",
    "FallbackResult",
    "MiddlewareChain",
    "StepMiddleware",
    "WorkflowState",
]


def __getattr__(name: str):
    if name in {"WorkflowEngine", "execute_workflow"}:
        from .engine import WorkflowEngine, execute_workflow

        if name == "WorkflowEngine":
            return WorkflowEngine
        return execute_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
