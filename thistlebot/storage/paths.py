from pathlib import Path


THISTLEBOT_DIR_NAME = ".thistlebot"


def base_dir() -> Path:
    return Path.home() / THISTLEBOT_DIR_NAME


def config_path() -> Path:
    return base_dir() / "config.json"


def prompts_dir() -> Path:
    return base_dir() / "prompts"


def memory_dir() -> Path:
    return base_dir() / "memory"


def sessions_dir() -> Path:
    return memory_dir() / "sessions"


def logs_dir() -> Path:
    return base_dir() / "logs"


def workspace_dir() -> Path:
    return base_dir() / "workspace"


def projects_dir() -> Path:
    return workspace_dir() / "projects"


def ensure_base_dirs() -> None:
    for path in [
        base_dir(),
        prompts_dir(),
        memory_dir(),
        sessions_dir(),
        logs_dir(),
        workspace_dir(),
        projects_dir(),
    ]:
        path.mkdir(parents=True, exist_ok=True)
