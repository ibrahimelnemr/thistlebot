from __future__ import annotations

import json
import os
import signal
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

from ..storage.paths import agent_log_path, agent_state_path


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


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()
