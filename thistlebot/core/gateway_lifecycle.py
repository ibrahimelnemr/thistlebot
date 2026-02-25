from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import DEVNULL, Popen
import subprocess
import sys
import time

import httpx


@dataclass
class GatewayHandle:
    gateway_url: str
    process: Popen | None = None

    @property
    def is_owned(self) -> bool:
        return self.process is not None

    def stop_if_owned(self, timeout: float = 5.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass


def _healthcheck(gateway_url: str, timeout: float = 1.0) -> bool:
    health_url = f"{gateway_url.rstrip('/')}/health"
    try:
        response = httpx.get(health_url, timeout=timeout)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _spawn_gateway_process(host: str, port: int) -> Popen:
    command = [
        sys.executable,
        "-m",
        "thistlebot",
        "gateway",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return Popen(command, stdout=DEVNULL, stderr=DEVNULL)


@contextmanager
def ensure_gateway(
    gateway_url: str,
    host: str,
    port: int,
    *,
    autostart: bool = True,
    start_timeout: float = 15.0,
    poll_interval: float = 0.2,
):
    if _healthcheck(gateway_url):
        handle = GatewayHandle(gateway_url=gateway_url, process=None)
        yield handle
        return

    if not autostart:
        raise RuntimeError(f"Gateway not reachable at {gateway_url}. Start it with 'thistlebot gateway'.")

    process = _spawn_gateway_process(host, port)
    handle = GatewayHandle(gateway_url=gateway_url, process=process)

    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        if _healthcheck(gateway_url):
            try:
                yield handle
            finally:
                handle.stop_if_owned()
            return

        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Gateway failed to start (exit code {return_code}).")

        time.sleep(poll_interval)

    handle.stop_if_owned()
    raise RuntimeError(f"Gateway did not become ready within {start_timeout:.1f}s at {gateway_url}.")
