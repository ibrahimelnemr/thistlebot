from __future__ import annotations

import uvicorn

from ..api.app import create_app
from ..storage.state import load_config
from ..utils.logging import configure_logging
from ..storage.paths import logs_dir


def run_gateway(host: str | None = None, port: int | None = None) -> None:
    config = load_config()
    gateway_cfg = config.get("gateway", {})
    server_host = host or gateway_cfg.get("host", "127.0.0.1")
    server_port = port or gateway_cfg.get("port", 7788)

    log_path = logs_dir() / "gateway.log"
    configure_logging(log_path=log_path)

    app = create_app(config)
    uvicorn.run(app, host=server_host, port=int(server_port))
