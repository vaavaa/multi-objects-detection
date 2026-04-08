from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    server_name: str = os.getenv("SERVER_NAME", "task-manager")
    sqlite_path: str = os.getenv("SQLITE_PATH", "tasks.db")
    mcp_transport: str = os.getenv("MCP_TRANSPORT", "stdio")
    mcp_http_host: str = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    mcp_http_port: int = int(os.getenv("MCP_HTTP_PORT", "8084"))


settings = Settings()
