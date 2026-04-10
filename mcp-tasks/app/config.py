from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def info_agent_mock_enabled() -> bool:
    """Читается на каждый вызов — удобно для тестов (monkeypatch env без перезагрузки модуля)."""
    return os.getenv("INFO_AGENT_MOCK", "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    server_name: str = os.getenv("SERVER_NAME", "task-manager")
    sqlite_path: str = os.getenv("SQLITE_PATH", "tasks.db")
    mcp_transport: str = os.getenv("MCP_TRANSPORT", "stdio")
    mcp_http_host: str = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    mcp_http_port: int = int(os.getenv("MCP_HTTP_PORT", "8084"))
    # Ollama (OpenAI-совместимый /v1): локально http://127.0.0.1:11434 , в compose — http://ollama:11434
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")


settings = Settings()
