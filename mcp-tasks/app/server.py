from __future__ import annotations

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings
from app.models import AssigneeKey, TaskStatus
from app.store import TaskStore

mcp = FastMCP(settings.server_name)


@mcp.custom_route("/health", methods=["GET"])
async def http_health(_request: Request) -> JSONResponse:
    """Plain HTTP check (curl / load balancers). MCP endpoint remains POST /mcp."""
    return JSONResponse(
        {
            "status": "ok",
            "service": settings.server_name,
            "mcp_url_path": "/mcp",
            "sqlite_path": settings.sqlite_path,
            "note": "Open WebUI: MCP server URL is http://<host>:<port>/mcp (streamable HTTP).",
        }
    )


_store: TaskStore | None = None


def get_store() -> TaskStore:
    global _store
    if _store is None:
        _store = TaskStore(settings.sqlite_path)
    return _store


@mcp.tool()
def task_health() -> dict:
    """Service status and storage path."""
    return {
        "status": "ok",
        "server": settings.server_name,
        "sqlite_path": settings.sqlite_path,
    }


@mcp.tool()
def create_handoff_task(
    chat_id: str,
    body: str,
    assignee: AssigneeKey,
    user_id: str | None = None,
    status: TaskStatus = "pending",
) -> dict:
    """
    Create a task. Returns task_number (global), id (UUID), and assignee_name for user-facing messages.

    assignee: serik | evgeniya | oksana
    """
    return get_store().create_task(
        chat_id=chat_id,
        body=body,
        assignee=assignee,
        user_id=user_id,
        status=status,
    )


@mcp.tool()
def list_tasks(chat_id: str, user_id: str | None = None) -> dict:
    """List tasks for a chat. Optionally filter by user_id."""
    tasks = get_store().list_tasks(chat_id=chat_id, user_id=user_id)
    return {"chat_id": chat_id, "user_id": user_id, "count": len(tasks), "tasks": tasks}


if __name__ == "__main__":
    if settings.mcp_transport == "http":
        mcp.run(
            transport="http",
            host=settings.mcp_http_host,
            port=settings.mcp_http_port,
        )
    else:
        mcp.run()
