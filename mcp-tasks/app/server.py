from __future__ import annotations

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.agents import run_info_responder_answer
from app.config import settings
from app.models import TaskStatus, build_get_task_response
from app.store import TaskStore
import random


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
def get_task(record_id: str) -> dict:
    """
    Полная карточка записи по UUID поля `id` (из ответов create_handoff_task / create_info_request).

    Возвращает оба типа: обычную задачу (type=task) и информационный запрос (type=info_request).
    Включены человекочитаемые пояснения для модели: что означает тип, статус и каждое поле.
    """
    raw = get_store().get_record_by_id(record_id.strip())
    return build_get_task_response(record_id.strip(), raw)


@mcp.tool()
def info_agent_answer(question: str, chat_id: str | None = None) -> dict:
    """
    Справочный ответ по информационному запросу через локальную Ollama (PydanticAI).

    Не создаёт запись в БД — только генерирует текст для пользователя (в тесте допускаются обобщения).
    Передай chat_id, если ответ должен учитывать разделение чатов в тексте ответа.
    """
    out = run_info_responder_answer(question=question, chat_id=chat_id)
    return {
        "answer": out.answer,
        "notes": out.notes,
        "model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
    }


@mcp.tool()
def task_health() -> dict:
    """Service status and storage path."""
    return {
        "status": "ok",
        "server": settings.server_name,
        "sqlite_path": settings.sqlite_path,
    }

@mcp.tool()
def create_info_request(
    chat_id: str,
    body: str,
    user_id: str | None = None,
    status: TaskStatus = "pending",
) -> dict:
    """
    Create an info request for dynamic information.
    Returns task_number (global), id (UUID), and assignee_name for user-facing messages.
    assignee: serik | evgeniya | oksana
    """
    assignee = random.choice(["serik", "evgeniya", "oksana"])
    return get_store().create_info_request(chat_id=chat_id, body=body, assignee=assignee, user_id=user_id, status=status)


@mcp.tool()
def create_handoff_task(
    chat_id: str,
    body: str,
    user_id: str | None = None,
    status: TaskStatus = "pending",
) -> dict:
    """
    Create a task. Returns task_number (global), id (UUID), and assignee_name for user-facing messages.
    assignee: serik | evgeniya | oksana
    """

    assignee = random.choice(["serik", "evgeniya", "oksana"])
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
