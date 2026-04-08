from __future__ import annotations

import pytest

from app.store import TaskStore


@pytest.fixture
def srv(monkeypatch, tmp_path):
    import app.server as server_module

    monkeypatch.setattr(server_module, "_store", None)
    isolated = TaskStore(str(tmp_path / "server.sqlite"))
    monkeypatch.setattr(server_module, "get_store", lambda: isolated)
    return server_module, isolated


def test_task_health(srv):
    server_module, _ = srv
    h = server_module.task_health()
    assert h["status"] == "ok"
    assert h["server"] == server_module.settings.server_name
    assert h["sqlite_path"] == server_module.settings.sqlite_path


def test_create_task_tool(srv):
    server_module, _ = srv
    out = server_module.create_task(
        chat_id="chat-1",
        body="Сделать отчёт",
        assignee="oksana",
        user_id="user-42",
        status="pending",
    )
    assert out["body"] == "Сделать отчёт"
    assert out["chat_id"] == "chat-1"
    assert out["user_id"] == "user-42"
    assert out["assignee"] == "oksana"
    assert out["assignee_name"] == "Оксана"
    assert out["status"] == "pending"
    assert out["task_number"] == 1


def test_list_tasks_tool(srv):
    server_module, store = srv
    server_module.create_task(chat_id="c", body="t1", assignee="serik")
    server_module.create_task(chat_id="c", body="t2", assignee="evgeniya", user_id="u1")
    wrapped = server_module.list_tasks(chat_id="c")
    assert wrapped["chat_id"] == "c"
    assert wrapped["user_id"] is None
    assert wrapped["count"] == 2
    assert len(wrapped["tasks"]) == 2
    assert wrapped["tasks"][0]["body"] == "t2"

    filt = server_module.list_tasks(chat_id="c", user_id="u1")
    assert filt["count"] == 1
    assert filt["tasks"][0]["body"] == "t2"
