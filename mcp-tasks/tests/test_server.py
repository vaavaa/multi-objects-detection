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


def test_info_agent_answer_tool(srv, monkeypatch):
    monkeypatch.setenv("INFO_AGENT_MOCK", "1")
    import app.config as cfg

    # сброс кэша настроек не нужен: mock читается через os.environ на каждый вызов
    server_module, _ = srv
    out = server_module.info_agent_answer("Тестовый вопрос", chat_id="c1")
    assert "MOCK" in out["answer"]
    assert out["model"] == cfg.settings.ollama_model


def test_get_task_tool(srv):
    server_module, _ = srv
    out = server_module.create_handoff_task(chat_id="c", body="детали")
    full = server_module.get_task(out["id"])
    assert full["found"] is True
    assert full["type"] == "task"
    assert full["body"] == "детали"
    assert "what_this_record_means" in full
    assert "field_reference" in full
    miss = server_module.get_task("00000000-0000-0000-0000-000000000001")
    assert miss["found"] is False


def test_task_health(srv):
    server_module, _ = srv
    h = server_module.task_health()
    assert h["status"] == "ok"
    assert h["server"] == server_module.settings.server_name
    assert h["sqlite_path"] == server_module.settings.sqlite_path


def test_create_handoff_task_tool(srv):
    server_module, _ = srv
    out = server_module.create_handoff_task(
        chat_id="chat-1",
        body="Сделать отчёт",
        user_id="user-42",
        status="pending",
    )
    assert out["body"] == "Сделать отчёт"
    assert out["chat_id"] == "chat-1"
    assert out["user_id"] == "user-42"
    assert out["assignee"] in ("serik", "evgeniya", "oksana")
    assert out["assignee_name"] in ("Серик", "Евгения", "Оксана")
    assert out["status"] == "pending"
    assert out["task_number"] == 1


def test_list_tasks_tool(srv):
    server_module, store = srv
    server_module.create_handoff_task(chat_id="c", body="t1")
    server_module.create_handoff_task(chat_id="c", body="t2", user_id="u1")
    wrapped = server_module.list_tasks(chat_id="c")
    assert wrapped["chat_id"] == "c"
    assert wrapped["user_id"] is None
    assert wrapped["count"] == 2
    assert len(wrapped["tasks"]) == 2
    assert wrapped["tasks"][0]["body"] == "t2"

    filt = server_module.list_tasks(chat_id="c", user_id="u1")
    assert filt["count"] == 1
    assert filt["tasks"][0]["body"] == "t2"
