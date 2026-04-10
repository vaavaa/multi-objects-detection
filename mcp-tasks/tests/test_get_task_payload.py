from __future__ import annotations

from app.models import build_get_task_response


def test_build_get_task_not_found():
    r = build_get_task_response("abc", None)
    assert r["found"] is False
    assert r["id"] == "abc"
    assert "hint" in r


def test_build_get_task_full_task_record():
    raw = {
        "id": "i1",
        "task_number": 5,
        "type": "task",
        "chat_id": "ch",
        "user_id": None,
        "body": "do it",
        "status": "pending",
        "assignee": "oksana",
        "assignee_name": "Оксана",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    r = build_get_task_response("i1", raw)
    assert r["found"] is True
    assert r["type"] == "task"
    assert r["type_label_ru"]
    assert r["what_this_record_means"]
    assert r["status_explained_ru"]
    assert "field_reference" in r
    assert r["body"] == "do it"


def test_build_get_task_info_request():
    raw = {
        "id": "i2",
        "task_number": 6,
        "type": "info_request",
        "chat_id": "ch",
        "user_id": "u",
        "body": "узнать срок",
        "status": "in_progress",
        "assignee": "evgeniya",
        "assignee_name": "Евгения",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    r = build_get_task_response("i2", raw)
    assert r["found"] is True
    assert r["type"] == "info_request"
    assert "информац" in r["type_label_ru"].lower()
