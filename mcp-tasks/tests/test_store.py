from __future__ import annotations

import re
import uuid

import pytest

import sqlite3

from app.store import TaskStore


def test_migration_adds_type_column_on_legacy_db(tmp_path):
    """БД до появления поля type — при открытии TaskStore колонка добавляется."""
    path = str(tmp_path / "legacy.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tasks (
          task_number INTEGER PRIMARY KEY AUTOINCREMENT,
          id TEXT NOT NULL UNIQUE,
          chat_id TEXT NOT NULL,
          user_id TEXT,
          body TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',
          assignee TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    store = TaskStore(path)
    t = store.create_task(
        chat_id="c",
        body="x",
        assignee="oksana",
        user_id=None,
        status="pending",
    )
    assert t["type"] == "task"


def test_create_returns_global_task_numbers_and_uuid(store):
    t1 = store.create_task(
        chat_id="chat-a",
        body="first",
        assignee="oksana",
        user_id=None,
        status="pending",
    )
    t2 = store.create_task(
        chat_id="chat-b",
        body="second",
        assignee="serik",
        user_id="user-1",
        status="in_progress",
    )
    assert t1["task_number"] == 1
    assert t2["task_number"] == 2
    assert t1["task_number"] < t2["task_number"]
    uuid.UUID(t1["id"])
    uuid.UUID(t2["id"])
    assert t1["id"] != t2["id"]


def test_create_includes_assignee_name(store):
    t = store.create_task(
        chat_id="c1",
        body="x",
        assignee="evgeniya",
        user_id=None,
        status="pending",
    )
    assert t["assignee"] == "evgeniya"
    assert t["assignee_name"] == "Евгения"


def test_invalid_assignee_raises(store):
    with pytest.raises(ValueError, match="assignee must be one of"):
        store.create_task(
            chat_id="c",
            body="b",
            assignee="not_a_manager",
            user_id=None,
            status="pending",
        )


def test_list_tasks_by_chat_only(store):
    store.create_task(chat_id="c1", body="a", assignee="oksana", user_id="u1", status="pending")
    store.create_task(chat_id="c1", body="b", assignee="serik", user_id="u2", status="done")
    store.create_task(chat_id="c2", body="c", assignee="oksana", user_id=None, status="pending")
    rows = store.list_tasks(chat_id="c1")
    assert len(rows) == 2
    bodies = {r["body"] for r in rows}
    assert bodies == {"a", "b"}


def test_list_tasks_filter_user_id(store):
    store.create_task(chat_id="c1", body="mine", assignee="oksana", user_id="u1", status="pending")
    store.create_task(chat_id="c1", body="other", assignee="serik", user_id="u2", status="pending")
    rows = store.list_tasks(chat_id="c1", user_id="u1")
    assert len(rows) == 1
    assert rows[0]["body"] == "mine"


def test_list_tasks_order_desc_by_task_number(store):
    store.create_task(chat_id="c1", body="old", assignee="oksana", user_id=None, status="pending")
    store.create_task(chat_id="c1", body="new", assignee="oksana", user_id=None, status="pending")
    rows = store.list_tasks(chat_id="c1")
    assert [r["body"] for r in rows] == ["new", "old"]


def test_timestamps_iso_utc_like(store):
    t = store.create_task(
        chat_id="c",
        body="x",
        assignee="oksana",
        user_id=None,
        status="cancelled",
    )
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", t["created_at"])
    assert t["created_at"] == t["updated_at"]


def test_get_record_by_id_task_and_info_request(store):
    t = store.create_task(
        chat_id="c",
        body="handoff",
        assignee="oksana",
        user_id="u1",
        status="pending",
    )
    i = store.create_info_request(
        chat_id="c",
        body="info?",
        assignee="serik",
        user_id=None,
        status="pending",
    )
    gt = store.get_record_by_id(t["id"])
    gi = store.get_record_by_id(i["id"])
    assert gt is not None and gt["type"] == "task"
    assert gi is not None and gi["type"] == "info_request"
    assert store.get_record_by_id("00000000-0000-0000-0000-000000000000") is None


def test_second_store_same_file_sees_data(store, db_path):
    t = store.create_task(
        chat_id="c",
        body="shared",
        assignee="oksana",
        user_id=None,
        status="pending",
    )
    other = TaskStore(db_path)
    rows = other.list_tasks(chat_id="c")
    assert len(rows) == 1
    assert rows[0]["task_number"] == t["task_number"]
