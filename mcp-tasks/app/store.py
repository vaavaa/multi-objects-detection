from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.models import ASSIGNEE_NAMES, assignee_display

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
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
CREATE INDEX IF NOT EXISTS idx_tasks_chat_id ON tasks(chat_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
"""


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._path)
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def create_task(
        self,
        *,
        chat_id: str,
        body: str,
        assignee: str,
        user_id: str | None,
        status: str,
    ) -> dict[str, Any]:
        if assignee not in ASSIGNEE_NAMES:
            raise ValueError(f"assignee must be one of: {', '.join(ASSIGNEE_NAMES)}")
        task_id = str(uuid4())
        now = _utc_iso()
        with self._lock:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    """
                    INSERT INTO tasks (id, chat_id, user_id, body, status, assignee, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task_id, chat_id, user_id, body, status, assignee, now, now),
                )
                num = cur.lastrowid
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_number = ?",
                    (num,),
                ).fetchone()
            finally:
                conn.close()
        return self._row_to_dict(row)

    def list_tasks(self, *, chat_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            try:
                if user_id is None:
                    cur = conn.execute(
                        """
                        SELECT * FROM tasks
                        WHERE chat_id = ?
                        ORDER BY task_number DESC
                        """,
                        (chat_id,),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT * FROM tasks
                        WHERE chat_id = ? AND user_id = ?
                        ORDER BY task_number DESC
                        """,
                        (chat_id, user_id),
                    )
                rows = cur.fetchall()
            finally:
                conn.close()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        key = d["assignee"]
        d["assignee_name"] = assignee_display(key)
        return d
