from __future__ import annotations

import pytest

from app.store import TaskStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "tasks.sqlite")


@pytest.fixture
def store(db_path) -> TaskStore:
    return TaskStore(db_path)
