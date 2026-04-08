from __future__ import annotations

from typing import Literal

AssigneeKey = Literal["serik", "evgeniya", "oksana"]

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"

TaskStatus = Literal["pending", "in_progress", "done", "cancelled"]

ASSIGNEE_NAMES: dict[str, str] = {
    "serik": "Серик",
    "evgeniya": "Евгения",
    "oksana": "Оксана",
}


def assignee_display(key: str) -> str:
    return ASSIGNEE_NAMES.get(key, key)
