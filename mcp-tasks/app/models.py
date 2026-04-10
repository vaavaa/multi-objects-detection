from __future__ import annotations

from typing import Any, Literal

AssigneeKey = Literal["serik", "evgeniya", "oksana"]

TaskRecordType = Literal["task", "info_request"]

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


RECORD_TYPE_FOR_MODEL: dict[str, dict[str, str]] = {
    "task": {
        "label_ru": "Задача для исполнителя (handoff)",
        "meaning": (
            "Это поручение сотруднику: по тексту в body нужно выполнить действие "
            "(обработать запрос пользователя, связаться, закрыть вопрос и т.д.). "
            "Статус отражает ход исполнения."
        ),
    },
    "info_request": {
        "label_ru": "Информационный запрос",
        "meaning": (
            "Это запрос динамической информации или справки: по body исполнитель должен "
            "собрать/найти данные и вернуть ответ (не обязательно «закрыть» как классическую задачу)."
        ),
    },
}

STATUS_FOR_MODEL: dict[str, str] = {
    "pending": "ожидает взятия в работу",
    "in_progress": "в работе у исполнителя",
    "done": "выполнено / дан ответ по информационному запросу",
    "cancelled": "отменено",
}


FIELD_REFERENCE_FOR_MODEL: dict[str, str] = {
    "id": "Уникальный UUID записи; передавай в get_task.",
    "task_number": "Глобальный порядковый номер для пользователя («задача №N»).",
    "type": "task — поручение; info_request — информационный запрос.",
    "chat_id": "Идентификатор чата Open WebUI / сессии; разделяет контексты.",
    "user_id": "Идентификатор пользователя-инициатора (если был передан при создании).",
    "body": "Смысл поручения или формулировка информационного запроса.",
    "status": "pending | in_progress | done | cancelled.",
    "assignee": "Ключ менеджера (serik | evgeniya | oksana).",
    "assignee_name": "Человекочитаемое имя менеджера для ответа пользователю.",
    "created_at": "Время создания (UTC, ISO 8601).",
    "updated_at": "Время последнего изменения (UTC, ISO 8601).",
}


def build_get_task_response(
    record_id: str,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Полное тело ответа MCP get_task для модели и UI."""
    if record is None:
        return {
            "found": False,
            "id": record_id,
            "message": "Запись с таким id не найдена.",
            "hint": "Используй UUID из ответа create_handoff_task, create_info_request или из поля id в списках.",
        }

    kind = str(record.get("type", "task"))
    meta = RECORD_TYPE_FOR_MODEL.get(
        kind,
        {"label_ru": kind, "meaning": "Неизвестный тип записи в БД."},
    )
    st = str(record.get("status", ""))
    status_note = STATUS_FOR_MODEL.get(st, st)

    return {
        "found": True,
        "id": record["id"],
        "task_number": record["task_number"],
        "type": kind,
        "type_label_ru": meta["label_ru"],
        "what_this_record_means": meta["meaning"],
        "chat_id": record["chat_id"],
        "user_id": record.get("user_id"),
        "body": record["body"],
        "status": record["status"],
        "status_explained_ru": status_note,
        "assignee": record["assignee"],
        "assignee_name": record["assignee_name"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "field_reference": FIELD_REFERENCE_FOR_MODEL,
    }
