"""Структурированный вывод агентов (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InfoResponderOutput(BaseModel):
    """Ответ информационного агента для MCP/модели."""

    answer: str = Field(description="Основной текст ответа пользователю на русском.")
    notes: str = Field(
        default="Тестовый режим: проверьте факты перед использованием в проде.",
        description="Краткое примечание об ограничениях или уверенности.",
    )
