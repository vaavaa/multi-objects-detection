"""
Синхронный вызов информационного агента (для MCP tools).
"""

from __future__ import annotations

from app.agents.prompts import INFO_RESPONDER_USER_TEMPLATE
from app.agents.registry import get_info_responder_agent
from app.agents.schemas import InfoResponderOutput
from app.config import info_agent_mock_enabled, settings


def run_info_responder_answer(question: str, chat_id: str | None = None) -> InfoResponderOutput:
    """Один запрос → структурированный ответ через Ollama (или mock)."""
    if info_agent_mock_enabled():
        q = (question or "").strip() or "(пустой запрос)"
        return InfoResponderOutput(
            answer=(
                f"[MOCK] Тестовый ответ без обращения к Ollama. "
                f"Вы спросили: «{q[:200]}». В проде здесь будет ответ модели "
                f"`{settings.ollama_model}`."
            ),
            notes="INFO_AGENT_MOCK=1. Для реального вызова отключите переменную.",
        )

    agent = get_info_responder_agent()
    chat_line = f"Идентификатор чата (контекст): {chat_id}\n" if chat_id else ""
    user_prompt = INFO_RESPONDER_USER_TEMPLATE.format(
        question=question.strip(),
        chat_line=chat_line,
    )
    result = agent.run_sync(user_prompt)
    return result.output
