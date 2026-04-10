"""
Реестр агентов PydanticAI: создание модели Ollama и кэш экземпляров.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agents.prompts import INFO_RESPONDER_SYSTEM
from app.agents.schemas import InfoResponderOutput
from app.config import settings

_CACHE: dict[str, Any] = {}


def _ollama_openai_base_url() -> str:
    base = settings.ollama_base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def build_info_responder_agent() -> Agent[None, InfoResponderOutput]:
    model = OpenAIChatModel(
        settings.ollama_model,
        provider=OpenAIProvider(
            base_url=_ollama_openai_base_url(),
            api_key="ollama",
        ),
    )
    return Agent(
        model,
        output_type=InfoResponderOutput,
        system_prompt=INFO_RESPONDER_SYSTEM,
    )


def get_info_responder_agent() -> Agent[None, InfoResponderOutput]:
    key = "info_responder"
    if key not in _CACHE:
        _CACHE[key] = build_info_responder_agent()
    return _CACHE[key]


def clear_agent_cache() -> None:
    _CACHE.clear()
