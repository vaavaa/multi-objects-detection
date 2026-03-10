from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBundle:
    text: str
    format_schema: Optional[Dict[str, Any]]
    temperature: Optional[float]
    parser_mode: Optional[str]
    # raw prompt object to link to generation (Langfuse SDK type)
    langfuse_prompt: Any | None
    source: str  # "langfuse" | "fallback"


_CACHE: dict[Tuple[str, str, str], tuple[float, PromptBundle]] = {}
_DEFAULT_TTL_SEC = 60.0


def _is_enabled() -> bool:
    return os.environ.get("LANGFUSE_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_schema(v: Any) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def model_to_prompt_name(model: str) -> str:
    """
    Маппинг модели Ollama → имя промпта в Langfuse.
    Можно расширять без изменения вызывающего кода.
    """
    m = (model or "").lower()
    if "qwen3vl" in m or "qwen3-vl" in m or "qwen3_vl" in m:
        return "hereyouare.detect.qwen3vl"
    if "qwen2.5vl" in m or "qwen2_5vl" in m or "qwen2.5-vl" in m:
        return "hereyouare.detect.qwen2_5vl"
    # default
    return "hereyouare.detect.qwen2_5vl"


def _get_langfuse_client():
    # импорт внутри, чтобы не требовать пакет/ключи при запуске тестов/без Langfuse
    from langfuse import get_client  # type: ignore

    return get_client()


def get_prompt_bundle(
    *,
    model: str,
    fallback_prompt: str,
    fallback_schema: Dict[str, Any],
    fallback_temperature: float,
    ttl_sec: float = _DEFAULT_TTL_SEC,
) -> PromptBundle:
    """
    Возвращает промпт+конфиг для Ollama, в приоритете Langfuse Prompt Management.
    Никогда не бросает исключения: при любой проблеме отдаёт fallback.
    """
    prompt_name = model_to_prompt_name(model)
    label = os.environ.get("LANGFUSE_PROMPT_LABEL", "production")

    if not _is_enabled():
        return PromptBundle(
            text=fallback_prompt,
            format_schema=fallback_schema,
            temperature=fallback_temperature,
            parser_mode=None,
            langfuse_prompt=None,
            source="fallback",
        )

    key = (prompt_name, label, model)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and cached[0] > now:
        return cached[1]

    try:
        lf = _get_langfuse_client()
        lf_prompt = lf.get_prompt(prompt_name, label=label)
        # текст промпта
        try:
            compiled = lf_prompt.compile()
        except TypeError:
            # на случай, если compile требует args; берём базовый шаблон
            compiled = lf_prompt.compile({})

        config = getattr(lf_prompt, "config", None) or {}
        if not isinstance(config, dict):
            config = {}

        temperature = _safe_float(config.get("ollama.temperature") or config.get("temperature"))
        schema = _safe_schema(config.get("ollama.format_schema") or config.get("format_schema"))
        parser_mode = config.get("parser.mode") if isinstance(config.get("parser.mode"), str) else None

        bundle = PromptBundle(
            text=str(compiled),
            format_schema=schema or fallback_schema,
            temperature=temperature if temperature is not None else fallback_temperature,
            parser_mode=parser_mode,
            langfuse_prompt=lf_prompt,
            source="langfuse",
        )
        _CACHE[key] = (now + float(ttl_sec), bundle)
        return bundle
    except Exception as e:
        logger.warning("Langfuse prompt fetch failed (%s/%s): %s", prompt_name, label, e)
        bundle = PromptBundle(
            text=fallback_prompt,
            format_schema=fallback_schema,
            temperature=fallback_temperature,
            parser_mode=None,
            langfuse_prompt=None,
            source="fallback",
        )
        _CACHE[key] = (now + float(ttl_sec), bundle)
        return bundle


def clear_cache() -> None:
    _CACHE.clear()

