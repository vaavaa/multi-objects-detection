import os

import pytest


def test_model_to_prompt_name_qwen3():
    from langfuse_runtime import model_to_prompt_name

    assert model_to_prompt_name("qwen3vl:7b") == "hereyouare.detect.qwen3vl"
    assert model_to_prompt_name("qwen3-vl:7b") == "hereyouare.detect.qwen3vl"


def test_model_to_prompt_name_qwen25():
    from langfuse_runtime import model_to_prompt_name

    assert model_to_prompt_name("qwen2.5vl:7b") == "hereyouare.detect.qwen2_5vl"
    assert model_to_prompt_name("qwen2.5-vl:7b") == "hereyouare.detect.qwen2_5vl"


def test_get_prompt_bundle_fallback_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from langfuse_runtime import get_prompt_bundle

    bundle = get_prompt_bundle(
        model="qwen2.5vl:7b",
        fallback_prompt="FALLBACK",
        fallback_schema={"type": "array"},
        fallback_temperature=0.1,
        ttl_sec=0.01,
    )
    assert bundle.source == "fallback"
    assert bundle.text == "FALLBACK"
    assert bundle.temperature == 0.1
    assert bundle.format_schema == {"type": "array"}
    assert bundle.langfuse_prompt is None


def test_get_prompt_bundle_uses_langfuse_prompt(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PROMPT_LABEL", "production")

    class FakePrompt:
        config = {
            "ollama.temperature": 0.42,
            "ollama.format_schema": {"type": "array", "items": {"type": "object"}},
            "parser.mode": "array",
        }

        def compile(self, *args, **kwargs):
            return "PROMPT_FROM_LANGFUSE"

    class FakeClient:
        def get_prompt(self, name, label=None):
            assert name in ("hereyouare.detect.qwen2_5vl", "hereyouare.detect.qwen3vl")
            assert label == "production"
            return FakePrompt()

    monkeypatch.setattr("langfuse_runtime._get_langfuse_client", lambda: FakeClient())

    from langfuse_runtime import clear_cache, get_prompt_bundle

    clear_cache()
    bundle = get_prompt_bundle(
        model="qwen2.5vl:7b",
        fallback_prompt="FALLBACK",
        fallback_schema={"type": "array"},
        fallback_temperature=0.1,
        ttl_sec=0.01,
    )
    assert bundle.source == "langfuse"
    assert bundle.text == "PROMPT_FROM_LANGFUSE"
    assert bundle.temperature == 0.42
    assert bundle.format_schema["type"] == "array"
    assert bundle.parser_mode == "array"
    assert bundle.langfuse_prompt is not None

