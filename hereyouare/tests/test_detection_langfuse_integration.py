import base64
import io

import pytest
from PIL import Image


def _make_jpeg_bytes(w: int = 64, h: int = 48) -> bytes:
    img = Image.new("RGB", (w, h), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_run_detection_uses_langfuse_prompt_in_payload(monkeypatch):
    # Важно: импортируем main после подмен, чтобы settings/окружение применились.
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PROMPT_LABEL", "production")

    # Подменяем get_prompt_bundle, чтобы не зависеть от реального langfuse SDK.
    from langfuse_runtime import PromptBundle

    def fake_bundle(**kwargs):
        return PromptBundle(
            text="PROMPT_FROM_LANGFUSE",
            format_schema={"type": "array"},
            temperature=0.2,
            parser_mode=None,
            langfuse_prompt=None,
            source="langfuse",
        )

    monkeypatch.setattr("main.get_prompt_bundle", fake_bundle, raising=False)

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            # минимальный валидный JSON для _safe_json_parse
            return {"response": '[{"label":"cat","position":[1,2,30,40]}]'}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, **kwargs):
            captured["url"] = url
            captured["payload"] = json
            # sanity: image should be base64
            assert "images" in json and isinstance(json["images"], list) and json["images"]
            base64.b64decode(json["images"][0])
            return FakeResponse()

    monkeypatch.setattr("main.httpx.AsyncClient", FakeClient)

    from main import _run_detection

    raw = _make_jpeg_bytes()
    prepared, w, h, dets = await _run_detection(raw)
    assert prepared
    assert w > 0 and h > 0
    assert dets and dets[0]["label"] == "cat"
    assert captured["payload"]["prompt"] == "PROMPT_FROM_LANGFUSE"
    assert captured["payload"]["options"]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_run_detection_fallback_when_langfuse_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"response": '[{"label":"dog","position":[1,2,30,40]}]'}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, **kwargs):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("main.httpx.AsyncClient", FakeClient)

    from main import PROMPT as DEFAULT_PROMPT  # from settings import PROMPT in main module
    from main import _run_detection

    raw = _make_jpeg_bytes()
    _prepared, _w, _h, dets = await _run_detection(raw)
    assert dets and dets[0]["label"] == "dog"
    assert captured["payload"]["prompt"] == DEFAULT_PROMPT

