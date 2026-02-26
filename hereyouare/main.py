import asyncio
import base64
import colorsys
import io
import json
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

import httpx
from fastapi import Body, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont

from settings import (
    FORMAT_SCHEMA,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_SIDE,
    MODEL,
    OLLAMA_CONCURRENCY,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
    PROMPT,
    PROVIDER_NAME,
)

_OLLAMA_SEM = asyncio.Semaphore(OLLAMA_CONCURRENCY)

app = FastAPI(
    title="HereYouAre",
    description="Сервис детекции объектов на изображениях (vision через Ollama)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/health", tags=["Service"])
async def healthcheck() -> dict[str, str]:
    """Проверка живости сервиса (для Docker/orchestrator)."""
    return {"status": "ok"}


@app.get("/echo", tags=["Service"])
@app.post("/echo", tags=["Service"])
async def echo(
    message: str | None = None,
    body: dict[str, Any] | None = Body(None),
) -> dict[str, Any]:
    """Возвращает переданные данные (для отладки и проверки доступности)."""
    if body is not None:
        return {"echo": body}
    if message is not None:
        return {"echo": {"message": message}}
    return {"echo": None, "hint": "Send JSON body (POST) or ?message=... (GET)"}


def _prepare_image_bytes(img_bytes: bytes, max_side: int = IMAGE_MAX_SIDE) -> tuple[bytes, int, int]:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        im = im.resize((new_w, new_h))
        w, h = new_w, new_h
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=False)
    return out.getvalue(), w, h

def _safe_json_parse(text: str) -> Dict[str, Any] | list:
    """
    Парсит первый JSON-объект/массив из строки, даже если модель добавила хвост.
    При включённом format/schema в Ollama хвоста обычно уже не будет.
    """
    text = text.strip()
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        raise ValueError("No JSON start found in model output")
    start = min(starts)
    s = text[start:]
    decoder = json.JSONDecoder()
    obj, _idx = decoder.raw_decode(s)
    return obj


def _normalize_detections(
    raw: List[Dict[str, Any]], img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """
    Приводит ответ модели к списку {label, position: [x1,y1,x2,y2]}.
    - position в пикселях, порядок x1<=x2, y1<=y2, в границах изображения
    - отфильтровать пустые label и слишком маленькие боксы
    """
    out: List[Dict[str, Any]] = []
    for d in raw or []:
        try:
            label = str(d.get("label", "")).strip()
            pos = d.get("position")
            if not label or not isinstance(pos, (list, tuple)) or len(pos) != 4:
                continue
            x1, y1, x2, y2 = float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3])
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1 = max(0, min(img_w, x1))
            x2 = max(0, min(img_w, x2))
            y1 = max(0, min(img_h, y1))
            y2 = max(0, min(img_h, y2))
            area = (x2 - x1) * (y2 - y1)
            if area < 100:  # слишком маленький бокс в пикселях
                continue
            out.append({
                "label": label,
                "position": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            })
        except Exception:
            continue
    return out[:20]


def _palette_same_tone(n: int, hue: float = 0.55) -> List[tuple[int, int, int]]:
    """Палитра из n цветов в одной тональности (HSL: один hue, разные S и L)."""
    colors = []
    for i in range(n):
        # hue 0.55 — синевато-бирюзовый; варьируем насыщенность и яркость
        s = 0.5 + (i % 5) * 0.12
        l = 0.35 + (i % 4) * 0.12
        r, g, b = colorsys.hls_to_rgb(hue, l, s)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _draw_detections(img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Рисует прямоугольники и подписи на копии изображения. Цвета в одной тональности."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    colors = _palette_same_tone(max(len(detections), 1))
    width = max(2, min(out.size) // 400)

    for i, d in enumerate(detections):
        pos = d.get("position")
        if not pos or len(pos) != 4:
            continue
        x1, y1, x2, y2 = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = str(d.get("label", ""))[:30]
        if label:
            # подпись сверху слева от бокса, чтобы не перекрывать
            ty = max(0, y1 - 18)
            draw.text((x1, ty), label, fill=color, font=font)
    return out


async def _run_detection(raw: bytes) -> tuple[bytes, int, int, List[Dict[str, Any]]]:
    """Общая логика: подготовка изображения, запрос к Ollama, парсинг, нормализация."""
    try:
        prepared, w, h = _prepare_image_bytes(raw, max_side=IMAGE_MAX_SIDE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_b64 = base64.b64encode(prepared).decode("utf-8")
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "format": FORMAT_SCHEMA,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }

    async with _OLLAMA_SEM:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SEC) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")

    data = r.json()
    model_text = data.get("response", "")
    logger.info("Ollama model response (before parse): %s", model_text)

    try:
        parsed = _safe_json_parse(model_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model did not return JSON: {e}")

    if isinstance(parsed, list):
        raw_detections = parsed
    elif isinstance(parsed, dict):
        raw_detections = parsed.get("detections", [])
    else:
        raw_detections = []
    detections = _normalize_detections(raw_detections, w, h)
    return prepared, w, h, detections


@app.post("/v1/detect", tags=["Detection"])
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    prepared, w, h, detections = await _run_detection(raw)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "image": {"width": w, "height": h},
        "detections": detections,
        "provider": PROVIDER_NAME,
        "latency_ms": latency_ms,
    }


@app.post("/v1/detect/image", tags=["Detection"])
async def detect_image(file: UploadFile = File(...)) -> Response:
    """
    Детекция объектов и возврат изображения с нарисованными боксами и подписями.
    Цвета прямоугольников в одной тональности (бирюзово-синие).
    """
    raw = await file.read()
    prepared, w, h, detections = await _run_detection(raw)
    img = Image.open(io.BytesIO(prepared)).convert("RGB")
    annotated = _draw_detections(img, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")