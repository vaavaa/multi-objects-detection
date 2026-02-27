import asyncio
import base64
import colorsys
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import orjson
import xxhash

logger = logging.getLogger(__name__)

import httpx
from fastapi import Body, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from settings import (
    CACHE_TTL_SEC,
    FORMAT_SCHEMA,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_SIDE,
    JOB_RESULT_TTL_SEC,
    MODEL,
    OLLAMA_CONCURRENCY,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
    PROMPT,
    PROVIDER_NAME,
    REDIS_URL,
    YOLO_BASE64_URL,
    YOLO_DEFAULT_CLASS_NAMES,
    YOLO_IOU_THRESHOLD,
    YOLO_MAX_NUM_DETECTIONS,
    YOLO_ONLY_BBOXS,
    YOLO_SCORE_THRESHOLD,
    YOLO_TIMEOUT_SEC,
)

_OLLAMA_SEM = asyncio.Semaphore(OLLAMA_CONCURRENCY)

# Redis: клиент создаётся при старте, закрывается при остановке
_redis: Optional[Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    try:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
        await _redis.ping()
    except (RedisConnectionError, OSError) as e:
        logger.warning("Redis unavailable at startup (%s), job queue disabled", e)
        _redis = None
    yield
    if _redis is not None:
        await _redis.aclose()
        _redis = None


app = FastAPI(
    title="HereYouAre",
    description="Сервис детекции объектов на изображениях (vision через Ollama)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _image_hash(img_bytes: bytes) -> str:
    """Хеш содержимого изображения (xxhash) для кеширования."""
    return xxhash.xxh64(img_bytes).hexdigest()


def _cache_key(img_hash: str) -> str:
    return f"cache:img:{img_hash}"


async def get_redis() -> Optional[Redis]:
    """Возвращает клиент Redis или None, если Redis недоступен."""
    return _redis


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

def _find_json_value_end(s: str, start: int) -> int | None:
    """Индекс закрывающей скобки первого JSON-значения (объект или массив) с учётом строк."""
    depth_curly = 0
    depth_square = 0
    i = start
    in_string = False
    escape = False
    quote = '"'
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == quote:
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            depth_curly += 1
        elif c == "}":
            depth_curly -= 1
            if depth_curly == 0 and depth_square == 0:
                return i
        elif c == "[":
            depth_square += 1
        elif c == "]":
            depth_square -= 1
            if depth_square == 0 and depth_curly == 0:
                return i
        i += 1
    return None


def _safe_json_parse(text: str) -> Dict[str, Any] | list:
    """
    Парсит первый JSON-объект/массив из строки, даже если модель добавила хвост.
    """
    text = text.strip()
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        raise ValueError("No JSON start found in model output")
    start = min(starts)
    end = _find_json_value_end(text, start)
    if end is None:
        raise ValueError("No complete JSON value in model output")
    return orjson.loads(text[start : end + 1])


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


# Доля стороны прямоугольника, видимая у каждого угла (15% — линия только у углов)
_CORNER_FRAC = 0.15


def _draw_corner_lines(
    draw: ImageDraw.Draw,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Рисует только углы прямоугольника: по 15% длины каждой стороны у каждого угла."""
    w = x2 - x1
    h = y2 - y1
    seg_w = max(1, int(w * _CORNER_FRAC))
    seg_h = max(1, int(h * _CORNER_FRAC))
    # верх-лево: горизонталь и вертикаль
    draw.line([(x1, y1), (x1 + seg_w, y1)], fill=color, width=width)
    draw.line([(x1, y1), (x1, y1 + seg_h)], fill=color, width=width)
    # верх-право
    draw.line([(x2 - seg_w, y1), (x2, y1)], fill=color, width=width)
    draw.line([(x2, y1), (x2, y1 + seg_h)], fill=color, width=width)
    # низ-право
    draw.line([(x2, y2 - seg_h), (x2, y2)], fill=color, width=width)
    draw.line([(x2 - seg_w, y2), (x2, y2)], fill=color, width=width)
    # низ-лево
    draw.line([(x1, y2 - seg_h), (x1, y2)], fill=color, width=width)
    draw.line([(x1, y2), (x1 + seg_w, y2)], fill=color, width=width)


def _draw_detections(img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Рисует угловые рамки и подписи на копии изображения. Цвета в одной тональности."""
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
        _draw_corner_lines(draw, x1, y1, x2, y2, color, width)
        label = str(d.get("label", ""))[:30]
        if label:
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


def _parse_yolo_bboxes_response(bboxes: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """
    Парсит ответ YOLO base64: {"bboxes": [{"0": [x1,y1,x2,y2], "label_name": "...", "score": ...}, ...]}
    в список {label, position: [x1,y1,x2,y2]}.
    """
    raw = []
    for item in bboxes or []:
        if not isinstance(item, dict):
            continue
        # Координаты в ключе "0", "1", "2" или "3" (индекс bbox в объекте)
        coords = None
        for k in ("0", "1", "2", "3"):
            v = item.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                coords = v
                break
        if coords is None:
            continue
        label = str(item.get("label_name", "")).strip()
        if not label:
            continue
        raw.append({"label": label, "position": [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]})
    return _normalize_detections(raw, img_w, img_h)


async def _call_yolo_base64(
    img_bytes: bytes,
    class_names: List[str],
    img_w: int = 0,
    img_h: int = 0,
) -> List[Dict[str, Any]]:
    """
    Вызов FastAPI-YOLO base64: multipart img_stream + class_names, only_bboxs=true.
    Ответ: {"bboxes": [{"0": [x1,y1,x2,y2], "label_name": "...", "score": ...}, ...]}.
    Возвращает список детекций {label, position: [x1,y1,x2,y2]}.
    """
    if not class_names:
        class_names = list(YOLO_DEFAULT_CLASS_NAMES)
    class_names_str = ",".join(class_names)
    only_bboxs = "true" if YOLO_ONLY_BBOXS else "false"
    url = (
        f"{YOLO_BASE64_URL.rstrip('/')}"
        f"?iou_threshold={YOLO_IOU_THRESHOLD}&score_threshold={YOLO_SCORE_THRESHOLD}"
        f"&max_num_detections={YOLO_MAX_NUM_DETECTIONS}&only_bboxs={only_bboxs}"
    )
    try:
        async with httpx.AsyncClient(timeout=YOLO_TIMEOUT_SEC) as client:
            files = {"img_stream": ("image.jpg", img_bytes, "image/jpeg")}
            data = {"class_names": class_names_str}
            r = await client.post(url, files=files, data=data)
            if r.status_code != 200:
                logger.warning("YOLO base64 error: %s %s", r.status_code, r.text)
                return []
            body = r.json()
    except Exception as e:
        logger.warning("YOLO base64 request failed: %s", e)
        return []
    bboxes = body.get("bboxes") if isinstance(body, dict) else []
    if not isinstance(bboxes, list):
        return []
    return _parse_yolo_bboxes_response(bboxes, img_w or 0, img_h or 0)


async def _run_async_job(
    job_id: str,
    img_bytes: bytes,
    class_names_override: Optional[List[str]] = None,
    image_hash: Optional[str] = None,
) -> None:
    """
    Фоновая задача: подготовка изображения, Qwen, YOLO, сохранение в job и в кеш по хешу изображения.
    """
    redis = await get_redis()
    if redis is None:
        logger.warning("Async job %s skipped: Redis unavailable", job_id)
        return
    key = _job_key(job_id)

    async def _set_status(payload: dict) -> bool:
        try:
            await redis.set(key, orjson.dumps(payload).decode(), ex=JOB_RESULT_TTL_SEC)
            return True
        except RedisConnectionError as e:
            logger.warning("Redis write failed for job %s: %s", job_id, e)
            return False

    try:
        if not await _set_status({"status": "processing", "job_id": job_id}):
            return
        prepared, w, h, qwen_detections = await _run_detection(img_bytes)
        # Сразу отдаём пользователю ответ Qwen (version v1), пока ждём YOLO
        await _set_status({
            "status": "processing",
            "version": "v1",
            "job_id": job_id,
            "image": {"width": w, "height": h},
            "detections": qwen_detections,
            "provider": PROVIDER_NAME,
        })
        class_names = class_names_override
        if not class_names and qwen_detections:
            class_names = [d.get("label", "") for d in qwen_detections if d.get("label")]
        if not class_names:
            class_names = list(YOLO_DEFAULT_CLASS_NAMES)
        yolo_detections = await _call_yolo_base64(prepared, class_names, w, h)
        # Финальный ответ — координаты от YOLO (version v2); если YOLO пусто — оставляем от Qwen
        detections = yolo_detections if yolo_detections else qwen_detections
        result = {
            "status": "done",
            "version": "v2",
            "job_id": job_id,
            "image": {"width": w, "height": h},
            "detections": detections,
            "provider": PROVIDER_NAME,
        }
        await _set_status(result)
        # Сохраняем в кеш по хешу изображения, чтобы не пересчитывать ту же картинку
        if image_hash:
            try:
                await redis.set(
                    _cache_key(image_hash),
                    orjson.dumps(result).decode(),
                    ex=CACHE_TTL_SEC,
                )
            except RedisConnectionError:
                pass
    except Exception as e:
        logger.exception("Async job %s failed", job_id)
        await _set_status({"status": "failed", "job_id": job_id, "error": str(e)})


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


# --- Асинхронная очередь заданий (Job_id + Redis + YOLO) ---


def _redis_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Redis unavailable; job queue disabled. Start Redis or check REDIS_URL.",
    )


@app.post("/v1/detect/async", tags=["Detection", "Jobs"])
async def detect_async(
    file: UploadFile = File(...),
    class_names: Optional[str] = Form(None),
) -> dict:
    """
    Создаёт отложенное задание: возвращает job_id сразу, детекция (Qwen + YOLO) выполняется в фоне.
    Результат забирать через GET /v1/job/{job_id}.
    Одинаковые по хешу изображения берутся из кеша — пересчёт не выполняется.
    class_names — опционально, строка через запятую (например: headphones,keyboard,mouse).
    """
    redis = await get_redis()
    if redis is None:
        raise _redis_unavailable()
    raw = await file.read()
    img_hash = _image_hash(raw)
    job_id = str(uuid.uuid4())
    # Проверяем кеш: если для этого хеша уже есть готовый результат (v2), отдаём его по новому job_id
    try:
        cached = await redis.get(_cache_key(img_hash))
        if cached:
            data = orjson.loads(cached)
            if data.get("status") == "done" and data.get("version") == "v2":
                data["job_id"] = job_id
                await redis.set(
                    _job_key(job_id),
                    orjson.dumps(data).decode(),
                    ex=JOB_RESULT_TTL_SEC,
                )
                return {"job_id": job_id, "status": "done", "cached": True}
    except RedisConnectionError:
        pass
    names_list: Optional[List[str]] = None
    if class_names:
        names_list = [s.strip() for s in class_names.split(",") if s.strip()]
    asyncio.create_task(_run_async_job(job_id, raw, names_list, image_hash=img_hash))
    return {"job_id": job_id, "status": "processing"}


@app.get("/v1/job/{job_id}", tags=["Jobs"])
async def get_job(job_id: str) -> dict:
    """
    Статус и результат задания по job_id.
    status: processing (есть промежуточный результат) | done | failed.
    version: v1 — ответ Qwen (можно показывать, пока идёт YOLO); v2 — финальный ответ с координатами YOLO.
    В теле при наличии: image, detections, provider.
    """
    redis = await get_redis()
    if redis is None:
        raise _redis_unavailable()
    key = _job_key(job_id)
    try:
        data = await redis.get(key)
    except RedisConnectionError as e:
        logger.warning("Redis connection error in get_job: %s", e)
        raise _redis_unavailable()
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return orjson.loads(data)