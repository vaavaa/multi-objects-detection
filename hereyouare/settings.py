"""
Настройки сервиса детекции объектов.
При необходимости можно добавить чтение из переменных окружения (os.environ).
"""

import os
from typing import Any, Dict, List

# Ollama API
OLLAMA_URL = "http://ollama:11434/api/generate"  # в docker-compose это имя сервиса
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q8_0") # qwen2.5vl:7b-q8_0
# MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b-q8_0")

# Ограничьте параллельные запросы к GPU (иначе очередь внутри Ollama и рост latency)
OLLAMA_CONCURRENCY = 1

# Промпт для модели (детекция объектов: массив {label, position [x1,y1,x2,y2]})
PROMPT = "List visible objects as JSON, labels in English:"

# Guided decoding: массив объектов с label и position [x1, y1, x2, y2] (пиксели)
FORMAT_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["label", "position"],
    },
}

# Обработка изображения
IMAGE_MAX_SIDE = 1024  # максимальная сторона при ресайзе
IMAGE_JPEG_QUALITY = 85

# Запрос к Ollama
OLLAMA_TIMEOUT_SEC = 120
OLLAMA_TEMPERATURE = 0.1  # ограничение "болтливости" модели

# Метаданные в ответе
PROVIDER_NAME = "qwen2.5vl"

# Qwen: при детекции > доли площади кадра — обрезка по боксу (отступ с каждой стороны) и повтор
QWEN_CROP_MAX_ITERATIONS = 3
QWEN_BIG_DETECTION_AREA_FRAC = 0.90  # порог: если одна детекция больше — кроп и повтор
QWEN_CROP_INSET_FRAC = 0.005  # 0.5% с каждой стороны при обрезке

# --- Очередь заданий (Redis) ---
USE_REDIS = os.environ.get("USE_REDIS", "true").lower() in ("1", "true", "yes")
USE_CACHE = os.environ.get("USE_CACHE", "false").lower() in ("1", "true", "yes")  # сохранять и отдавать кеш распознавания
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_RESULT_TTL_SEC = 3600  # время жизни результата задания в Redis (1 ч)
CACHE_TTL_SEC = 86400  # TTL кеша по хешу изображения (24 ч)
# Задержка (мс) между записью v1 в Redis и записью v2 после получения результата YOLO
V2_DELAY_MS = int(os.environ.get("V2_DELAY_MS", "0"))

# --- FastAPI-YOLO (YOLO World): детекция по заданным классам ---
# Базовый URL эндпоинта base64. Из контейнера до хоста: host.docker.internal (Docker 20.10+)
YOLO_BASE64_URL = os.environ.get(
    "YOLO_BASE64_URL",
    "http://host.docker.internal:8001/api/v1/yworld/base64",
)
YOLO_IOU_THRESHOLD = 0.5
YOLO_SCORE_THRESHOLD = 0.2
YOLO_MAX_NUM_DETECTIONS = 15
YOLO_ONLY_BBOXS = True  # only_bboxs=true в query
YOLO_TIMEOUT_SEC = 60
# Классы по умолчанию, если не переданы и не получены из Qwen
YOLO_DEFAULT_CLASS_NAMES: List[str] = []

# Режим объединения детекций v1 (Qwen) и v2 (YOLO):
# True  - объединять боксы (YOLO + Qwen-добавки по недостающим классам)
# False - использовать только детекции YOLO (если они есть)
MERGE_QWEN_YOLO_DETECTIONS = os.environ.get(
    "MERGE_QWEN_YOLO_DETECTIONS",
    "true",
).lower() in ("1", "true", "yes")


# --- Синонимы label (для согласования Qwen ↔ YOLO) ---
# Канон задаётся динамически: набор меток, которые Qwen вернула первой детекцией на конкретном изображении.
# Если YOLO вернула другой label, то для отсутствующих Qwen-меток подбираем синоним среди label, которые YOLO реально вернула.

# Модель для поиска синонимов (текстовая). По умолчанию используем ту же, что и для vision.
LABEL_CANON_MODEL = os.environ.get("LABEL_CANON_MODEL", "kimi-k2.5:cloud")
LABEL_CANON_TIMEOUT_SEC = int(os.environ.get("LABEL_CANON_TIMEOUT_SEC", "30"))
LABEL_CANON_TEMPERATURE = float(os.environ.get("LABEL_CANON_TEMPERATURE", "0.0"))
LABEL_CANON_MIN_CONFIDENCE = float(os.environ.get("LABEL_CANON_MIN_CONFIDENCE", "0.6"))
LABEL_CANON_CACHE_TTL_SEC = int(os.environ.get("LABEL_CANON_CACHE_TTL_SEC", "86400"))
LABEL_CANON_FORCE = os.environ.get("LABEL_CANON_FORCE", "false").lower() in ("1", "true", "yes", "on")
