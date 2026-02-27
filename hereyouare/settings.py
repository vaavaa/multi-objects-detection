"""
Настройки сервиса детекции объектов.
При необходимости можно добавить чтение из переменных окружения (os.environ).
"""

import os
from typing import Any, Dict, List

# Ollama API
OLLAMA_URL = "http://ollama:11434/api/generate"  # в docker-compose это имя сервиса
MODEL = "qwen2.5vl:7b-q4_K_M"

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

# --- Очередь заданий (Redis) ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_RESULT_TTL_SEC = 3600  # время жизни результата задания в Redis (1 ч)

# --- FastAPI-YOLO (YOLO World): детекция по заданным классам ---
# Базовый URL эндпоинта base64 (без query). В Docker: http://fastapi-yolo:8001
YOLO_BASE64_URL = os.environ.get(
    "YOLO_BASE64_URL",
    "http://0.0.0.0:8001/api/v1/yworld/base64",
)
YOLO_IOU_THRESHOLD = 0.5
YOLO_SCORE_THRESHOLD = 0.2
YOLO_MAX_NUM_DETECTIONS = 5
YOLO_ONLY_BBOXS = True  # only_bboxs=true в query
YOLO_TIMEOUT_SEC = 60
# Классы по умолчанию, если не переданы и не получены из Qwen
YOLO_DEFAULT_CLASS_NAMES: List[str] = [
    "headphones",
    "keyboard",
    "mouse",
    "mousepad",
]
