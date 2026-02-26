"""
Настройки сервиса детекции объектов.
При необходимости можно добавить чтение из переменных окружения (os.environ).
"""

from typing import Any, Dict

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
