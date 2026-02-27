# HereYouAre — Object Detection API

FastAPI-сервис детекции объектов на изображениях через **Ollama** (vision-модели, например qwen2.5-vl). Поддерживается асинхронная очередь заданий (Redis) и опционально уточнение боксов через **YOLO World** (FastAPI-YOLO).

## Возможности

- Загрузка изображения → запрос к Ollama (JSON с объектами) → опционально вызов YOLO по классам → ответ с координатами и метками.
- Синхронные эндпоинты: `POST /v1/detect`, `POST /v1/detect/image` (картинка с разметкой).
- Асинхронный поток: `POST /v1/detect/async` (возвращает `job_id`) → опрос `GET /v1/job/{job_id}` (4 раза в секунду) → версии **v1** (Qwen) и **v2** (YOLO), кеш по хешу изображения (xxhash).

## Зависимости

- Python 3.10+
- Ollama (локально или в Docker) с vision-моделью
- Redis (для очереди заданий и кеша)
- Опционально: сервис YOLO World (base64 API) для уточнения боксов

## Установка и запуск

```bash
pip install -r requirements.txt
# Запуск (OLLAMA_URL и REDIS_URL через переменные окружения или settings.py)
uvicorn main:app --host 0.0.0.0 --port 8082
```

В Docker сборка и запуск через `docker compose` из корня **DockerCore** (см. основной README).

## Конфигурация

Файл **settings.py** (при необходимости — переменные окружения):

- **Ollama:** `OLLAMA_URL`, `MODEL`, `OLLAMA_CONCURRENCY`, `PROMPT`, `FORMAT_SCHEMA`, таймаут, температура.
- **Изображение:** `IMAGE_MAX_SIDE`, `IMAGE_JPEG_QUALITY`.
- **Redis:** `REDIS_URL`, `JOB_RESULT_TTL_SEC`, `CACHE_TTL_SEC`.
- **YOLO:** `YOLO_BASE64_URL`, пороги, `YOLO_DEFAULT_CLASS_NAMES`.

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка живости сервиса |
| GET / POST | `/echo` | Эхо (для проверки доступности) |
| POST | `/v1/detect` | Детекция, ответ — JSON |
| POST | `/v1/detect/image` | Детекция, ответ — PNG с угловыми рамками |
| POST | `/v1/detect/async` | Асинхронная детекция, ответ — `job_id` |
| GET | `/v1/job/{job_id}` | Статус/результат задания (v1/v2, detections) |

Документация: http://localhost:8082/docs (Swagger), http://localhost:8082/redoc (ReDoc).

## Тесты

```bash
pip install -r requirements.txt
pytest tests/ -v
```

См. **tests/README.md**.
