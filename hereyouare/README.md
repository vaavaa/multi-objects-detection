# HereYouAre — Object Detection API

FastAPI-сервис детекции объектов на изображениях через **Ollama** (vision-модели: Qwen2.5-VL, Qwen3-VL и др.). Поддерживается асинхронная очередь заданий (Redis), опциональное уточнение боксов через **YOLO World** и управление промптами/трейсинг через **Langfuse**.

## Возможности

- Загрузка изображения → запрос к Ollama (JSON с объектами) → опционально вызов YOLO по классам → ответ с координатами и метками.
- **Langfuse:** подгрузка промпта и конфига (температура, JSON-schema) по имени модели; трейсинг вызовов (span + generation) в Langfuse. При недоступности Langfuse используется fallback из `settings.py`.
- Синхронные эндпоинты: `POST /v1/detect`, `POST /v1/detect/image` (картинка с разметкой).
- Асинхронный поток: `POST /v1/detect/async` (возвращает `job_id`) → опрос `GET /v1/job/{job_id}` → версии **v1** (Qwen) и **v2** (YOLO), кеш по хешу изображения.

## Зависимости

- Python 3.10+
- **Ollama** (локально или в Docker) с vision-моделью (например `qwen2.5vl:7b-q4_K_M`, `qwen3-vl:8b-instruct-q8_0`).
- **Redis** — очередь заданий и опциональный кеш.
- **Langfuse** (опционально) — Prompt Management и observability; при отсутствии ключей или недоступности используется fallback.
- Опционально: сервис YOLO World (base64 API) для уточнения боксов.

## Установка и запуск

### Локально

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8082
```

Переменные окружения (или правки в `settings.py`): `OLLAMA_URL`, `OLLAMA_MODEL`, `REDIS_URL` и при необходимости `LANGFUSE_*`.

### Docker

Сборка и запуск через `docker compose` из корня репозитория **DockerCore**. Сервис `hereyouare` описан в `docker-compose.yaml`; нужны сервисы `ollama`, `redis`, при использовании Langfuse — `langfuse-web`.

## Конфигурация

### Переменные окружения (приоритет над `settings.py`)

| Переменная | Описание | Пример |
|------------|----------|--------|
| `OLLAMA_MODEL` | Модель Ollama; по имени выбирается промпт в Langfuse | `qwen2.5vl:7b-q4_K_M`, `qwen3-vl:8b-instruct-q8_0` |
| `REDIS_URL` | URL Redis для очереди и кеша | `redis://:secret@redis:6379/0` |
| `LANGFUSE_ENABLED` | Включить подгрузку промптов и трейсинг | `true` / `false` |
| `LANGFUSE_PUBLIC_KEY` | Ключ из Langfuse → Project Settings → API Keys | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | Секрет из того же проекта | `sk-lf-...` |
| `LANGFUSE_BASE_URL` | URL Langfuse (из контейнера: `http://langfuse-web:3000`) | — |
| `LANGFUSE_PROMPT_LABEL` | Метка версии промпта | `production`, `latest` |
| `USE_REDIS` | Включить очередь заданий | `true` / `false` |
| `USE_CACHE` | Кешировать результат по хешу изображения | `true` / `false` |
| `YOLO_BASE64_URL` | URL API YOLO World (base64) | по необходимости |

Остальное (таймауты, размер изображения, температура по умолчанию и т.д.) — в **settings.py**.

### Маппинг модели → промпт Langfuse

В коде (`langfuse_runtime.model_to_prompt_name`) задано:

- `qwen2.5vl` / `qwen2.5-vl` / `qwen2_5vl` → промпт **hereyouare.detect.qwen2_5vl**
- `qwen3vl` / `qwen3-vl` / `qwen3_vl` → промпт **hereyouare.detect.qwen3vl**
- иначе → **hereyouare.detect.qwen2_5vl**

В Langfuse нужно создать промпты с такими именами и выставить у версии label (например `production`). В конфиге промпта можно задать `ollama.temperature`, `ollama.format_schema`, `parser.mode` — они подхватываются вместо значений из `settings.py`.

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка живости сервиса |
| GET / POST | `/echo` | Эхо (проверка доступности) |
| POST | `/v1/detect` | Детекция, ответ — JSON (`image`, `detections`, `provider`, `latency_ms`) |
| POST | `/v1/detect/image` | Детекция, ответ — PNG с угловыми рамками и подписями |
| POST | `/v1/detect/async` | Асинхронная детекция, ответ — `job_id`; опционально `class_names` (строка через запятую) |
| GET | `/v1/job/{job_id}` | Статус и результат задания (`status`, `version` v1/v2, `detections`, `image`, `provider`) |
| POST | `/v1/cache/clear` | Очистка кеша и заданий в Redis (требуется Redis) |

Документация: http://localhost:8082/docs (Swagger), http://localhost:8082/redoc (ReDoc).

## Тесты

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Или из корня DockerCore: `pytest hereyouare/tests/ -v`. См. **tests/README.md**.

Тесты покрывают: health/echo, маппинг модели на имя промпта Langfuse, fallback при отключённом Langfuse, использование промпта из Langfuse в payload запроса к Ollama.
