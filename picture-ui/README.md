# Picture UI — OpenWebUI Tools (загрузка и анализ изображений)

Сервис инструментов для OpenWebUI: загрузка фотографий и анализ объектов.

## Возможности

- **POST /upload_photo_for_analysis** — загрузка изображения (JPEG, PNG, GIF, WebP, BMP); возвращает метаданные (размеры, формат) и заготовку под анализ объектов. Для полноценной детекции объектов можно подключить внешний сервис (например [HereYouAre](../hereyouare)).

## Запуск

```bash
# Локально
pip install -r requirements.txt
uvicorn main_pictures:app --reload --port 8083

# Через Docker (из корня DockerCore)
docker compose up -d picture-ui
```

Сервис слушает порт **8083**. OpenAPI: `http://localhost:8083/openapi.json`.

## Подключение в OpenWebUI

В **Settings → Connections → Tool Server** добавьте подключение типа OpenAPI:

- **URL:** `http://<host>:8083` (или `http://picture-ui:8083` внутри Docker-сети)
- **Path:** `openapi.json`
- **Auth:** none (или по необходимости)

После сохранения инструмент «Upload photo for analysis» будет доступен в чате.
