# FastMCP FAQ Agent

Минимальный FAQ-агент на FastMCP.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
cp .env.example .env
python -m app.server
```

## Что умеет

- `faq_health()`
- `faq_search(question, limit=3)`
- `faq_answer(question)`

## Пример FAQ markdown

Каждый FAQ лежит отдельным `.md` файлом с YAML frontmatter.

## Что улучшать дальше

1. Добавить нормализацию русского/казахского текста
2. Добавить BM25 или embeddings
3. Добавить freshness-check
4. Добавить handoff на человека
5. Хранить FAQ не только в md, но и в CRM / Google Sheets / Postgres

## Как запускать

### Локально

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
cp .env.example .env
python -m app.server
```

### Docker

```bash
docker build -t fastmcp-faq-agent .
docker run --rm -p 8000:8000 fastmcp-faq-agent
```
