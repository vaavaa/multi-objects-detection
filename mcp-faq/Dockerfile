FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY knowledge ./knowledge
COPY .env.example ./.env

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

CMD ["python", "-m", "app.server"]
