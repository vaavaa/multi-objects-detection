FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENV SQLITE_PATH=/data/tasks.db
ENV MCP_TRANSPORT=http
ENV MCP_HTTP_HOST=0.0.0.0
ENV MCP_HTTP_PORT=8084
# По умолчанию совместимость с Open WebUI (Streamable HTTP + GET /mcp); см. комментарий в docker-compose.
ENV FASTMCP_JSON_RESPONSE=false
ENV FASTMCP_STATELESS_HTTP=false

RUN mkdir -p /data

VOLUME ["/data"]

EXPOSE 8084

CMD ["python", "-m", "app.server"]
