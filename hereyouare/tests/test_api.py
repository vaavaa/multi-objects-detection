"""Тесты API hereyouare."""
import pytest
from httpx import AsyncClient, ASGITransport

# Импорт app после настройки окружения (Redis может быть недоступен)
import os
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def app():
    from main import app
    return app


@pytest.mark.asyncio
async def test_health(app):
    """GET /health возвращает 200 и status=ok."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_echo_get(app):
    """GET /echo с query message."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/echo", params={"message": "hello"})
    assert r.status_code == 200
    assert r.json() == {"echo": {"message": "hello"}}


@pytest.mark.asyncio
async def test_echo_post(app):
    """POST /echo с JSON body."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/echo", json={"key": "value"})
    assert r.status_code == 200
    assert r.json() == {"echo": {"key": "value"}}
