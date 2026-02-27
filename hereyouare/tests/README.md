# Тесты hereyouare

```bash
# Установка зависимостей (в т.ч. pytest, pytest-asyncio)
pip install -r requirements.txt

# Запуск (из корня hereyouare или DockerCore)
pytest hereyouare/tests/ -v

# Только тесты API (без Redis для health/echo)
pytest hereyouare/tests/test_api.py -v
```

Полные тесты с очередью заданий требуют запущенный Redis.
