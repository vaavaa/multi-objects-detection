"""Pytest configuration. Запуск тестов: pytest (из корня hereyouare или DockerCore)."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("asyncio_mode", "auto")
