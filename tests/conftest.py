"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from solarpulse_ai.config import Settings, get_settings
from solarpulse_ai.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide an isolated FastAPI test client."""

    test_settings = Settings(environment="test")
    application = create_app(test_settings)
    application.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()
