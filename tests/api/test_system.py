"""Tests for service metadata and health routes."""

from fastapi import status
from fastapi.testclient import TestClient


def test_get_service_returns_metadata(client: TestClient) -> None:
    """The root endpoint exposes stable service metadata."""

    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "name": "SolarPulse AI",
        "version": "0.1.0",
        "environment": "test",
        "documentation": "/docs",
    }


def test_get_health_returns_healthy_status(client: TestClient) -> None:
    """The health endpoint reports that the API process is live."""

    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}
