"""Service metadata and health routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from solarpulse_ai.config import Settings, get_settings

router = APIRouter(tags=["system"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]


class ServiceResponse(BaseModel):
    """Public service metadata."""

    name: str
    version: str
    environment: str
    documentation: str


class HealthResponse(BaseModel):
    """Service liveness response."""

    status: str


@router.get("/", response_model=ServiceResponse, status_code=status.HTTP_200_OK)
def get_service(settings: SettingsDependency) -> ServiceResponse:
    """Return service metadata and the API documentation location."""
    return ServiceResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        documentation="/docs",
    )


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health() -> HealthResponse:
    """Return the service liveness state."""
    return HealthResponse(status="healthy")
