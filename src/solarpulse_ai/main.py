"""FastAPI application factory and ASGI entry point."""

from fastapi import FastAPI

from solarpulse_ai.api.router import api_router
from solarpulse_ai.config import Settings, get_settings

APP_DESCRIPTION = (
    "API foundation for solar PV production forecasting and underperformance detection."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the SolarPulse AI FastAPI application."""
    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description=APP_DESCRIPTION,
    )
    application.include_router(api_router)
    return application


app = create_app()
