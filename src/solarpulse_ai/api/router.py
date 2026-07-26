"""Top-level API router composition."""

from fastapi import APIRouter

from solarpulse_ai.api.routes import system

api_router = APIRouter()
api_router.include_router(system.router)
